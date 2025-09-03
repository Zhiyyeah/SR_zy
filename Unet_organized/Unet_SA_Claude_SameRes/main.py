import os

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm

from .data_loader import PairTifDataset
from .model_attention import create_model
from .metrics import batch_psnr
from .utils import set_seed, get_device, save_checkpoint
from .losses import make_loss, CombinedLoss


def split_dataset(ds: PairTifDataset, val_split: float, seed: int):
    n = len(ds)
    val_n = int(n * val_split)
    train_n = n - val_n
    return random_split(ds, [train_n, val_n], generator=torch.Generator().manual_seed(seed))


def _is_finite_tensor(x: torch.Tensor) -> bool:
    return torch.isfinite(x).all().item()


VERBOSE = True            # 是否打印详细信息
LOG_EVERY_N = 50          # 训练过程中每多少个 batch 追加一条详细日志


def train_one_epoch(model, loader, optimizer, scaler, device, loss_fn):
    model.train()
    losses = []
    psnrs = []
    pbar = tqdm(loader, desc="train", leave=False)
    for batch_idx, batch in enumerate(pbar):
        # 解包（兼容未来可能不同返回格式）
        if len(batch) == 3:
            lr_img, hr_img, name = batch
        else:
            lr_img, hr_img = batch
            name = None
        lr_img = lr_img.to(device)
        hr_img = hr_img.to(device)
        # 替换输入中的 NaN/Inf，避免传播
        lr_img = torch.nan_to_num(lr_img, nan=0.0, posinf=1e6, neginf=-1e6)
        hr_img = torch.nan_to_num(hr_img, nan=0.0, posinf=1e6, neginf=-1e6)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            # Use AMP for model forward only
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                pred = model(lr_img)
            pred = torch.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=-1e6)
            # Compute losses with autocast disabled (float32) to avoid FP16 instability
            if isinstance(loss_fn, CombinedLoss):
                with torch.autocast(device_type=device.type, enabled=False):
                    charb = loss_fn.charb(pred.float(), hr_img.float())
                    ssimc = loss_fn.ssim_loss(pred.float(), hr_img.float())
                    loss = loss_fn.alpha * charb + loss_fn.beta * ssimc
            else:
                with torch.autocast(device_type=device.type, enabled=False):
                    loss = loss_fn(pred.float(), hr_img.float())
            # 数值稳定性检查：在反传和step之前拦截，避免污染权重
            if not torch.isfinite(loss):
                try:
                    c = charb.item() if 'charb' in locals() else float('nan')
                    s = ssimc.item() if 'ssimc' in locals() else float('nan')
                    pmax = float(pred.max().detach().cpu()); pmin = float(pred.min().detach().cpu())
                    tmax = float(hr_img.max().detach().cpu()); tmin = float(hr_img.min().detach().cpu())
                    print(f"[warn] loss non-finite; skip. charb={c:.3e} ssimc={s:.3e} pred[min,max]=[{pmin:.3e},{pmax:.3e}] target[min,max]=[{tmin:.3e},{tmax:.3e}]")
                except Exception:
                    print("[warn] loss is non-finite; skipping batch. Enable SSIM_DATA_RANGE or reduce BETA_SSIM if frequent.")
                continue
            scaler.scale(loss).backward()
            try:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            except Exception:
                pass
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(lr_img)
            pred = torch.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=-1e6)
            if isinstance(loss_fn, CombinedLoss):
                charb = loss_fn.charb(pred.float(), hr_img.float())
                ssimc = loss_fn.ssim_loss(pred.float(), hr_img.float())
                loss = loss_fn.alpha * charb + loss_fn.beta * ssimc
            else:
                loss = loss_fn(pred.float(), hr_img.float())
            if not torch.isfinite(loss):
                try:
                    c = charb.item() if 'charb' in locals() else float('nan')
                    s = ssimc.item() if 'ssimc' in locals() else float('nan')
                    pmax = float(pred.max().detach().cpu()); pmin = float(pred.min().detach().cpu())
                    tmax = float(hr_img.max().detach().cpu()); tmin = float(hr_img.min().detach().cpu())
                    print(f"[warn] loss non-finite; skip. charb={c:.3e} ssimc={s:.3e} pred[min,max]=[{pmin:.3e},{pmax:.3e}] target[min,max]=[{tmin:.3e},{tmax:.3e}]")
                except Exception:
                    print("[warn] loss is non-finite; skipping batch. Enable SSIM_DATA_RANGE or reduce BETA_SSIM if frequent.")
                continue
            loss.backward()
            try:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            except Exception:
                pass
            optimizer.step()
        # 到此处 loss 已通过数值检查
        with torch.no_grad():
            tr_psnr = batch_psnr(pred, hr_img)
        losses.append(loss.item())
        psnrs.append(tr_psnr)
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "psnr": f"{tr_psnr:.2f}"})

        if VERBOSE and (batch_idx % LOG_EVERY_N == 0):
            try:
                name_str = name if name is not None else "(no-name)"
                print(f"[batch {batch_idx}/{len(loader)}]")
                # print(f"name={name_str}")
                print(f"pred[min,max]=[{pred.min().item():.3e},{pred.max().item():.3e}]")
                print(f"target[min,max]=[{hr_img.min().item():.3e},{hr_img.max().item():.3e}]")
                print(f"loss={loss.item():.4f}")
                print(f"psnr={tr_psnr:.2f}")
            except Exception:
                pass
    avg_loss = sum(losses) / max(1, len(losses))
    avg_psnr = sum(psnrs) / max(1, len(psnrs))
    return avg_loss, avg_psnr


@torch.no_grad()
def validate(model, loader, device, loss_fn):
    model.eval()
    psnrs = []
    losses = []
    for lr_img, hr_img, _ in tqdm(loader, desc="val", leave=False):
        lr_img = lr_img.to(device)
        hr_img = hr_img.to(device)
        lr_img = torch.nan_to_num(lr_img, nan=0.0, posinf=1e6, neginf=-1e6)
        hr_img = torch.nan_to_num(hr_img, nan=0.0, posinf=1e6, neginf=-1e6)
        pred = model(lr_img)
        psnrs.append(batch_psnr(pred, hr_img))
        if isinstance(loss_fn, CombinedLoss):
            charb = loss_fn.charb(pred, hr_img)
            ssimc = loss_fn.ssim_loss(pred, hr_img)
            loss = loss_fn.alpha * charb + loss_fn.beta * ssimc
        else:
            loss = loss_fn(pred, hr_img)
        if torch.isfinite(loss):
            losses.append(loss.item())
        else:
            print("[warn] val loss non-finite (ignored)")
    avg_psnr = sum(psnrs) / max(1, len(psnrs))
    avg_loss = sum(losses) / max(1, len(losses))
    return avg_psnr, avg_loss


def main():
    """No-args training entry; edit paths/hparams below if needed."""

    # Paths
    LR_DIR = r"D:\Py_Code\img_match\SR_Imagery\tif\LR"
    HR_DIR = r"D:\Py_Code\img_match\SR_Imagery\tif\HR"
    OUT_DIR = "Unet_organized/Unet_SA_Claude_SameRes/outputs/run_auto"

    # Hyperparameters
    EPOCHS = 100
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-4
    NUM_WORKERS = 0
    LOSS_NAME = "charbonnier"  # 可选："charbonnier" 或 "charbonnier+ssim"
    # 组合损失权重：loss = alpha * charbonnier + beta * (1 - SSIM)
    ALPHA_CHARB = 1.0
    # 建议 beta ∈ [0.02, 0.2]；100 过大易不稳定
    BETA_SSIM = 10.0
    SSIM_DATA_RANGE = None        # 若输入已归一化到[0,1]，可设为 1.0；否则保持 None 自动估计
    VAL_SPLIT = 0.3
    SEED = 42
    USE_AMP = True

    set_seed(SEED)
    device = get_device()
    # ===== 基本运行配置输出 =====
    print("================ 运行配置 ================")
    print(f"设备: {device} | CUDA 可用: {torch.cuda.is_available()}")
    print(f"数据路径: LR_DIR={LR_DIR} | HR_DIR={HR_DIR}")
    print(f"输出目录: {OUT_DIR}")
    print(f"数据集划分: 验证集比例={VAL_SPLIT:.2f} | | 全局种子(SEED)={SEED}")
    print(f"Batch 大小: {BATCH_SIZE} | Epoch 数: {EPOCHS}")
    print(f"学习率: {LEARNING_RATE} | 优化器: AdamW | Scheduler: ReduceLROnPlateau(patience=3,factor=0.5)")
    print(f"AMP: {'启用' if USE_AMP else '关闭'}")
    if LOSS_NAME.lower() == "charbonnier+ssim":
        print(f"损失函数: {LOSS_NAME} (alpha={ALPHA_CHARB}, beta={BETA_SSIM}, ssim_data_range={SSIM_DATA_RANGE})")
    else:
        print(f"损失函数: {LOSS_NAME}")
    print(f"详细日志: VERBOSE={VERBOSE} | LOG_EVERY_N={LOG_EVERY_N}")
    print("==========================================")
    os.makedirs(os.path.join(OUT_DIR, "models"), exist_ok=True)

    ds_full = PairTifDataset(lr_dir=LR_DIR, hr_dir=HR_DIR, require_bands=5)
    print(f"完整数据集样本数: {len(ds_full)}")
    # 展示前 3 个原始文件名（若存在）
    try:
        preview = [os.path.basename(p[0]) for p in ds_full.files[:3]]
        print("样本文件预览(前3个 LR):", preview)
    except Exception:
        pass

    train_ds, val_ds = split_dataset(ds_full, VAL_SPLIT, seed=SEED)
    print(f"训练集样本数: {len(train_ds)} | 验证集样本数: {len(val_ds)}")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=max(1, BATCH_SIZE // 2), shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    model = create_model(in_channels=5, out_channels=5).to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.99), weight_decay=1e-4)
    # LR scheduler: ReduceLROnPlateau lowers LR when val loss plateaus
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5,  threshold=0.01,patience=3)
    loss_fn = make_loss(
        LOSS_NAME,
        alpha=ALPHA_CHARB,
        beta=BETA_SSIM,
        ssim_data_range=SSIM_DATA_RANGE,
    )
    scaler = (torch.amp.GradScaler('cuda') if (USE_AMP and device.type == "cuda") else None)

    best_psnr = -1.0
    psnr_history = []
    log_dir = OUT_DIR
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "metrics.csv")
    # CSV header with Train/Val losses and PSNRs
    if not os.path.exists(log_path):
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("epoch,train_loss,val_loss,train_psnr,val_psnr\n")

    for epoch in range(1, EPOCHS + 1):
        print(f"\n----- 开始 Epoch {epoch}/{EPOCHS} -----")
        tr_loss, tr_psnr = train_one_epoch(model, train_loader, optimizer, scaler, device, loss_fn)
        val_psnr, val_loss = validate(model, val_loader, device, loss_fn)
        # Step scheduler with validation loss (plateau tracking)
        scheduler.step(val_loss)

        psnr_history.append(val_psnr)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{tr_loss:.6f},{val_loss:.6f},{tr_psnr:.6f},{val_psnr:.6f}\n")

        hist_tail = ", ".join(f"{p:.2f}" for p in psnr_history[-10:])
        cur_lr = optimizer.param_groups[0].get("lr", None)
        lr_str = f" | lr {cur_lr:.2e}" if cur_lr is not None else ""
        print(
            f"Epoch {epoch:03d} 完成 | train loss={tr_loss:.4f} | val loss={val_loss:.4f}{lr_str} | "
            f"train PSNR={tr_psnr:.3f} | val PSNR={val_psnr:.3f} | 最近10轮PSNR: [{hist_tail}]"
        )

        # Save checkpoints
        save_checkpoint(model, os.path.join(OUT_DIR, "models", f"epoch_{epoch:03d}.pth"))
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            save_checkpoint(model, os.path.join(OUT_DIR, "models", "best.pth"))

    save_checkpoint(model, os.path.join(OUT_DIR, "models", "last.pth"))


if __name__ == "__main__":
    main()
