import os
import math

import torch
from torch.optim import AdamW
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
LOG_EVERY_N = 100          # 训练过程中每多少个 batch 追加一条详细日志


def train_one_epoch(model, loader, optimizer, scaler, device, loss_fn):
    """单个 epoch 的训练过程。

    步骤:
      1. 取出一个 batch (包含 LR / HR 图像)
      2. 前向推理 (支持 AMP)
      3. 计算损失 (支持 Charbonnier 或 Charbonnier+SSIM 组合)
      4. 反向传播 + 梯度裁剪 + 参数更新
      5. 统计指标 (loss / PSNR / 组件 loss)
      6. 进度条与可选详细日志输出
    """
    model.train()
    losses = []
    psnrs = []
    pbar = tqdm(loader, desc="train", leave=False)
    for batch_idx, batch in enumerate(pbar):
        # 1. 取数据
        if len(batch) == 3:
            lr_img, hr_img, name = batch
        else:
            lr_img, hr_img = batch
            name = None
        lr_img = torch.nan_to_num(lr_img.to(device), nan=0.0, posinf=1e6, neginf=-1e6)
        hr_img = torch.nan_to_num(hr_img.to(device), nan=0.0, posinf=1e6, neginf=-1e6)

        optimizer.zero_grad(set_to_none=True)
        # 仅用于日志展示的中间值（不是参与反传的变量）
        charb_val = None          # Charbonnier 原始损失值
        ssim_loss_val = None      # 加权后的 SSIM 损失 (beta * (1 - ssim))
        ssim_val = None           # 还原的 SSIM 指标 (1 - (1-ssim))

        # 2. 前向 + 损失
        if scaler is not None:  # 使用 AMP
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                pred = model(lr_img)
            pred = torch.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=-1e6)
            if isinstance(loss_fn, CombinedLoss):
                # 关闭 autocast 以避免低精度导致数值问题：单独在 FP32 下计算 loss
                with torch.autocast(device_type=device.type, enabled=False):
                    charb = loss_fn.charb(pred.float(), hr_img.float())
                    ssimc = loss_fn.ssim_loss(pred.float(), hr_img.float())  # 1 - ssim
                    loss = loss_fn.alpha * charb + loss_fn.beta * ssimc
                charb_val = float(charb.detach())
                ssim_loss_raw = float(ssimc.detach())              # 未加权 (1-ssim)
                ssim_loss_val = loss_fn.beta * ssim_loss_raw       # 加权后
                ssim_val = 1.0 - ssim_loss_raw                     # ssim 原值
            else:
                # 单一损失同样在 FP32 计算
                with torch.autocast(device_type=device.type, enabled=False):
                    loss = loss_fn(pred.float(), hr_img.float())
            if not torch.isfinite(loss):
                # 若出现 NaN/Inf，跳过该 batch 防止污染梯度
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
            # 不使用 AMP 的纯 FP32 分支
            pred = model(lr_img)
            pred = torch.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=-1e6)
            if isinstance(loss_fn, CombinedLoss):
                charb = loss_fn.charb(pred.float(), hr_img.float())
                ssimc = loss_fn.ssim_loss(pred.float(), hr_img.float())  # 1 - ssim
                loss = loss_fn.alpha * charb + loss_fn.beta * ssimc
                charb_val = float(charb.detach())
                ssim_loss_raw = float(ssimc.detach())              # 未加权 (1-ssim)
                ssim_loss_val = loss_fn.beta * ssim_loss_raw       # 加权后
                ssim_val = 1.0 - ssim_loss_raw                     # ssim 原值
            else:
                loss = loss_fn(pred.float(), hr_img.float())
            if not torch.isfinite(loss):
                continue
            loss.backward()
            try:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            except Exception:
                pass
            optimizer.step()

        # 3. 统计指标
        with torch.no_grad():
            tr_psnr = batch_psnr(pred, hr_img)
        losses.append(loss.item())
        psnrs.append(tr_psnr)

        # 4. 进度条后缀
        if isinstance(loss_fn, CombinedLoss) and charb_val is not None and ssim_loss_val is not None:
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "charb": f"{charb_val:.4f}",
                "ssim_loss": f"{ssim_loss_val:.4f}",
                "ssim": f"{ssim_val:.4f}",
                "psnr": f"{tr_psnr:.2f}"
            })
        else:
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "psnr": f"{tr_psnr:.2f}"})

        # 5. 详细日志
        if VERBOSE and (batch_idx % LOG_EVERY_N == 0):
            print(f"[batch {batch_idx}/{len(loader)}]")
            print(f"pred[min,max]=[{pred.min().item():.3g},{pred.max().item():.3g}]")
            print(f"target[min,max]=[{hr_img.min().item():.3g},{hr_img.max().item():.3g}]")
            if isinstance(loss_fn, CombinedLoss) and charb_val is not None and ssim_loss_val is not None:
                print(
                    f"loss=(total={loss.item():.4f}, charbonnier={charb_val:.4f}, "
                    f"ssim_loss={ssim_loss_val:.4f}, ssim={ssim_val:.4f})"
                )
            else:
                print(f"loss={loss.item():.4f}")
            print(f"psnr={tr_psnr:.2f}")

    avg_loss = sum(losses) / max(1, len(losses))
    avg_psnr = sum(psnrs) / max(1, len(psnrs))
    return avg_loss, avg_psnr


@torch.no_grad()
def validate(model, loader, device, loss_fn):
    """验证流程：不反传，只统计平均 loss 与 PSNR。"""
    model.eval()
    psnrs = []
    losses = []
    for lr_img, hr_img, _ in tqdm(loader, desc="val", leave=False):
        lr_img = torch.nan_to_num(lr_img.to(device), nan=0.0, posinf=1e6, neginf=-1e6)
        hr_img = torch.nan_to_num(hr_img.to(device), nan=0.0, posinf=1e6, neginf=-1e6)
        pred = model(lr_img)
        psnrs.append(batch_psnr(pred, hr_img))
        if isinstance(loss_fn, CombinedLoss):
            # 验证阶段同样复用组合损失公式（注意：这里未单独记录组件）
            charb = loss_fn.charb(pred.float(), hr_img.float())
            ssimc = loss_fn.ssim_loss(pred.float(), hr_img.float())
            loss = loss_fn.alpha * charb + loss_fn.beta * ssimc
        else:
            loss = loss_fn(pred.float(), hr_img.float())
        if torch.isfinite(loss):
            losses.append(loss.item())
    avg_psnr = sum(psnrs) / max(1, len(psnrs))
    avg_loss = sum(losses) / max(1, len(losses))
    return avg_psnr, avg_loss


def main():
    """程序入口：构建数据集 / 模型 / 优化器，执行纯余弦学习率衰减的训练循环，并保存指标与权重。"""
    # Paths
    LR_DIR = r"D:\Py_Code\img_match\SR_Imagery\tif\LR"
    HR_DIR = r"D:\Py_Code\img_match\SR_Imagery\tif\HR"
    OUT_DIR = "Unet_organized/Unet_SA_Claude_SameRes/outputs/run_auto"

    # Hyperparameters
    EPOCHS = 100
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    NUM_WORKERS = 0
    LOSS_NAME = "charbonnier"  # or "charbonnier+ssim"
    ALPHA_CHARB = 1.0
    # SSIM 权重分段：前 SWITCH_EPOCH 个 epoch 使用较小值，之后使用较大值
    BETA_SSIM_FIRST = 5.0
    BETA_SSIM_LATER = 10.0
    SWITCH_EPOCH = 5  # epoch 从 1 开始计数，<= SWITCH_EPOCH 用 FIRST
    SSIM_DATA_RANGE = None
    VAL_SPLIT = 0.3
    SEED = 42
    USE_AMP = True

    set_seed(SEED)
    device = get_device()
    print("================ 运行配置 ================")
    print(f"设备: {device} | CUDA 可用: {torch.cuda.is_available()}")
    print(f"数据路径: LR_DIR={LR_DIR} | HR_DIR={HR_DIR}")
    print(f"输出目录: {OUT_DIR}")
    print(f"数据集划分: 验证集比例={VAL_SPLIT:.2f} | 全局种子(SEED)={SEED}")
    print(f"Batch 大小: {BATCH_SIZE} | Epoch 数: {EPOCHS}")
    print(f"学习率: {LEARNING_RATE} | 优化器: AdamW | Scheduler: warmup+cosine (warmup=3)")
    print(f"AMP: {'启用' if USE_AMP else '关闭'}")
    print(f"损失函数: {LOSS_NAME}")
    print(f"详细日志: VERBOSE={VERBOSE} | LOG_EVERY_N={LOG_EVERY_N}")
    print("==========================================")
    os.makedirs(os.path.join(OUT_DIR, "models"), exist_ok=True)

    ds_full = PairTifDataset(lr_dir=LR_DIR, hr_dir=HR_DIR, require_bands=5)
    print(f"完整数据集样本数: {len(ds_full)}")

    from torch.utils.data import random_split
    train_ds, val_ds = random_split(ds_full, [int(len(ds_full)*(1-VAL_SPLIT)), len(ds_full)-int(len(ds_full)*(1-VAL_SPLIT))],
                                    generator=torch.Generator().manual_seed(SEED))
    print(f"训练集样本数: {len(train_ds)} | 验证集样本数: {len(val_ds)}")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=max(1, BATCH_SIZE//2), shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    model = create_model(in_channels=5, out_channels=5).to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9,0.99), weight_decay=1e-4)
    # 学习率调度：3 epoch 线性 warmup + 余弦衰减
    WARMUP_EPOCHS = 3
    MIN_LR_RATIO = 0.05  # 余弦阶段最终相对最小学习率 (base_lr * ratio)
    # 初始以第一阶段权重创建
    loss_fn = make_loss(LOSS_NAME, alpha=ALPHA_CHARB, beta=BETA_SSIM_FIRST, ssim_data_range=SSIM_DATA_RANGE)
    scaler = (torch.amp.GradScaler('cuda') if (USE_AMP and device.type == 'cuda') else None)

    best_psnr = -1.0
    log_dir = OUT_DIR
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, 'metrics.csv')
    if not os.path.exists(log_path):
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write('epoch,train_loss,val_loss,train_psnr,val_psnr\n')

    # 指标历史
    psnr_history = []

    def compute_lr(epoch_idx: int):
        """3 epoch 线性 warmup + 余弦衰减。

        Warmup: lr = base_lr * epoch / WARMUP_EPOCHS (epoch ∈ [1, WARMUP_EPOCHS])
        Cosine: lr = min_lr + (base_lr - min_lr)*0.5*(1+cos(pi * progress))
                progress ∈ [0,1]
        """
        if epoch_idx <= WARMUP_EPOCHS:
            return LEARNING_RATE * epoch_idx / max(1, WARMUP_EPOCHS)
        # 余弦阶段
        progress = (epoch_idx - WARMUP_EPOCHS) / max(1, (EPOCHS - WARMUP_EPOCHS))
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        min_lr = LEARNING_RATE * MIN_LR_RATIO
        return min_lr + (LEARNING_RATE - min_lr) * cosine

    for epoch in range(1, EPOCHS+1):  # epoch 从 1 开始
        # 更新学习率（warmup+cosine）
        new_lr = compute_lr(epoch)
        for g in optimizer.param_groups:
            g['lr'] = new_lr
        # 动态调整 SSIM 权重
        if isinstance(loss_fn, CombinedLoss):
            loss_fn.beta = (BETA_SSIM_FIRST if epoch <= SWITCH_EPOCH else BETA_SSIM_LATER)
        print(f"\n----- 开始 Epoch {epoch}/{EPOCHS} (lr={new_lr:.2e}, beta_ssim={getattr(loss_fn,'beta','-')}) -----")
        tr_loss, tr_psnr = train_one_epoch(model, train_loader, optimizer, scaler, device, loss_fn)
        val_psnr, val_loss = validate(model, val_loader, device, loss_fn)
        psnr_history.append(val_psnr)
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(f"{epoch},{tr_loss:.6f},{val_loss:.6f},{tr_psnr:.6f},{val_psnr:.6f}\n")
        cur_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:03d} 完成 | train loss={tr_loss:.4f} | val loss={val_loss:.4f} | train PSNR={tr_psnr:.3f} | val PSNR={val_psnr:.3f} | lr={cur_lr:.2e}")
        save_checkpoint(model, os.path.join(OUT_DIR, 'models', f'epoch_{epoch:03d}.pth'))
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            save_checkpoint(model, os.path.join(OUT_DIR, 'models', 'best.pth'))
    save_checkpoint(model, os.path.join(OUT_DIR, 'models', 'last.pth'))


if __name__ == "__main__":
    main()