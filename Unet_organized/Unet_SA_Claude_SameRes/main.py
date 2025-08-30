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
from .losses import make_loss


def split_dataset(ds: PairTifDataset, val_split: float):
    n = len(ds)
    val_n = int(n * val_split)
    train_n = n - val_n
    return random_split(ds, [train_n, val_n], generator=torch.Generator().manual_seed(123))


def train_one_epoch(model, loader, optimizer, scaler, device, loss_fn):
    model.train()
    losses = []
    psnrs = []
    pbar = tqdm(loader, desc="train", leave=False)
    for lr_img, hr_img, _ in pbar:
        lr_img = lr_img.to(device)
        hr_img = hr_img.to(device)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                pred = model(lr_img)
                loss = loss_fn(pred, hr_img)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(lr_img)
            loss = loss_fn(pred, hr_img)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            tr_psnr = batch_psnr(pred, hr_img)
        losses.append(loss.item())
        psnrs.append(tr_psnr)
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "psnr": f"{tr_psnr:.2f}"})
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
        pred = model(lr_img)
        psnrs.append(batch_psnr(pred, hr_img))
        losses.append(loss_fn(pred, hr_img).item())
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
    BETA_SSIM = 10.0
    SSIM_DATA_RANGE = None        # 若输入已归一化到[0,1]，可设为 1.0；否则保持 None 自动估计
    VAL_SPLIT = 0.1
    SEED = 42
    USE_AMP = True

    set_seed(SEED)
    device = get_device()
    # 控制台打印：设备与损失函数类型
    print(f"Using device: {device} | CUDA available: {torch.cuda.is_available()}")
    if LOSS_NAME.lower() == "charbonnier+ssim":
        print(f"Loss: {LOSS_NAME} (alpha={ALPHA_CHARB}, beta={BETA_SSIM}, data_range={SSIM_DATA_RANGE})")
    else:
        print(f"Loss: {LOSS_NAME}")
    os.makedirs(os.path.join(OUT_DIR, "models"), exist_ok=True)

    ds_full = PairTifDataset(lr_dir=LR_DIR, hr_dir=HR_DIR, require_bands=5)
    train_ds, val_ds = split_dataset(ds_full, VAL_SPLIT)
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
            f"Epoch {epoch:03d} | train loss {tr_loss:.4f} | val loss {val_loss:.4f}{lr_str} | "
            f"train PSNR {tr_psnr:.3f} | val PSNR {val_psnr:.3f} | PSNR hist(last10): [{hist_tail}]"
        )

        # Save checkpoints
        save_checkpoint(model, os.path.join(OUT_DIR, "models", f"epoch_{epoch:03d}.pth"))
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            save_checkpoint(model, os.path.join(OUT_DIR, "models", "best.pth"))

    save_checkpoint(model, os.path.join(OUT_DIR, "models", "last.pth"))


if __name__ == "__main__":
    main()
