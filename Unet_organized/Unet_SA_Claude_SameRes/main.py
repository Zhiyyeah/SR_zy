import os

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
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
    pbar = tqdm(loader, desc="train", leave=False)
    for lr_img, hr_img, _ in pbar:
        lr_img = lr_img.to(device)
        hr_img = hr_img.to(device)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                pred = model(lr_img)
                loss = loss_fn(torch.clamp(pred, 0, 1), hr_img)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(lr_img)
            loss = loss_fn(torch.clamp(pred, 0, 1), hr_img)
            loss.backward()
            optimizer.step()
        # compute batch PSNR (training) for live feedback
        with torch.no_grad():
            tr_psnr = batch_psnr(torch.clamp(pred, 0, 1), hr_img)
        losses.append(loss.item())
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "psnr": f"{tr_psnr:.2f}"})
    return sum(losses) / max(1, len(losses))


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    psnrs = []
    for lr_img, hr_img, _ in tqdm(loader, desc="val", leave=False):
        lr_img = lr_img.to(device)
        hr_img = hr_img.to(device)
        pred = torch.clamp(model(lr_img), 0, 1)
        psnrs.append(batch_psnr(pred, hr_img))
    return sum(psnrs) / max(1, len(psnrs))


def main():
    """Train with embedded defaults. Edit here to change paths/hyperparams."""

    # Paths (5-band same-size TIFF pairs with LR_/HR_ prefixes)
    LR_DIR = r"D:\Py_Code\img_match\SR_Imagery\tif\LR"
    HR_DIR = r"D:\Py_Code\img_match\SR_Imagery\tif\HR"
    OUT_DIR = "Unet_organized/Unet_SA_Claude_SameRes/outputs/run_auto"

    # Hyperparameters
    EPOCHS = 100
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    PATCH_SIZE = 256
    NUM_WORKERS = 0
    LOSS_NAME = "charbonnier"  # or "l1", "l2"
    VAL_SPLIT = 0.1
    SEED = 42
    USE_AMP = True

    set_seed(SEED)
    device = get_device()
    os.makedirs(os.path.join(OUT_DIR, "models"), exist_ok=True)

    ds_full = PairTifDataset(lr_dir=LR_DIR, hr_dir=HR_DIR, size=PATCH_SIZE, augment=True, require_bands=5)
    train_ds, val_ds = split_dataset(ds_full, VAL_SPLIT)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=max(1, BATCH_SIZE // 2), shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    model = create_model(in_channels=5, out_channels=5).to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.99), weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    loss_fn = make_loss(LOSS_NAME)
    # Use new GradScaler API to avoid deprecation warning
    scaler = (torch.amp.GradScaler('cuda') if (USE_AMP and device.type == "cuda") else None)

    best_psnr = -1.0
    psnr_history = []
    log_dir = OUT_DIR
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "metrics.csv")
    # write header if not exists
    if not os.path.exists(log_path):
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("epoch,train_loss,val_psnr\n")
    for epoch in range(1, EPOCHS + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, loss_fn)
        val_psnr = validate(model, val_loader, device)
        scheduler.step()
        # record and show PSNR history (epoch-wise)
        psnr_history.append(val_psnr)
        # persist metrics row
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{tr_loss:.6f},{val_psnr:.6f}\n")
        # console output with running PSNR history (last 10)
        hist_tail = ", ".join(f"{p:.2f}" for p in psnr_history[-10:])
        print(f"Epoch {epoch:03d} | train loss {tr_loss:.4f} | val PSNR {val_psnr:.3f} | PSNR hist(last10): [{hist_tail}]")

        # save per-epoch pth
        save_checkpoint(model, os.path.join(OUT_DIR, "models", f"epoch_{epoch:03d}.pth"))

        # track best
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            save_checkpoint(model, os.path.join(OUT_DIR, "models", "best.pth"))

    # save last
    save_checkpoint(model, os.path.join(OUT_DIR, "models", "last.pth"))


if __name__ == "__main__":
    main()
