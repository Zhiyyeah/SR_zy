import os
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import random_split
from tqdm import tqdm

from data_loader import PairTifDataset
from model import create_model
from metrics import batch_psnr
from utils import set_seed, get_device, save_checkpoint
from losses import make_loss


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
        losses.append(loss.item())
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
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
    """Train the model with built-in defaults (no CLI parameters needed).

    Edit the Config section below to change paths and hyperparameters.
    """

    # --------------------
    # Config (edit as needed)
    # --------------------
    LR_DIR = r"D:\Py_Code\img_match\SR_Imagery\tif\LR"  # Low-res TIFF folder (5 bands)
    HR_DIR = r"D:\Py_Code\img_match\SR_Imagery\tif\HR"  # High-res TIFF folder (5 bands), same filenames
    OUT_DIR = "UNet_SA_SameRes_TIF/outputs/run_auto"  # Output folder for checkpoints and logs

    EPOCHS = 50               # training epochs
    BATCH_SIZE = 8            # training batch size
    LEARNING_RATE = 1e-3      # optimizer lr
    PATCH_SIZE = 256          # crop/pad size; model preserves spatial size
    NUM_WORKERS = 0           # set >0 if dataloader can use subprocesses
    LOSS_NAME = "charbonnier" # one of: "l1", "l2", "charbonnier"
    VAL_SPLIT = 0.1           # fraction of samples for validation
    SEED = 42                 # reproducibility
    USE_AMP = True            # mixed precision on GPU

    # --------------------
    # Setup
    # --------------------
    set_seed(SEED)
    device = get_device()
    os.makedirs(OUT_DIR, exist_ok=True)

    # --------------------
    # Dataset and loaders
    # --------------------
    ds_full = PairTifDataset(
        lr_dir=LR_DIR,
        hr_dir=HR_DIR,
        size=PATCH_SIZE,
        augment=True,
        require_bands=5,
    )
    if len(ds_full) == 0:
        raise RuntimeError("No training samples found. Check LR/HR directories and filenames.")

    train_ds, val_ds = split_dataset(ds_full, VAL_SPLIT)
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=max(1, BATCH_SIZE // 2),
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # --------------------
    # Model / Optim / Loss
    # --------------------
    model = create_model(in_channels=5, out_channels=5).to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.99), weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    loss_fn = make_loss(LOSS_NAME)
    scaler = torch.cuda.amp.GradScaler() if (USE_AMP and device.type == "cuda") else None

    # --------------------
    # Training loop
    # --------------------
    best_psnr = -1.0
    for epoch in range(1, EPOCHS + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, loss_fn)
        val_psnr = validate(model, val_loader, device)
        scheduler.step()
        print(f"Epoch {epoch:03d} | train loss {tr_loss:.4f} | val PSNR {val_psnr:.3f}")
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            save_checkpoint(model, os.path.join(OUT_DIR, "best.ckpt"))

    # Save last checkpoint
    save_checkpoint(model, os.path.join(OUT_DIR, "last.ckpt"))


if __name__ == "__main__":
    # Run training with built-in defaults. Adjust Config in main() if needed.
    main()
