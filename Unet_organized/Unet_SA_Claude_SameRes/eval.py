import os
from typing import Tuple

import numpy as np
import torch
from tqdm import tqdm

from .data_loader import PairTifDataset
from .model_attention import create_model
from .utils import get_device, load_checkpoint
from .metrics import psnr as psnr_fn


def _to_uint8(img01: np.ndarray) -> np.ndarray:
    """Assumes img01 is float32 in [0,1]. Converts to uint8 [0,255]."""
    return (np.clip(img01, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def make_sr_hr_grid(sr_chw: np.ndarray, hr_chw: np.ndarray) -> np.ndarray:
    """Create a (5 rows x 2 cols) mosaic for SR/HR comparison.

    - Each row is a band (grayscale). Left: SR, Right: HR.
    - Assumes sr_chw, hr_chw are (C,H,W) in [0,1] with C>=5.
    - Returns uint8 H_out x W_out (single-channel) image.
    """
    c, h, w = sr_chw.shape
    rows = []
    bands = min(5, c)
    for i in range(bands):
        sr = _to_uint8(sr_chw[i])
        hr = _to_uint8(hr_chw[i])
        row = np.concatenate([sr, hr], axis=1)  # (H, 2W)
        rows.append(row)
    mosaic = np.concatenate(rows, axis=0)  # (5H, 2W)
    return mosaic


def save_png_gray(path: str, arr2d_uint8: np.ndarray):
    from PIL import Image

    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(arr2d_uint8, mode="L").save(path)


@torch.no_grad()
def main():
    # --------------------
    # Config (edit as needed)
    # --------------------
    LR_DIR = r"D:\Py_Code\img_match\SR_Imagery\tif\LR"
    HR_DIR = r"D:\Py_Code\img_match\SR_Imagery\tif\HR"
    WEIGHTS = "Unet_organized/Unet_SA_Claude_SameRes/outputs/run_auto/models/best.pth"
    SAVE_DIR = "Unet_organized/Unet_SA_Claude_SameRes/outputs/run_auto/eval"
    PATCH_SIZE = 256

    device = get_device()
    model = create_model(in_channels=5, out_channels=5).to(device)
    load_checkpoint(model, WEIGHTS, map_location=device)
    model.eval()

    ds = PairTifDataset(lr_dir=LR_DIR, hr_dir=HR_DIR, size=PATCH_SIZE, augment=False, require_bands=5)

    psnrs = []
    for i in tqdm(range(len(ds)), desc="eval"):
        lr_t, hr_t, name = ds[i]
        lr_t = lr_t.unsqueeze(0).to(device)
        hr_t = hr_t.unsqueeze(0).to(device)

        sr = torch.clamp(model(lr_t), 0, 1)
        psnrs.append(psnr_fn(sr, hr_t))

        # to numpy CHW for grid
        sr_np = sr.squeeze(0).detach().cpu().numpy()
        hr_np = hr_t.squeeze(0).detach().cpu().numpy()
        grid = make_sr_hr_grid(sr_np, hr_np)

        # filename: remove LR_/HR_ prefix pattern if present
        base = os.path.splitext(name)[0]
        low = base.lower()
        if low.startswith("lr_"):
            base = base[len("LR_"):]
        out_png = os.path.join(SAVE_DIR, f"{base}_compare.png")
        save_png_gray(out_png, grid)

    if psnrs:
        mean_psnr = sum(psnrs) / len(psnrs)
        print(f"Mean PSNR: {mean_psnr:.3f} dB over {len(psnrs)} samples")


if __name__ == "__main__":
    main()

