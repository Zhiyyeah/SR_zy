import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .data_loader import PairTifDataset
from .model import create_model
from .utils import get_device, load_checkpoint


def _write_tif(path: str, arr_chw: np.ndarray):
    try:
        import tifffile as tiff

        # ensure HWC for save
        hwc = arr_chw.transpose(1, 2, 0)
        tiff.imwrite(path, (hwc * 65535.0).clip(0, 65535).astype(np.uint16))
    except Exception:
        from PIL import Image

        hwc = (arr_chw.transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
        # Save first 3 bands if PIL can't handle >3 channels
        if hwc.shape[2] >= 3:
            Image.fromarray(hwc[:, :, :3]).save(path)
        else:
            Image.fromarray(hwc[:, :, 0]).save(path)


@torch.no_grad()
def main():
    """Export predictions as TIFFs using built-in defaults (no CLI parameters).

    Edit the Config section below to change paths and settings.
    """

    # --------------------
    # Config (edit as needed)
    # --------------------
    LR_DIR = r"D:\Py_Code\img_match\SR_Imagery\tif\LR"     # folder of LR TIFFs to predict
    WEIGHTS = "UNet_SA_SameRes_TIF/outputs/run_auto/best.ckpt"         # trained weights (from training)
    SAVE_DIR = "UNet_SA_SameRes_TIF/outputs/run_auto/preds"            # output folder for predictions
    PATCH_SIZE = 256                                                    # crop/pad to this size

    # --------------------
    # Setup
    # --------------------
    device = get_device()
    os.makedirs(SAVE_DIR, exist_ok=True)

    # --------------------
    # Model
    # --------------------
    model = create_model(in_channels=5, out_channels=5).to(device)
    load_checkpoint(model, WEIGHTS, map_location=device)
    model.eval()

    # --------------------
    # Data (we re-use LR filenames; HR is not needed for export)
    # --------------------
    ds = PairTifDataset(lr_dir=LR_DIR, hr_dir=LR_DIR, size=PATCH_SIZE, augment=False, require_bands=5)

    # --------------------
    # Predict and write TIFFs
    # --------------------
    for i in tqdm(range(len(ds)), desc="export"):
        lr_t, _, name = ds[i]
        lr_t = lr_t.unsqueeze(0).to(device)
        pred = torch.clamp(model(lr_t), 0, 1).squeeze(0).cpu().numpy()
        _write_tif(os.path.join(SAVE_DIR, name), pred)


if __name__ == "__main__":
    # Export predictions with built-in defaults. Adjust Config in main() if needed.
    main()
