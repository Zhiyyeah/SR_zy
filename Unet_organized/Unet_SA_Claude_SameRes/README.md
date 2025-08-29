Unet_SA_Claude_SameRes (5-band TIFF, same-size SR)

- Input/Output: 5×256×256 (channels match; no scaling).
- Model: Same-resolution U-Net style with Spatial Attention in residual blocks and global residual.
- Data pairing: Assumes filenames differ only by LR_/HR_ prefixes.

Paths (edit in main.py):
- LR: D:\Py_Code\img_match\SR_Imagery\tif\LR
- HR: D:\Py_Code\img_match\SR_Imagery\tif\HR

Run training:
- python -m Unet_organized.Unet_SA_Claude_SameRes.main

Outputs:
- Prints epoch Train Loss and Val PSNR
- Saves checkpoints per epoch under outputs/run_auto/models: epoch_XXX.pth, best.pth, last.pth

