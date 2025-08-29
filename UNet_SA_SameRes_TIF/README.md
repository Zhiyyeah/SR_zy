UNet-SR (Same Resolution, 5-band TIFF)

Overview

- Task: Map 5-band low-resolution inputs to same-size high-resolution outputs.
- Input/Output size: 256×256 (C=5). The model preserves spatial size.
- Model: U-Net with Spatial Attention (CBAM-style) on skip connections + global residual learning.
- Data: Paired 5-band TIFF files with matching filenames in LR and HR folders.

Default Data Paths

- LR: `D:\Py_Code\img_match\SR_Imagery\tif\LR`
- HR: `D:\Py_Code\img_match\SR_Imagery\tif\HR`

Run (Training)

```bash
python -m UNet_SA_SameRes_TIF.main
```

- Defaults are embedded in `UNet_SA_SameRes_TIF/main.py` (edit the Config section at the top of `main()` to change paths and hyperparameters).

Run (Evaluation / Export TIFs)

```bash
python -m UNet_SA_SameRes_TIF.eval
```

- Defaults are embedded in `UNet_SA_SameRes_TIF/eval.py` (edit the Config section in `main()` to change the weights, LR folder, or save location).

Key Optimizations

- Spatial Attention: CBAM-style spatial attention applied on skip connections to emphasize informative spatial locations.
- GroupNorm: Replaces BatchNorm for stability on small batches and heterogeneous data.
- Global Residual: Predicts residual added to input (1×1 conv aligns channels), accelerating convergence and improving fidelity.
- Charbonnier + L1 Loss: Stable and edge-preserving supervision option.
- Mixed precision (optional): Can be enabled via `--amp` to speed up training on GPU.

Dependencies

- Python 3.8+
- PyTorch, numpy, tqdm
- tifffile (preferred) or PIL as fallback for TIFF IO
- Optional: scikit-image (if you want SSIM metric)

Install

```bash
pip install -r UNet_SA_SameRes_TIF/requirements.txt
```

Notes

- If TIFFs are not exactly 256×256, the loader will center-crop or pad to 256.
- Band order is preserved as-is (assumes LR and HR share the same 5-band order: GOCI/Landsat consistent).
- Replace defaults with your own paths via CLI flags.
