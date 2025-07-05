#!/usr/bin/env python3
# validate_sr_fixed_no_psnr.py

import os
import glob
import time
import numpy as np
import rasterio
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- 用户请在此处修改路径和参数 ---
LR_DIR        = r"D:\Py_Code\Unet_SR\SR_zy\Imagey\Imagery_WaterLand\WaterLand_TOA_tiles_lr"
HR_DIR        = r"D:\Py_Code\Unet_SR\SR_zy\Imagey\Imagery_WaterLand\WaterLand_TOA_tiles_hr"
MODEL_PATH    = r"D:\Py_Code\Unet_SR\SR_zy\Unet_organized\Unet_SA_Claude\outputs\drop_lr_first_0.00043\models\epoch_74.pth"
OUTPUT_DIR    = r"D:\Py_Code\Unet_SR\SR_zy\Unet_organized\Unet_SA_Claude\outputs\drop_lr_first_0.00043\test_results"
IDX           =  1563           # 指定景序号(0-based)，不想遍历所有则设为 int，否则设为 None
UP_SCALE      = 8
WIDTH         = 64
RGB_CHANNELS  = [3, 2, 1]       # 1-based 波段序号，用于 RGB 可视化
# -----------------------------------

from model_attention import UNetSA

# 新增：与训练保持一致
USE_ATTENTION = True
DROPOUT_P = 0.2

def load_model(path, device, up_scale, num_channels, width, use_attention, dropout_p):
    model = UNetSA(
        up_scale=up_scale,
        img_channel=num_channels,
        width=width,
        use_attention=use_attention,
        dropout_p=dropout_p
    ).to(device)
    ckpt  = torch.load(path, map_location=device)
    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()
    return model

def read_image(path):
    with rasterio.open(path) as src:
        return src.read().astype(np.float32)

def validate_scene(lr_path, hr_path, model, device):
    lr_np = read_image(lr_path)
    hr_np = read_image(hr_path)

    lr_t = torch.from_numpy(lr_np).unsqueeze(0).to(device)
    hr_t = torch.from_numpy(hr_np).unsqueeze(0).to(device)
    with torch.no_grad():
        sr_t = model(lr_t)

    interp_t = F.interpolate(
        lr_t,
        size=hr_t.shape[-2:],
        mode="bilinear",
        align_corners=False
    )

    return (
        lr_np,
        interp_t.cpu().squeeze(0).numpy(),
        sr_t.cpu().squeeze(0).numpy(),
        hr_np,
    )

def plot_comparison(
    lr: np.ndarray,
    interp: np.ndarray,
    sr: np.ndarray,
    hr: np.ndarray,
    save_path: str,
    rgb_channels: list[int]
):
    idx = [b-1 for b in rgb_channels]
    def to_rgb(arr):
        rgb = arr[idx,:,:].transpose(1,2,0)
        vmin, vmax = np.percentile(rgb, (0.01,99.99))
        return np.clip((rgb-vmin)/(vmax-vmin), 0, 1)

    fig, axes = plt.subplots(2,2, figsize=(8,8))
    axes = axes.flatten()

    titles = [
        f"LowRes  {lr.shape[1]}\u00d7{lr.shape[2]}",
        f"Bilinear  {hr.shape[1]}\u00d7{hr.shape[2]}",
        f"Super-Res  {sr.shape[1]}\u00d7{sr.shape[2]}",
        f"GroundTruth  {hr.shape[1]}\u00d7{hr.shape[2]}"
    ]
    imgs = [lr, interp, sr, hr]

    for ax, img, ttl in zip(axes, imgs, titles):
        ax.imshow(to_rgb(img))
        ax.set_title(ttl, fontsize=10)
        ax.axis("off")

    fig.suptitle("Super-Resolution Results Comparison", fontsize=14)
    plt.tight_layout(rect=[0,0,1,0.96])
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

def plot_density_scatter_linear(hr: np.ndarray, sr: np.ndarray, save_path: str):
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    import numpy as np

    C = hr.shape[0]
    ncol = 4
    nrow = int(np.ceil(C / ncol))
    cell = 3.2
    cbar_ratio = 0.2

    fig = plt.figure(figsize=(ncol*cell + cbar_ratio*cell, nrow*cell))
    gs = gridspec.GridSpec(
        nrow, ncol+1,
        width_ratios=[*([1]*ncol), cbar_ratio],
        height_ratios=[1]*nrow,
        wspace=0.6, hspace=0.6,
        figure=fig
    )

    hb0 = None
    axes = []

    for b in range(C):
        r, c = divmod(b, ncol)
        ax = fig.add_subplot(gs[r, c])
        axes.append(ax)

        x = hr[b].ravel()
        y = sr[b].ravel()

        # 计算 RMSE 和 R²
        mse  = np.mean((y - x)**2)
        rmse = np.sqrt(mse)
        ssr  = np.sum((y - x)**2)
        sst  = np.sum((x - x.mean())**2)
        r2   = 1 - ssr / sst if sst > 0 else np.nan

        hb = ax.hexbin(
            x, y,
            gridsize=160,
            bins=None,
            cmap='viridis',
            mincnt=1
        )
        if hb0 is None:
            hb0 = hb

        lo_x, hi_x = np.percentile(x, (0.001, 99.99))
        lo_y, hi_y = np.percentile(y, (0.001, 99.99))

        mgn_x = (hi_x - lo_x) * 0.05
        mgn_y = (hi_y - lo_y) * 0.05

        lo_x -= mgn_x
        hi_x += mgn_x
        lo_y -= mgn_y
        hi_y += mgn_y

        lo = min(lo_x, lo_y)
        hi = max(hi_x, hi_y)

        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

        ax.plot([lo, hi], [lo, hi], 'k--', lw=1)

        txt = (
            f"RMSE {rmse:.3e}\n"
            f"R²  {r2:.3f}"
        )
        ax.text(
            0.02, 0.98, txt,
            transform=ax.transAxes,
            va='top', ha='left',
            fontsize=7.5,
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7)
        )

        ax.set_title(f"Band {b+1}", fontsize=9)
        ax.set_xlabel("GT", fontsize=8)
        ax.set_ylabel("SR", fontsize=8)

    for b in range(C, nrow*(ncol+1)-1):
        r, c = divmod(b, ncol+1)
        if c < ncol:
            fig.add_subplot(gs[r, c]).set_visible(False)

    cax = fig.add_subplot(gs[:, -1])
    cb = fig.colorbar(hb0, cax=cax, orientation='vertical')
    cb.set_label('Density', fontsize=8)

    fig.suptitle("Band-wise SR vs GT (Density)", fontsize=14)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    lr_list = sorted(glob.glob(os.path.join(LR_DIR, "*.tif")))
    hr_list = sorted(glob.glob(os.path.join(HR_DIR, "*.tif")))
    assert len(lr_list)==len(hr_list), "LR/HR 数量不匹配！"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tmp = read_image(lr_list[0])
    num_bands = tmp.shape[0]
    model = load_model(
        MODEL_PATH, device, UP_SCALE, num_bands, WIDTH,
        USE_ATTENTION, DROPOUT_P
    )

    if IDX is not None:
        indices = [IDX]
    else:
        indices = list(range(len(lr_list)))

    for i in indices:
        print(f"\n>>> Scene #{i}")
        lr, interp, sr, hr = validate_scene(
            lr_list[i], hr_list[i], model, device
        )
        cmp_path = os.path.join(OUTPUT_DIR, f"scene_{i:03d}_cmp.png")
        sca_path = os.path.join(OUTPUT_DIR, f"scene_{i:03d}_sca.png")

        plot_comparison(
            lr, interp, sr, hr,
            save_path=cmp_path,
            rgb_channels=RGB_CHANNELS
        )

        plot_density_scatter_linear(hr, sr, sca_path)

    print("\n验证完成。")

if __name__ == "__main__":
    main()
