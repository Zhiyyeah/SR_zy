import os
import re
import csv
import time
from typing import Optional, Tuple, List

import numpy as np
import torch
import rasterio
from rasterio.warp import transform_bounds
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，不用Tkinter

from .data_loader import PairTifDataset
from .model_attention import create_model
from .utils import get_device, load_checkpoint, set_seed
from .metrics import psnr as psnr_fn
from .main import split_dataset


# --------------------
# Config (edit as needed)
# --------------------
# 使用与训练相同的数据根目录，并按相同策略随机划分训练/验证
LR_DIR = r"D:\Py_Code\img_match\SR_Imagery\tif\LR"
HR_DIR = r"D:\Py_Code\img_match\SR_Imagery\tif\HR"
WEIGHTS = "Unet_organized/Unet_SA_Claude_SameRes/outputs/run_auto_charbonnier/models/best.pth"##记着�?
SAVE_DIR = "Unet_organized/Unet_SA_Claude_SameRes/outputs/run_auto_charbonnier/eval"
GOCI_ORI_ROOT = r"D:\Py_Code\img_match\batch_outputs"
BAND_LABELS = ["443 nm", "490 nm", "555 nm", "660 nm", "865 nm"]

# If LR and SR have identical spatial size (SameRes), bicubic upsample equals
# the original. Set this>1 (e.g., 2) to build a blurred bicubic baseline by
# downsampling then bicubic upsampling back to the original size.
BICUBIC_BASELINE_FACTOR = 1  # 1 = disable; 2 or 4 = enable baseline blur

# Colorbar unit label
TOA_UNIT_LABEL = "TOA radiance (W·m⁻²·sr⁻¹·µm⁻¹)"
VERBOSE_SKIP_LOG = True
# ## per-image CSV 路径�?eval_subset 按子集分别创�?VAL_SPLIT = 0.3  # 与训练默认一�?
EVAL_LOG_EVERY = 25  # 每多少个样本打印一次进度指�?
PRINT_CONFIG = True  # 是否在开头打印配�?
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2  # ʣ�� 0.1 ��Ϊ holdout
SPLIT_SEED = 42


# --------------------
# Helpers
# --------------------
def scene_id_from_patch_name(name: str) -> str:
    base = os.path.splitext(name)[0]
    low = base.lower()
    if low.startswith("lr_"):
        base = base[len("LR_"):]
    if low.startswith("hr_"):
        base = base[len("HR_"):]
    return re.sub(r"_r\d+_c\d+$", "", base, flags=re.IGNORECASE)


def find_nc_for_scene(root: str, scene_core: str) -> Optional[str]:
    """Find the GOCI NetCDF for a given scene.

    Expected layout (given by user):
        <root>/<scene_core>/GK2_GOCI2_L1B_..._subset_footprint.nc

    Strategy:
        1) Look for any *.nc in the exact subfolder <root>/<scene_core>/ matching
           pattern '*subset_footprint.nc'.
        2) If none, pick the first *.nc inside that folder.
        3) As a last resort, search recursively for a file whose dir contains the scene_core.
    """
    if not root or not os.path.isdir(root):
        return None

    scene_dir = os.path.join(root, scene_core)
    if os.path.isdir(scene_dir):
        # Prefer files that contain 'subset_footprint.nc'
        specific: List[str] = []
        any_nc: List[str] = []
        for r, _, files in os.walk(scene_dir):
            for f in files:
                if f.lower().endswith('.nc'):
                    p = os.path.join(r, f)
                    any_nc.append(p)
                    if 'subset_footprint.nc' in f.lower():
                        specific.append(p)
        if specific:
            # Return the lexicographically first for determinism
            return sorted(specific)[0]
        if any_nc:
            return sorted(any_nc)[0]

    # Fallback: search under root for any folder with scene_core in its name
    for r, dirs, files in os.walk(root):
        for d in dirs:
            if scene_core.lower() in d.lower():
                dd = os.path.join(r, d)
                cands = [os.path.join(dd, f) for f in os.listdir(dd) if f.lower().endswith('.nc')]
                if cands:
                    # Prefer subset_footprint
                    spec = [p for p in cands if 'subset_footprint.nc' in os.path.basename(p).lower()]
                    if spec:
                        return sorted(spec)[0]
                    return sorted(cands)[0]
    return None


def get_patch_bounds_wgs84(tif_path: str) -> Tuple[Tuple[float, float, float, float], str]:
    with rasterio.open(tif_path) as ds:
        b = ds.bounds
        crs = ds.crs
    b_wgs84 = transform_bounds(crs, "EPSG:4326", *b, densify_pts=21)
    return b_wgs84, str(crs)


def _resize_chw(chw: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    c, h, w = chw.shape
    if (h, w) == (out_h, out_w):
        return chw
    try:
        from skimage.transform import resize  # type: ignore
        out = np.stack([
            resize(chw[i], (out_h, out_w), order=1, preserve_range=True, anti_aliasing=False).astype(np.float32)
            for i in range(c)
        ], axis=0)
        return out
    except Exception:
        ys = (np.linspace(0, h - 1, out_h)).round().astype(int)
        xs = (np.linspace(0, w - 1, out_w)).round().astype(int)
        return chw[:, ys][:, :, xs]


def _resize_chw_order(chw: np.ndarray, out_h: int, out_w: int, order: int = 3) -> np.ndarray:
    """Resize CHW to target size with specified interpolation order.
    order=3 means bicubic; order=1 bilinear.
    """
    c, h, w = chw.shape
    if (h, w) == (out_h, out_w):
        return chw
    try:
        from skimage.transform import resize  # type: ignore
        out = np.stack([
            resize(chw[i], (out_h, out_w), order=order, preserve_range=True, anti_aliasing=False).astype(np.float32)
            for i in range(c)
        ], axis=0)
        return out
    except Exception:
        # Fallback nearest grid sampling
        ys = (np.linspace(0, h - 1, out_h)).round().astype(int)
        xs = (np.linspace(0, w - 1, out_w)).round().astype(int)
        return chw[:, ys][:, :, xs]


def _psnr_np(pred: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
    """PSNR using target dynamic range. Works on 2D arrays. NaN-safe."""
    p = pred.astype(np.float64)
    t = target.astype(np.float64)
    mask = np.isfinite(p) & np.isfinite(t)
    if not np.any(mask):
        return float('nan')
    diff = p[mask] - t[mask]
    mse = float(np.nanmean(diff * diff))
    if mse <= eps:
        return 100.0
    t_min = float(np.nanmin(t[mask]))
    t_max = float(np.nanmax(t[mask]))
    peak = max(t_max - t_min, eps)
    return 20.0 * np.log10(peak) - 10.0 * np.log10(mse)


def _r2_np(pred: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
    """R^2 score for 2D arrays. NaN-safe."""
    y = target.astype(np.float64)
    yhat = pred.astype(np.float64)
    mask = np.isfinite(y) & np.isfinite(yhat)
    if not np.any(mask):
        return float('nan')
    yv = y[mask]
    yh = yhat[mask]
    ss_res = float(np.sum((yv - yh) ** 2))
    y_mean = float(np.mean(yv))
    ss_tot = float(np.sum((yv - y_mean) ** 2))
    if ss_tot <= eps:
        return 1.0 if ss_res <= eps else 0.0
    return 1.0 - (ss_res / ss_tot)


def _mape_np(pred: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
    """MAPE (%) for 2D arrays with safe denom. NaN-safe."""
    y = target.astype(np.float64)
    yhat = pred.astype(np.float64)
    mask = np.isfinite(y) & np.isfinite(yhat)
    if not np.any(mask):
        return float('nan')
    yv = y[mask]
    yh = yhat[mask]
    denom = np.maximum(np.abs(yv), eps)
    return float(np.mean(np.abs((yv - yh) / denom)) * 100.0)


def read_nc_roi(nc_path: str, bbox_wgs84: Tuple[float, float, float, float], out_shape: Optional[Tuple[int, int, int]] = None) -> np.ndarray:
    """Read ROI from GOCI NetCDF using WGS84 bbox. Returns CHW float32.
    Uses netCDF4 directly to avoid heavy deps.
    """
    try:
        from netCDF4 import Dataset  # type: ignore
    except Exception:
        raise RuntimeError("netCDF4 not available (pip install netCDF4)")

    ds = Dataset(nc_path, "r")
    # Find lat/lon variables
    lat_name = None
    lon_name = None
    for name, var in ds.variables.items():
        n = name.lower()
        if lat_name is None and ("lat" in n or "latitude" in n):
            lat_name = name
        if lon_name is None and ("lon" in n or "longitude" in n):
            lon_name = name
    if lat_name is None or lon_name is None:
        raise RuntimeError("No lat/lon variables in NetCDF")
    lat = ds.variables[lat_name][:]
    lon = ds.variables[lon_name][:]
    minx, miny, maxx, maxy = bbox_wgs84

    def _roi_indices(lat, lon):
        if getattr(lat, "ndim", 0) == 1 and getattr(lon, "ndim", 0) == 1:
            ys = np.where((lat >= miny) & (lat <= maxy))[0]
            xs = np.where((lon >= minx) & (lon <= maxx))[0]
            if ys.size == 0 or xs.size == 0:
                raise RuntimeError("ROI outside lat/lon grid")
            return ys.min(), ys.max(), xs.min(), xs.max()
        else:
            mask = (lat >= miny) & (lat <= maxy) & (lon >= minx) & (lon <= maxx)
            ys, xs = np.where(mask)
            if ys.size == 0 or xs.size == 0:
                raise RuntimeError("ROI mask empty for lat/lon grid")
            return ys.min(), ys.max(), xs.min(), xs.max()

    y0, y1, x0, x1 = _roi_indices(lat, lon)

    # Pick 5 band variables (prefer names with expected wavelengths)
    order = ["443", "490", "555", "660", "865"]
    band_vars: List[str] = []
    for w in order:
        found = None
        for name, var in ds.variables.items():
            if name in (lat_name, lon_name):
                continue
            if getattr(var, "ndim", 0) == 2 and w in name:
                found = name
                break
        if found is not None:
            band_vars.append(found)
    if len(band_vars) < 5:
        # fallback: first 5 2D variables excluding lat/lon
        band_vars = []
        for name, var in ds.variables.items():
            if name in (lat_name, lon_name):
                continue
            if getattr(var, "ndim", 0) == 2:
                band_vars.append(name)
            if len(band_vars) >= 5:
                break
    if len(band_vars) < 5:
        raise RuntimeError("Not enough 2D band variables in NetCDF")

    data = []
    for name in band_vars[:5]:
        arr = ds.variables[name][y0:y1 + 1, x0:x1 + 1].astype(np.float32)
        data.append(arr)
    data = np.stack(data, axis=0)  # (C,H,W)

    if out_shape is not None:
        c, h, w = out_shape
        data = _resize_chw(data, h, w)
        if data.shape[0] < c:
            pad = np.zeros((c - data.shape[0], h, w), dtype=np.float32)
            data = np.concatenate([data, pad], axis=0)
    return data


def read_nc_full_and_latlon(nc_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read full GOCI NetCDF bands plus lat/lon grids.

    Returns:
        data: (C,H,W) float32 for 5 bands (preferred wavelengths order)
        lat:  (H,W) or (H,) latitude grid
        lon:  (H,W) or (W,) longitude grid
    """
    try:
        from netCDF4 import Dataset  # type: ignore
    except Exception:
        raise RuntimeError("netCDF4 not available (pip install netCDF4)")

    ds = Dataset(nc_path, "r")

    # Helper: recursively iterate variables with their full path
    def _iter_vars(g, prefix=""):
        for name, var in g.variables.items():
            yield prefix + name, var
        for sub, sg in g.groups.items():
            yield from _iter_vars(sg, prefix + sub + "/")

    # 1) Latitude/Longitude: prefer navigation_data group
    lat = lon = None
    if "navigation_data" in ds.groups:
        nav = ds.groups["navigation_data"]
        if "latitude" in nav.variables and "longitude" in nav.variables:
            lat = nav.variables["latitude"][:]
            lon = nav.variables["longitude"][:]
    # Fallback: search recursively
    if lat is None or lon is None:
        for full_name, var in _iter_vars(ds):
            lname = full_name.lower()
            if lat is None and (lname.endswith("latitude") or "/latitude" in lname or "latitude" in lname):
                lat = var[:]
            if lon is None and (lname.endswith("longitude") or "/longitude" in lname or "longitude" in lname):
                lon = var[:]
            if lat is not None and lon is not None:
                break
    if lat is None or lon is None:
        raise RuntimeError("No lat/lon variables in NetCDF (including navigation_data group)")

    # 2) Pick 5 band variables in preferred order, search recursively (exclude lat/lon)
    order = ["443", "490", "555", "660", "865"]
    chosen_vars: List[np.ndarray] = []
    for w in order:
        found_arr = None
        for full_name, var in _iter_vars(ds):
            lname = full_name.lower()
            if "latitude" in lname or "longitude" in lname:
                continue
            if getattr(var, "ndim", 0) == 2 and w in lname:
                found_arr = var[:].astype(np.float32)
                break
        if found_arr is not None:
            chosen_vars.append(found_arr)
    if len(chosen_vars) < 5:
        # fallback: take first 5 2D non-latlon variables found recursively
        chosen_vars = []
        for full_name, var in _iter_vars(ds):
            lname = full_name.lower()
            if "latitude" in lname or "longitude" in lname:
                continue
            if getattr(var, "ndim", 0) == 2:
                chosen_vars.append(var[:].astype(np.float32))
            if len(chosen_vars) >= 5:
                break
    if len(chosen_vars) < 5:
        raise RuntimeError("Not enough 2D band variables in NetCDF (recursive search)")

    data = np.stack(chosen_vars[:5], axis=0)
    return data, np.array(lat), np.array(lon)


def roi_indices_from_latlon(
    lat: np.ndarray, lon: np.ndarray, bbox_wgs84: Tuple[float, float, float, float]
) -> Tuple[int, int, int, int]:
    """Compute y0,y1,x0,x1 indices of ROI based on lat/lon arrays and bbox.
    Handles lat/lon in 1D or 2D.
    """
    minx, miny, maxx, maxy = bbox_wgs84
    if getattr(lat, "ndim", 0) == 1 and getattr(lon, "ndim", 0) == 1:
        ys = np.where((lat >= miny) & (lat <= maxy))[0]
        xs = np.where((lon >= minx) & (lon <= maxx))[0]
        if ys.size == 0 or xs.size == 0:
            raise RuntimeError("ROI outside lat/lon grid")
        return int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
    else:
        mask = (lat >= miny) & (lat <= maxy) & (lon >= minx) & (lon <= maxx)
        ys, xs = np.where(mask)
        if ys.size == 0 or xs.size == 0:
            raise RuntimeError("ROI mask empty for lat/lon grid")
        return int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())


def save_four_col_figure(path: str, goci_raw: np.ndarray, interp: np.ndarray, sr: np.ndarray, hr: np.ndarray,
                         band_labels: List[str], title: str):
    import matplotlib.pyplot as plt
    # 底纹样式（半透明黑）
    text_bbox = dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.35, edgecolor="none")

    c = min(5, sr.shape[0], hr.shape[0], goci_raw.shape[0], interp.shape[0])
    fig, axes = plt.subplots(
        nrows=c,
        ncols=4,
        figsize=(14.0, 2.6 * c),
        gridspec_kw=dict(wspace=0.25, hspace=0.35),
        constrained_layout=False,
    )
    try:
        fig.suptitle(title, fontsize=12, y=0.985)
    except Exception:
        pass

    for i in range(c):
        raw_b = goci_raw[i]
        int_b = interp[i]
        sr_b = sr[i]
        hr_b = hr[i]
        vmin = float(min(raw_b.min(), int_b.min(), sr_b.min(), hr_b.min()))
        vmax = float(max(raw_b.max(), int_b.max(), sr_b.max(), hr_b.max()))
        label = band_labels[i] if i < len(band_labels) else f"Band {i+1}"

    ax = axes[i, 0]
    im0 = ax.imshow(raw_b, cmap="viridis", vmin=vmin, vmax=vmax)
    ax.axis("off")
    ax.text(0.98, 0.04, f"GOCI Raw\n{label}", ha="right", va="bottom", color="w", fontsize=11, transform=ax.transAxes, bbox=text_bbox)

    ax = axes[i, 1]
    ax.imshow(int_b, cmap="viridis", vmin=vmin, vmax=vmax)
    bic_psnr = _psnr_np(int_b, hr_b)
    bic_mape = _mape_np(int_b, hr_b)
    ax.axis("off")
    ax.text(0.98, 0.04, f"Interp\nPSNR {bic_psnr:.2f} dB\nMAPE {bic_mape:.1f}%", ha="right", va="bottom", color="w", fontsize=11, transform=ax.transAxes, bbox=text_bbox)

    sr_t = torch.from_numpy(sr_b[None, None, ...])
    hr_t = torch.from_numpy(hr_b[None, None, ...])
    bpsnr = psnr_fn(sr_t, hr_t)
    sr_mape = _mape_np(sr_b, hr_b)

    ax = axes[i, 2]
    ax.imshow(sr_b, cmap="viridis", vmin=vmin, vmax=vmax)
    ax.axis("off")
    ax.text(0.98, 0.04, f"SR\nPSNR {bpsnr:.2f} dB\nMAPE {sr_mape:.1f}%", ha="right", va="bottom", color="w", fontsize=11, transform=ax.transAxes, bbox=text_bbox)

    ax = axes[i, 3]
    im3 = ax.imshow(hr_b, cmap="viridis", vmin=vmin, vmax=vmax)
    ax.axis("off")
    ax.text(0.98, 0.04, f"HR\n{label}", ha="right", va="bottom", color="w", fontsize=11, transform=ax.transAxes, bbox=text_bbox)
    cbar = fig.colorbar(im3, ax=ax, fraction=0.05, pad=0.02, aspect=35)
    cbar.ax.tick_params(labelsize=8)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.subplots_adjust(top=0.92, bottom=0.06, left=0.06, right=0.98)
    fig.savefig(path, dpi=400)
    plt.close(fig)


def save_compact_fourcol_subplot(
    path: str,
    goci_raw: np.ndarray,
    bicubic: np.ndarray,
    sr: np.ndarray,
    hr: np.ndarray,
    band_labels: List[str],
    unit_label: str = TOA_UNIT_LABEL,
    col_texts: Tuple[str, str, str, str] = ("GOCI-II", "Bicubic", "Super Resolution", "HR2"),
    goci_roi_box: Optional[Tuple[int, int, int, int]] = None,  # (y0,y1,x0,x1) to limit view
):
    """Save a 4-column compact figure with tiny gaps and labels inside tiles.

    Layout per row (very small gaps):
        [ GOCI-II | Bicubic | Estimated map | HR2 | colorbar ]

    - Text is drawn inside each image (bottom-left) with a semi-transparent
      background patch plus stroke so其在浅色背景上依然清晰�?
    - Row markers (a), (b), (c), ... on the first column.
    - Each row has its own colorbar in a slim 5th column.
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    nband = int(min(5, goci_raw.shape[0], bicubic.shape[0], sr.shape[0], hr.shape[0]))

    # Figure geometry: 4 image columns + 1 colorbar column
    fig = plt.figure(figsize=(15.0, 2.6 * nband))
    gs = GridSpec(
        nrows=nband,
        ncols=5,
        figure=fig,
        # Make colorbar a bit slimmer
        width_ratios=[1, 1, 1, 1, 0.045],
        # Make horizontal gaps even smaller
        wspace=0.006,
        hspace=0.01,
    )

    # Helper to draw a single tile with inside text
    def _draw(ax, img, vmin, vmax, label_text, imshow_kwargs=None):
        if imshow_kwargs is None:
            imshow_kwargs = {}
        im = ax.imshow(img, cmap="viridis", vmin=vmin, vmax=vmax, **imshow_kwargs)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        # 右下角文字，无底纹
        ax.text(
            0.98, 0.04, label_text,
            transform=ax.transAxes,
            fontsize=10,
            color="w",
            ha="right",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.35, edgecolor="none")
        )
        return im

    for i in range(nband):
        band_text = band_labels[i] if i < len(band_labels) else f"Band {i+1}"
        raw_b = goci_raw[i]
        bic_b = bicubic[i]
        sr_b = sr[i]
        hr_b = hr[i]

        if goci_roi_box is not None:
            y0, y1, x0, x1 = goci_roi_box
            raw_for_range = raw_b[y0:y1 + 1, x0:x1 + 1]
        else:
            raw_for_range = raw_b
        vmin = float(min(raw_for_range.min(), bic_b.min(), sr_b.min(), hr_b.min()))
        vmax = float(max(raw_for_range.max(), bic_b.max(), sr_b.max(), hr_b.max()))

        ax0 = fig.add_subplot(gs[i, 0])
        ax1 = fig.add_subplot(gs[i, 1])
        ax2 = fig.add_subplot(gs[i, 2])
        ax3 = fig.add_subplot(gs[i, 3])
        cax = fig.add_subplot(gs[i, 4])

        # Draw images with very small gaps and internal labels
        # Column-1: GOCI ROI displayed as a square by compressing vertical to match width
        if goci_roi_box is not None:
            y0, y1, x0, x1 = goci_roi_box
            roi_img = raw_b[y0:y1 + 1, x0:x1 + 1]
            h_roi = (y1 - y0 + 1)
            w_roi = (x1 - x0 + 1)
            im0 = _draw(
                ax0,
                roi_img,
                vmin,
                vmax,
                f"{col_texts[0]} ({band_text})",
                # Make display coordinates a square: width == height == w_roi
                imshow_kwargs=dict(origin="upper", extent=[0, w_roi, w_roi, 0]),
            )
            ax0.set_aspect('equal')
        else:
            im0 = _draw(ax0, raw_b, vmin, vmax, f"{col_texts[0]} ({band_text})")
            ax0.set_aspect('equal')
        # Compute metrics per band (bicubic vs HR, SR vs HR)
        bic_psnr = _psnr_np(bic_b, hr_b)
        bic_mape = _mape_np(bic_b, hr_b)
        sr_psnr = _psnr_np(sr_b, hr_b)
        sr_mape = _mape_np(sr_b, hr_b)

        bic_text = f"{col_texts[1]} ({band_text})\nPSNR {bic_psnr:.2f} dB | MAPE {bic_mape:.1f}%"
        sr_text = f"{col_texts[2]} ({band_text})\nPSNR {sr_psnr:.2f} dB | MAPE {sr_mape:.1f}%"

        _draw(ax1, bic_b, vmin, vmax, bic_text)
        _draw(ax2, sr_b, vmin, vmax, sr_text)
        im3 = _draw(ax3, hr_b, vmin, vmax, f"{col_texts[3]} ({band_text})")
        # Keep other columns square (most patches are H==W)
        ax1.set_aspect('equal'); ax2.set_aspect('equal'); ax3.set_aspect('equal')

        # Row marker and per-row colorbar
        ax0.text(0.02, 0.96, f"({chr(97 + i)})", transform=ax0.transAxes, fontsize=13, color="w", va="top", ha="left")
        cb = fig.colorbar(im3, cax=cax)
        cb.ax.tick_params(labelsize=7, pad=1)
        try:
            cb.outline.set_linewidth(0.6)
        except Exception:
            pass
        try:
            cb.set_label(unit_label, fontsize=7, rotation=270, labelpad=6)
            cb.ax.yaxis.set_label_position("right")
        except Exception:
            pass
        try:
            bbox = cax.get_position()
            shrink = 0.9
            new_h = bbox.height * shrink
            new_y = bbox.y0 + (bbox.height - new_h) / 2.0
            cax.set_position([bbox.x0, new_y, bbox.width, new_h])
        except Exception:
            pass

    # Keep margins small but shift a bit left so the colorbar is fully visible
    fig.subplots_adjust(left=0.01, right=0.975, top=0.99, bottom=0.01)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=400)
    plt.close(fig)


def eval_subset(ds_subset, split_name: str, model, device):
    split_dir = os.path.join(SAVE_DIR, split_name)
    os.makedirs(split_dir, exist_ok=True)
    csv_path = os.path.join(split_dir, "metrics.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "filename",
                "sr_psnr_mean", "sr_mape_mean",
                "sr_psnr_b1","sr_psnr_b2","sr_psnr_b3","sr_psnr_b4","sr_psnr_b5",
                "sr_mape_b1","sr_mape_b2","sr_mape_b3","sr_mape_b4","sr_mape_b5",
            ])
    saved = 0
    for i in tqdm(range(len(ds_subset)), desc=f"eval-{split_name}"):
        lr_t, hr_t, name = ds_subset[i]
        lr_t = lr_t.unsqueeze(0).to(device)
        hr_t = hr_t.unsqueeze(0).to(device)
        with torch.no_grad():
            sr = model(lr_t)
        sr_np = sr.squeeze(0).cpu().numpy()
        hr_np = hr_t.squeeze(0).cpu().numpy()
        lr_np = lr_t.squeeze(0).cpu().numpy()
        # Restore GOCI first-column rendering using original logic
        scene_core = scene_id_from_patch_name(name)
        # resolve LR full path from subset
        try:
            if hasattr(ds_subset, 'files'):
                lr_full, _ = ds_subset.files[i]
            elif hasattr(ds_subset, 'dataset') and hasattr(ds_subset, 'indices') and hasattr(ds_subset.dataset, 'files'):
                lr_full, _ = ds_subset.dataset.files[ds_subset.indices[i]]
            else:
                raise AttributeError('dataset has no files mapping')
        except Exception:
            lr_full = os.path.join(LR_DIR, os.path.basename(name))

        # Default fallbacks
        goci_label = "GOCI-II"
        goci_full = None
        roi_box = None

        # Compute patch bounds and find GOCI NetCDF
        try:
            bbox_wgs84, _ = get_patch_bounds_wgs84(lr_full)
        except Exception:
            bbox_wgs84 = None
        nc_path = find_nc_for_scene(GOCI_ORI_ROOT, scene_core)
        if nc_path is not None:
            try:
                goci_full, goci_lat, goci_lon = read_nc_full_and_latlon(nc_path)
                if bbox_wgs84 is not None:
                    y0, y1, x0, x1 = roi_indices_from_latlon(goci_lat, goci_lon, bbox_wgs84)
                    roi_box = (y0, y1, x0, x1)
            except Exception:
                goci_full = None
                goci_label = "GOCI read failed"
        else:
            goci_label = "GOCI not found"

        goci_raw = goci_full if goci_full is not None else np.zeros_like(lr_np)

        patch_base = os.path.splitext(os.path.basename(name))[0]
        out_png_compact = os.path.join(split_dir, f"{patch_base}_compare_compact.png")
        try:
            save_compact_fourcol_subplot(
                out_png_compact,
                goci_raw,
                lr_np,
                sr_np,
                hr_np,
                BAND_LABELS,
                col_texts=(goci_label, "Interpolated GOCI", "Super Resolution", "Landsat"),
                goci_roi_box=roi_box,
            )
            saved += 1
        except Exception:
            out_png = os.path.join(split_dir, f"{patch_base}_compare_4col.png")
            save_four_col_figure(out_png, goci_raw, lr_np, sr_np, hr_np, BAND_LABELS, scene_core)
            saved += 1
        # SR metrics only
        try:
            nb = int(min(5, sr_np.shape[0], hr_np.shape[0]))
            sr_psnrs = []
            sr_mapes = []
            for bi in range(nb):
                sr_psnrs.append(_psnr_np(sr_np[bi], hr_np[bi]))
                sr_mapes.append(_mape_np(sr_np[bi], hr_np[bi]))
            while len(sr_psnrs) < 5:
                sr_psnrs.append(float('nan'))
            while len(sr_mapes) < 5:
                sr_mapes.append(float('nan'))
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    os.path.basename(name),
                    float(np.nanmean(sr_psnrs)), float(np.nanmean(sr_mapes)),
                    *[float(x) for x in sr_psnrs[:5]],
                    *[float(x) for x in sr_mapes[:5]],
                ])
        except Exception as e:
            tqdm.write(f"[warn] write CSV failed {name}: {type(e).__name__}: {e}")
    tqdm.write(f"[summary:{split_name}] saved={saved} csv={csv_path}")


@torch.no_grad()
def main():
    t0 = time.time()
    device = get_device()
    if PRINT_CONFIG:
        print("================= Eval 配置 =================")
        print(f"设备: {device} | CUDA 可用: {torch.cuda.is_available()}")
        print(f"权重文件: {WEIGHTS}")
        print(f"LR_DIR: {LR_DIR}\nHR_DIR: {HR_DIR}")
        print(f"保存目录: {SAVE_DIR}")
#         print(f"VAL_SPLIT: {VAL_SPLIT} (使用与训练一致的 split_dataset 随机划分)")
        print(f"EVAL_LOG_EVERY: {EVAL_LOG_EVERY}")
        print("============================================")
    model = create_model(in_channels=5, out_channels=5).to(device)
    load_checkpoint(model, WEIGHTS, map_location=device)
    model.eval()

    # Three-way split with fixed seed
    set_seed(SPLIT_SEED)
    ds_full = PairTifDataset(lr_dir=LR_DIR, hr_dir=HR_DIR, require_bands=5)
    total_n = len(ds_full)
    train_n = int(total_n * TRAIN_RATIO)
    val_n = int(total_n * VAL_RATIO)
    holdout_n = total_n - train_n - val_n
    from torch.utils.data import random_split
    train_ds, val_ds, holdout_ds = random_split(
        ds_full,
        [train_n, val_n, holdout_n],
        generator=torch.Generator().manual_seed(SPLIT_SEED)
    )
    if PRINT_CONFIG:
        print(f"训练集样本数: {len(train_ds)} | 验证集样本数: {len(val_ds)} | 保留集样本数: {len(holdout_ds)}")
        print("说明: 保留集(holdout)未参与当前训练与验证, 可在单独脚本中做最终评估。")
    os.makedirs(SAVE_DIR, exist_ok=True)
    eval_subset(train_ds, 'train', model, device)
    eval_subset(val_ds, 'val', model, device)
    eval_subset(holdout_ds, 'holdout', model, device)
    print(f"总耗时: {time.time()-t0:.2f} 秒 | 输出目录: {SAVE_DIR}")
    return

if __name__ == "__main__":
    main()




