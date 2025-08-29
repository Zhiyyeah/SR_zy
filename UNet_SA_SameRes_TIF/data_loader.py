import os
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import tifffile as tiff  # 直接导入，若未安装会在启动时报错，便于尽早发现


# --------------------
# TIFF IO helpers
# --------------------
def _read_tif(path: str) -> np.ndarray:
    """使用 tifffile 读取 TIFF -> float32 CHW [0,1]。无额外 try/except 包装。"""
    arr = tiff.imread(path)

    # HWC -> CHW if needed
    if arr.ndim == 2:
        arr = arr[None, ...]
    elif arr.ndim == 3:
        # 如果第一维不是典型波段数，假设是 HWC
        if arr.shape[0] not in (1, 3, 4, 5):
            arr = arr.transpose(2, 0, 1)
    else:
        raise ValueError(f"不支持的 TIFF 形状: {arr.shape}")

    # 归一化到 [0,1]
    if np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.float32) / float(np.iinfo(arr.dtype).max)
    else:
        arr = arr.astype(np.float32)
    return arr


def _center_crop_or_pad(arr: np.ndarray, size: int = 256) -> np.ndarray:
    """Ensure arr (C,H,W) is exactly size×size via center crop or symmetric pad."""
    c, h, w = arr.shape
    pad_h = max(0, size - h)
    pad_w = max(0, size - w)
    if pad_h > 0 or pad_w > 0:
        pad = ((0, 0), (pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2))
        arr = np.pad(arr, pad, mode="reflect")
        _, h, w = arr.shape
    top = max(0, (h - size) // 2)
    left = max(0, (w - size) // 2)
    return arr[:, top:top + size, left:left + size]


# --------------------
# Pairing helpers (fixed prefixes: "LR_", "HR_")
# --------------------
def _list_tifs(folder: str) -> List[str]:
    return [f for f in os.listdir(folder) if f.lower().endswith((".tif", ".tiff"))]


def _core_name(fname: str) -> str:
    """Return core name after stripping LR_/HR_ prefix (case-insensitive) and extension."""
    name = os.path.splitext(os.path.basename(fname))[0]
    low = name.lower()
    if low.startswith("lr_"):
        return name[len("LR_"):]  # keep original case length; slicing is fine
    if low.startswith("hr_"):
        return name[len("HR_"):]  # ditto
    return name


def _pair_by_prefix(lr_dir: str, hr_dir: str) -> List[Tuple[str, str]]:
    """Pair files by stripping fixed prefixes LR_/HR_ and matching core names.

    - If multiple files share the same core on one side, pairs are created in
      index order after sorting both sides' filenames to keep it deterministic.
    """
    lr_files = _list_tifs(lr_dir)
    hr_files = _list_tifs(hr_dir)

    lr_map = {}
    for f in lr_files:
        lr_map.setdefault(_core_name(f).lower(), []).append(f)
    hr_map = {}
    for f in hr_files:
        hr_map.setdefault(_core_name(f).lower(), []).append(f)

    pairs: List[Tuple[str, str]] = []
    for core in sorted(set(lr_map.keys()) & set(hr_map.keys())):
        llist = sorted(lr_map[core])
        hlist = sorted(hr_map[core])
        m = min(len(llist), len(hlist))
        for i in range(m):
            pairs.append((os.path.join(lr_dir, llist[i]), os.path.join(hr_dir, hlist[i])))
    return pairs


def validate_pair_counts(lr_dir: str, hr_dir: str) -> dict:
    """Print minimal pairing stats: LR total, HR total, Paired (common names)."""
    lr_total = len(_list_tifs(lr_dir))
    hr_total = len(_list_tifs(hr_dir))
    paired = len(_pair_by_prefix(lr_dir, hr_dir))
    print(f"LR total: {lr_total}")
    print(f"HR total: {hr_total}")
    print(f"Paired (common names): {paired}")
    return {"lr_total": lr_total, "hr_total": hr_total, "paired": paired}


# --------------------
# Dataset
# --------------------
class PairTifDataset(Dataset):
    """Paired 5-band TIFF dataset (256×256 I/O enforced by crop/pad).

    - Pairs LR/HR TIFF files by stripping LR_/HR_ prefixes and matching cores.
    - Expects 5-band TIFFs; normalizes to [0,1].
    """

    def __init__(self, lr_dir: str, hr_dir: str, size: int = 256, augment: bool = False, require_bands: int = 5):
        super().__init__()
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.size = size
        self.augment = augment
        self.require_bands = require_bands

        self.files: List[Tuple[str, str]] = _pair_by_prefix(lr_dir, hr_dir)
        if len(self.files) == 0:
            raise RuntimeError("No paired TIFFs found. Ensure filenames share the same core after LR_/HR_ prefix.")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        lr_path, hr_path = self.files[idx]
        lr = _read_tif(lr_path)
        hr = _read_tif(hr_path)

        if self.require_bands is not None:
            if lr.shape[0] != self.require_bands:
                raise ValueError(f"LR has {lr.shape[0]} bands, expected {self.require_bands}: {lr_path}")
            if hr.shape[0] != self.require_bands:
                raise ValueError(f"HR has {hr.shape[0]} bands, expected {self.require_bands}: {hr_path}")

        lr = _center_crop_or_pad(lr, self.size)
        hr = _center_crop_or_pad(hr, self.size)

        if self.augment:
            if np.random.rand() < 0.5:
                lr = lr[:, :, ::-1]
                hr = hr[:, :, ::-1]
            if np.random.rand() < 0.5:
                lr = lr[:, ::-1, :]
                hr = hr[:, ::-1, :]
            k = np.random.randint(0, 4)
            if k:
                lr = np.rot90(lr, k, axes=(1, 2)).copy()
                hr = np.rot90(hr, k, axes=(1, 2)).copy()

        lr_t = torch.from_numpy(lr)
        hr_t = torch.from_numpy(hr)
        return lr_t, hr_t, os.path.basename(lr_path)


def make_loader(
    lr_dir: str,
    hr_dir: str,
    size: int = 256,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    augment: bool = False,
    require_bands: int = 5,
):
    ds = PairTifDataset(
        lr_dir=lr_dir,
        hr_dir=hr_dir,
        size=size,
        augment=augment,
        require_bands=require_bands,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)


if __name__ == "__main__":
    # Minimal self-check: only print the required three lines
    LR_DIR = r"D:\Py_Code\img_match\SR_Imagery\tif\LR"
    HR_DIR = r"D:\Py_Code\img_match\SR_Imagery\tif\HR"
    validate_pair_counts(LR_DIR, HR_DIR)

