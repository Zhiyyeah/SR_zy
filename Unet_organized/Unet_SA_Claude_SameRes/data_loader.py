import os
from typing import List, Tuple
import tifffile as tiff
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def _read_tif(path: str) -> np.ndarray:
    """Read TIFF into CHW array. Do NOT rescale radiance; only cast to float32.

    Tries tifffile, falls back to PIL.
    """
    img = None
    arr = tiff.imread(path)
    img = arr


    if img is None:
        raise RuntimeError(f"Failed to read TIFF: {path}")

    arr = img
    if arr.ndim == 2:
        arr = arr[None, ...]
    elif arr.ndim == 3:
        # convert HWC -> CHW if needed
        if arr.shape[0] not in (1, 3, 4, 5):
            arr = arr.transpose(2, 0, 1)
    else:
        raise ValueError(f"Unsupported TIFF shape: {arr.shape}")

    # Do NOT normalize; preserve physical values. Cast to float32 for torch.
    return arr.astype(np.float32, copy=False)


## Removed any crop/pad: keep original spatial size


def _list_tifs(folder: str) -> List[str]:
    return [f for f in os.listdir(folder) if f.lower().endswith((".tif", ".tiff"))]


def _core_name(fname: str) -> str:
    name = os.path.splitext(os.path.basename(fname))[0]
    low = name.lower()
    if low.startswith("lr_"):
        return name[len("LR_"):]
    if low.startswith("hr_"):
        return name[len("HR_"):]
    return name


def _pair_by_prefix(lr_dir: str, hr_dir: str) -> List[Tuple[str, str]]:
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
    lr_total = len(_list_tifs(lr_dir))
    hr_total = len(_list_tifs(hr_dir))
    paired = len(_pair_by_prefix(lr_dir, hr_dir))
    print(f"LR total: {lr_total}")
    print(f"HR total: {hr_total}")
    print(f"Paired (common names): {paired}")
    return {"lr_total": lr_total, "hr_total": hr_total, "paired": paired}


class PairTifDataset(Dataset):
    """Paired 5-band TIFF dataset. No augmentation, no rescale, no resize."""

    def __init__(self, lr_dir: str, hr_dir: str, require_bands: int = 5):
        super().__init__()
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.require_bands = require_bands
        self.files: List[Tuple[str, str]] = _pair_by_prefix(lr_dir, hr_dir)
        if len(self.files) == 0:
            raise RuntimeError("No paired TIFFs found. Ensure filenames only differ by LR_/HR_ prefixes.")

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

        # Keep original size (assumed consistent across dataset)

        # Make contiguous to avoid negative/irregular strides in torch.from_numpy
        lr_c = np.ascontiguousarray(lr)
        hr_c = np.ascontiguousarray(hr)

        lr_t = torch.from_numpy(lr_c)
        hr_t = torch.from_numpy(hr_c)
        return lr_t, hr_t, os.path.basename(lr_path)


def make_loader(
    lr_dir: str,
    hr_dir: str,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    require_bands: int = 5,
):
    ds = PairTifDataset(lr_dir=lr_dir, hr_dir=hr_dir, require_bands=require_bands)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)


if __name__ == "__main__":
    # Minimal print for quick check
    LR_DIR = r"D:\Py_Code\img_match\SR_Imagery\tif\LR"
    HR_DIR = r"D:\Py_Code\img_match\SR_Imagery\tif\HR"
    validate_pair_counts(LR_DIR, HR_DIR)

    # 测试 _read_tif 函数
    def test_read_tif(path):
        arr = _read_tif(path)
        print(f"读取成功，shape: {arr.shape}, dtype: {arr.dtype}")


    # 示例路径，请替换为实际存在的tif文件路径
    test_tif_path = LR_DIR + "\\" + os.listdir(LR_DIR)[0] if os.listdir(LR_DIR) else None
    if test_tif_path:
        test_read_tif(test_tif_path)
    else:
        print("LR目录下没有tif文件可供测试")
