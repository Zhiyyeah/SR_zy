import os
import argparse
import shutil
from typing import List

import torch


def list_top_level_files(folder: str, exts: List[str]) -> List[str]:
    exts = [e.lower() for e in exts]
    out: List[str] = []
    for name in os.listdir(folder):
        p = os.path.join(folder, name)
        if os.path.isfile(p) and any(name.lower().endswith(e) for e in exts):
            out.append(name)
    return out


def main():
    parser = argparse.ArgumentParser(description="Deterministically split viz4 results into train/val using seed=123")
    parser.add_argument("--input-dir", default=r"D:\\Py_Code\\Unet_SR\\SR_zy\\Unet_organized\\Unet_SA_Claude_SameRes\\outputs\\run_auto_ssim_10\\eval\\viz4", help="Folder containing result images")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split fraction (e.g., 0.1)")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for deterministic split")
    parser.add_argument("--move", action="store_true", help="Move files instead of copying")
    parser.add_argument("--exts", nargs="*", default=[".png"], help="File extensions to include")
    args = parser.parse_args()

    root = args.input_dir
    if not os.path.isdir(root):
        raise SystemExit(f"Input directory does not exist: {root}")

    files = list_top_level_files(root, args.exts)
    if not files:
        raise SystemExit(f"No files with extensions {args.exts} found under {root}")

    n = len(files)
    val_n = int(n * args.val_split)
    train_n = n - val_n

    # Deterministic permutation using torch.Generator like your training split
    gen = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(n, generator=gen).tolist()
    files_perm = [files[i] for i in perm]
    val_files = set(files_perm[:val_n])

    train_dir = os.path.join(root, "train")
    val_dir = os.path.join(root, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    moved_train = 0
    moved_val = 0
    op = shutil.move if args.move else shutil.copy2

    for name in files:
        src = os.path.join(root, name)
        # Skip if already inside subfolder
        if os.path.dirname(src) in (train_dir, val_dir):
            continue
        dst_dir = val_dir if name in val_files else train_dir
        dst = os.path.join(dst_dir, name)
        if os.path.exists(dst):
            # Skip duplicates
            continue
        op(src, dst)
        if dst_dir == val_dir:
            moved_val += 1
        else:
            moved_train += 1

    mode = "moved" if args.move else "copied"
    print(f"Total files: {n} | train={train_n} val={val_n}")
    print(f"{mode} -> train: {moved_train}, val: {moved_val}")
    print(f"Train dir: {train_dir}")
    print(f"Val dir:   {val_dir}")


if __name__ == "__main__":
    main()

