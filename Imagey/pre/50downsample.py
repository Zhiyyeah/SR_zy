import os
import rasterio
from rasterio.enums import Resampling
import numpy as np
from tqdm import tqdm

# 输入输出目录
input_dir = "/home/zyye/SR/SRCNN_my/Imagery/Water_TOA_tiles"
output_dir = "/home/zyye/SR/SRCNN_my/Imagery/Water_TOA_tiles_lr"
os.makedirs(output_dir, exist_ok=True)

# 降采样倍数
scale = 8

count = 0
for fname in tqdm(os.listdir(input_dir)):
    if not fname.lower().endswith(".tif"):
        continue

    in_path = os.path.join(input_dir, fname)
    out_path = os.path.join(output_dir, fname)

    with rasterio.open(in_path) as src:
        # 原始尺寸
        h, w = src.height, src.width
        new_h, new_w = h // scale, w // scale

        # 读取所有波段数据并重采样
        data = src.read(
            out_shape=(src.count, new_h, new_w),
            resampling=Resampling.bilinear  # 双线性插值
        )

        # 更新 transform
        transform = src.transform * src.transform.scale(
            (w / new_w),
            (h / new_h)
        )

        # 更新 metadata
        out_meta = src.meta.copy()
        out_meta.update({
            "height": new_h,
            "width": new_w,
            "transform": transform
        })

        # 写入文件
        with rasterio.open(out_path, 'w', **out_meta) as dst:
            dst.write(data)

    count += 1

print(f"✅ 成功降采样 {count} 个文件并保存至：{output_dir}")
