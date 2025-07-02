import os
import numpy as np
import rasterio
from rasterio.windows import Window

# ========== 用户配置 ==========
input_folder  = "/public/home/zyye/SR_organized/Imagey"
output_folder = "/public/home/zyye/SR_organized/Imagey_tiles"
tile_size     = 512
# ===============================

os.makedirs(output_folder, exist_ok=True)
total_count = 0

for fname in os.listdir(input_folder):
    if not fname.lower().endswith(".tif"):
        continue

    src_path = os.path.join(input_folder, fname)
    with rasterio.open(src_path) as src:
        H, W = src.height, src.width
        bands = src.count

        scene_count = 0
        for i in range(0, H, tile_size):
            for j in range(0, W, tile_size):
                # 跳过边缘不满的
                if i + tile_size > H or j + tile_size > W:
                    continue

                window = Window(j, i, tile_size, tile_size)
                # 读取所有波段数据，shape = (bands, tile_size, tile_size)
                block = src.read(window=window)

                # 如果任何像元是 NaN，就跳过
                if np.isnan(block).any():
                    continue

                # 否则写出这个有效 patch
                meta = src.meta.copy()
                meta.update({
                    "height": tile_size,
                    "width" : tile_size,
                    "transform": src.window_transform(window),
                    "dtype": block.dtype,
                    "count": bands,
                })

                out_name = f"{os.path.splitext(fname)[0]}_r{i:05d}_c{j:05d}.tif"
                out_path = os.path.join(output_folder, out_name)
                with rasterio.open(out_path, "w", **meta) as dst:
                    dst.write(block)

                scene_count += 1
                total_count += 1

        print(f"✅ {fname} 生成有效切片: {scene_count} 张")

print(f"\n🎉 全部完成，总共生成有效 512×512 切片：{total_count} 张")
