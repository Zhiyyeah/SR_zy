import os
import shutil
from tqdm import tqdm

# 输入目录（包含子文件夹）
src_root = "/home/zyye/SR/SRCNN_my/Imgaery/Water_tiles"

# 输出目录（统一存放）
dst_dir = "/home/zyye/SR/SRCNN_my/Imgaery/Water_tiles_all"
os.makedirs(dst_dir, exist_ok=True)

count = 0

# 遍历所有子文件夹
for root, dirs, files in os.walk(src_root):
    for file in tqdm(files,desc='Processing files'):
        if file.lower().endswith('.tif'):
            src_path = os.path.join(root, file)
            dst_path = os.path.join(dst_dir, file)

            # 如果文件名重复可加前缀防止覆盖
            if os.path.exists(dst_path):
                prefix = os.path.basename(root)
                dst_path = os.path.join(dst_dir, f"{prefix}_{file}")

            shutil.copy2(src_path, dst_path)
            count += 1

print(f"✅ 已复制 {count} 个 .tif 文件到：{dst_dir}")
