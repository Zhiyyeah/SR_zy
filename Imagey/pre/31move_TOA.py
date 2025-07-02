import os
import shutil

# 源目录和目标目录
src_root = "Imagery/Water"
dst_folder = "Imagery/Water_TOA"

# 创建目标目录
os.makedirs(dst_folder, exist_ok=True)

# 初始化计数器
count = 0

# 遍历文件夹
for root, dirs, files in os.walk(src_root):
    for fname in files:
        if "TOA" in fname and fname.lower().endswith(".tif"):
            src_path = os.path.join(root, fname)
            dst_path = os.path.join(dst_folder, fname)

            if os.path.exists(dst_path):
                print(f"⚠️ 已存在，跳过：{dst_path}")
                continue

            shutil.move(src_path, dst_path)
            print(f"✅ 移动：{src_path} → {dst_path}")
            count += 1

# 最终统计
print(f"\n📦 总共移动了 {count} 个 TOA tif 文件到：{dst_folder}")
