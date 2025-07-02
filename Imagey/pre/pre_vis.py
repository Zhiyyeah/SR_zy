import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt

# 设置目录路径
lr_dir = '/public/home/zyye/SR_organized/Imagey/WaterLand_TOA_tiles_lr'
hr_dir = '/public/home/zyye/SR_organized/Imagey/WaterLand_TOA_tiles_hr'

# RGB 波段索引（注意 rasterio 中波段索引从 1 开始）
rgb_band_idx = [4, 3, 2]  # Landsat 8 的 B4, B3, B2

# 获取 patch 文件名列表
filenames = sorted([f for f in os.listdir(hr_dir) if f.endswith('.tif')])

# 设置索引值（你可以更改这里）
idx = 4212

# 选择文件
filename = filenames[idx]
print(f"当前处理文件：{filename}")
lr_path = os.path.join(lr_dir, filename)
hr_path = os.path.join(hr_dir, filename)

# 读取并转换为 RGB 图像（值归一化+转置）
def read_rgb_image(path, band_idx, gamma=0.75):
    with rasterio.open(path) as src:
        img = src.read(band_idx)
    img = img.astype(np.float32)
    
    # 百分位拉伸
    p2, p98 = np.percentile(img, (8, 80))
    img = np.clip(img, p2, p98)
    img = (img - p2) / (p98 - p2 + 1e-6)
    
    # Gamma校正
    img = np.power(img, gamma)
    
    return np.transpose(img, (1, 2, 0))

# 加载图像
lr_rgb = read_rgb_image(lr_path, rgb_band_idx)
hr_rgb = read_rgb_image(hr_path, rgb_band_idx)

# 可视化并保存为图片
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].imshow(lr_rgb)
axs[0].set_title(f' Landsat 8 Low Resolution Imagery Patch (64*64)', fontsize=10)
axs[0].axis('off')

axs[1].imshow(hr_rgb)
axs[1].set_title(f' Landsat 8 High Resolution Imagery Patch (512*512)', fontsize=10)
axs[1].axis('off')

# plt.tight_layout()

# 当前 py 文件所在文件夹
script_dir = os.path.dirname(os.path.abspath(__file__))

# 拼接输出路径
save_path = os.path.join(script_dir, f'comparison_{idx}.png')
plt.savefig(save_path, dpi=300)
plt.close()

print(f'对比图已保存为: {save_path}')
