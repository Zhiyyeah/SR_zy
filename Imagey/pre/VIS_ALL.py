import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# 设置目录路径
lr_dir = '/public/home/zyye/SR_organized/Imagey/WaterLand_TOA_tiles_lr'
hr_dir = '/public/home/zyye/SR_organized/Imagey/WaterLand_TOA_tiles_hr'
output_dir = '/public/home/zyye/SR_organized/Imagey/WaterLand_TOA_tiles_VIS'

# 创建输出目录（如果不存在）
os.makedirs(output_dir, exist_ok=True)

# RGB 波段索引
rgb_band_idx = [4, 3, 2]  # Landsat 8 的 B4, B3, B2

def read_rgb_image_improved(path, band_idx, method='percentile'):
    """
    改进的RGB图像读取函数，提供多种归一化方法
    
    Parameters:
    path: 文件路径
    band_idx: 波段索引
    method: 归一化方法 ('percentile', 'minmax', 'std', 'gamma')
    """
    try:
        with rasterio.open(path) as src:
            img = src.read(band_idx).astype(np.float32)
        
        if method == 'percentile':
            # 方法1: 使用2%和98%分位数进行拉伸
            p2, p98 = np.percentile(img, (2, 98))
            img = np.clip(img, p2, p98)
            img = (img - p2) / (p98 - p2 + 1e-8)
            
        elif method == 'minmax':
            # 方法2: 最小最大值归一化
            img_min, img_max = img.min(), img.max()
            img = (img - img_min) / (img_max - img_min + 1e-8)
            
        elif method == 'std':
            # 方法3: 标准差拉伸
            img_mean = img.mean()
            img_std = img.std()
            img = np.clip(img, img_mean - 2*img_std, img_mean + 2*img_std)
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            
        elif method == 'gamma':
            # 方法4: Gamma校正 + 百分位数拉伸
            p1, p99 = np.percentile(img, (2, 98))
            img = np.clip(img, p1, p99)
            img = (img - p1) / (p99 - p1 + 1e-8)
            # 应用gamma校正增强对比度
            img = np.power(img, 0.7)
        
        # 确保值在[0,1]范围内
        img = np.clip(img, 0, 1)
        
        return np.transpose(img, (1, 2, 0))  # (C, H, W) -> (H, W, C)
    
    except Exception as e:
        print(f"读取图像时出错: {path}, 错误: {e}")
        return None

def create_comparison_plot(lr_rgb, hr_rgb, filename, idx, save_dir):
    """创建对比图并保存"""
    fig, axs = plt.subplots(1, 2, figsize=(14, 7))
    
    # 显示低分辨率图像
    axs[0].imshow(lr_rgb)
    axs[0].set_title(f'Low Resolution (64×64)\n{filename}', fontsize=11, pad=10)
    axs[0].axis('off')
    
    # 显示高分辨率图像
    axs[1].imshow(hr_rgb)
    axs[1].set_title(f'High Resolution (512×512)\n{filename}', fontsize=11, pad=10)
    axs[1].axis('off')
    
    plt.tight_layout()
    
    # 保存图片
    base_name = os.path.splitext(filename)[0]
    save_filename = f'comparison_{idx:04d}_{base_name}.png'
    save_path = os.path.join(save_dir, save_filename)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return save_path

# 获取文件列表
filenames = sorted([f for f in os.listdir(hr_dir) if f.endswith('.tif')])
print(f"总共找到 {len(filenames)} 个影像文件")

# 可以选择不同的归一化方法进行测试
normalization_methods = ['percentile', 'gamma', 'std', 'minmax']
current_method = 'gamma'  # 推荐使用gamma方法

print(f"使用归一化方法: {current_method}")

# 处理所有文件
success_count = 0
error_count = 0

for idx, filename in enumerate(tqdm(filenames, desc="处理影像")):
    try:
        lr_path = os.path.join(lr_dir, filename)
        hr_path = os.path.join(hr_dir, filename)
        
        # 检查文件是否存在
        if not os.path.exists(lr_path) or not os.path.exists(hr_path):
            print(f"文件不存在: {filename}")
            error_count += 1
            continue
        
        # 使用改进的方法读取图像
        lr_rgb = read_rgb_image_improved(lr_path, rgb_band_idx, method=current_method)
        hr_rgb = read_rgb_image_improved(hr_path, rgb_band_idx, method=current_method)
        
        if lr_rgb is None or hr_rgb is None:
            print(f"读取失败: {filename}")
            error_count += 1
            continue
        
        # 创建并保存对比图
        save_path = create_comparison_plot(lr_rgb, hr_rgb, filename, idx, output_dir)
        success_count += 1
        
        # 每处理50个文件输出一次进度
        if (idx + 1) % 50 == 0:
            print(f"已处理 {idx + 1}/{len(filenames)} 个文件")
            
    except Exception as e:
        print(f"处理文件 {filename} 时出错: {e}")
        error_count += 1
        plt.close('all')
        continue

print(f"\n处理完成!")
print(f"成功处理: {success_count} 个文件")
print(f"失败文件: {error_count} 个文件")
print(f"使用的归一化方法: {current_method}")
print(f"结果保存在: {output_dir}")

# 如果你想测试单个文件的不同归一化效果，可以使用以下代码：
def test_normalization_methods(test_filename):
    """测试不同归一化方法的效果"""
    test_hr_path = os.path.join(hr_dir, test_filename)
    
    if not os.path.exists(test_hr_path):
        print(f"测试文件不存在: {test_filename}")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    
    for i, method in enumerate(normalization_methods):
        img = read_rgb_image_improved(test_hr_path, rgb_band_idx, method=method)
        if img is not None:
            axes[i].imshow(img)
            axes[i].set_title(f'Method: {method}')
            axes[i].axis('off')
    
    plt.tight_layout()
    test_save_path = os.path.join(output_dir, 'normalization_comparison.png')
    plt.savefig(test_save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"归一化方法对比图保存为: {test_save_path}")

# 取消注释下面这行来测试不同归一化方法的效果
# test_normalization_methods(filenames[0] if filenames else None)