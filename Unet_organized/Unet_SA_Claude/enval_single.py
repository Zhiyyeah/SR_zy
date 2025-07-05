import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image
import glob
from tqdm import tqdm
import rasterio

# 导入必要的本地模块
from model_attention import UNetSA
from model_io import load_model
from utils import get_device

def bilinear_interpolation(lr_img, scale_factor):
    """
    对低分辨率图像进行双线性插值
    Args:
        lr_img: 低分辨率图像 tensor [B, C, H, W]
        scale_factor: 上采样倍数
    Returns:
        插值后的图像 tensor [B, C, H*scale, W*scale]
    """
    return F.interpolate(lr_img, scale_factor=scale_factor, mode='bilinear', align_corners=False)

def normalize_for_display(img):
    """
    将图像归一化到[0, 1]范围用于显示
    Args:
        img: numpy array
    Returns:
        归一化后的图像
    """
    img_min = img.min()
    img_max = img.max()
    if img_max > img_min:
        return (img - img_min) / (img_max - img_min)
    return img

def load_test_images(lr_dir, hr_dir, num_images=5):
    """
    加载测试图像
    Args:
        lr_dir: 低分辨率图像目录
        hr_dir: 高分辨率图像目录
        num_images: 要加载的图像数量
    Returns:
        lr_images, hr_images: 图像路径列表
    """
    lr_paths = sorted(glob.glob(os.path.join(lr_dir, "*.tif")))[:num_images]
    hr_paths = sorted(glob.glob(os.path.join(hr_dir, "*.tif")))[:num_images]
    
    if len(lr_paths) == 0:
        # 尝试其他格式
        lr_paths = sorted(glob.glob(os.path.join(lr_dir, "*.png")))[:num_images]
        hr_paths = sorted(glob.glob(os.path.join(hr_dir, "*.png")))[:num_images]
    
    if len(lr_paths) == 0:
        lr_paths = sorted(glob.glob(os.path.join(lr_dir, "*.jpg")))[:num_images]
        hr_paths = sorted(glob.glob(os.path.join(hr_dir, "*.jpg")))[:num_images]
    
    return lr_paths, hr_paths

def read_multi_channel_image(path):
    """
    读取多通道图像
    Args:
        path: 图像路径
    Returns:
        numpy array [C, H, W]
    """
    # 尝试使用rasterio读取（适用于GeoTIFF）
    try:
        with rasterio.open(path) as src:
            img_array = src.read()  # rasterio 默认返回 [C, H, W] 格式
        return img_array.astype(np.float32)
    except Exception as e:
        # 如果rasterio无法读取，尝试PIL（用于普通图像格式）
        try:
            img = Image.open(path)
            img_array = np.array(img)
            
            # 如果是灰度图，添加通道维度
            if len(img_array.shape) == 2:
                img_array = img_array[np.newaxis, :, :]
            # 如果是彩色图，转换维度顺序
            elif len(img_array.shape) == 3:
                img_array = np.transpose(img_array, (2, 0, 1))
            
            return img_array.astype(np.float32)
        except Exception as e2:
            raise ValueError(f"无法读取图像: {path}\nRasterio错误: {e}\nPIL错误: {e2}")

def visualize_single_image_all_bands(lr_img, sr_img, hr_img, save_path, image_name, up_scale):
    """
    可视化单张图像的所有波段
    Args:
        lr_img: 低分辨率图像 tensor [C, H, W]
        sr_img: 超分辨率图像 tensor [C, H, W]
        hr_img: 高分辨率图像 tensor [C, H, W]
        save_path: 保存路径
        image_name: 图像名称
        up_scale: 上采样倍数
    """
    num_channels = lr_img.shape[0]
    
    # 计算双线性插值结果
    lr_img_tensor = lr_img.unsqueeze(0)  # 添加batch维度
    bicubic_img = bilinear_interpolation(lr_img_tensor, up_scale).squeeze(0)
    
    # 转换为numpy数组
    lr_np = lr_img.cpu().numpy()
    bicubic_np = bicubic_img.cpu().numpy()
    sr_np = sr_img.cpu().numpy()
    hr_np = hr_img.cpu().numpy()
    
    # 创建更紧凑的图形
    fig = plt.figure(figsize=(20, 3 * num_channels))
    gs = GridSpec(num_channels, 4, figure=fig, hspace=0.1, wspace=0.03)
    
    for band_idx in range(num_channels):
        # 低分辨率图像
        ax1 = fig.add_subplot(gs[band_idx, 0])
        img1 = normalize_for_display(lr_np[band_idx])
        im1 = ax1.imshow(img1, cmap='viridis')
        ax1.set_title(f'Band {band_idx + 1} - LR ({lr_np[band_idx].shape[0]}×{lr_np[band_idx].shape[1]})', fontsize=9)
        ax1.axis('off')
        # 添加colorbar
        cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        cbar1.ax.tick_params(labelsize=7)
        
        # 双线性插值结果
        ax2 = fig.add_subplot(gs[band_idx, 1])
        img2 = normalize_for_display(bicubic_np[band_idx])
        im2 = ax2.imshow(img2, cmap='viridis')
        ax2.set_title(f'Band {band_idx + 1} - Bilinear ({bicubic_np[band_idx].shape[0]}×{bicubic_np[band_idx].shape[1]})', fontsize=9)
        ax2.axis('off')
        # 添加colorbar
        cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        cbar2.ax.tick_params(labelsize=7)
        
        # 超分辨率结果
        ax3 = fig.add_subplot(gs[band_idx, 2])
        img3 = normalize_for_display(sr_np[band_idx])
        im3 = ax3.imshow(img3, cmap='viridis')
        ax3.set_title(f'Band {band_idx + 1} - SR ({sr_np[band_idx].shape[0]}×{sr_np[band_idx].shape[1]})', fontsize=9)
        ax3.axis('off')
        # 添加colorbar
        cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        cbar3.ax.tick_params(labelsize=7)
        
        # 高分辨率图像（Ground Truth）
        ax4 = fig.add_subplot(gs[band_idx, 3])
        img4 = normalize_for_display(hr_np[band_idx])
        im4 = ax4.imshow(img4, cmap='viridis')
        ax4.set_title(f'Band {band_idx + 1} - HR ({hr_np[band_idx].shape[0]}×{hr_np[band_idx].shape[1]})', fontsize=9)
        ax4.axis('off')
        # 添加colorbar
        cbar4 = plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
        cbar4.ax.tick_params(labelsize=7)
    
    plt.suptitle(f'Multi-band Super-Resolution Results - {image_name}', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(save_path, f'{image_name}_all_bands.png'), dpi=150, bbox_inches='tight')
    plt.close()

def calculate_metrics(sr_img, hr_img):
    """
    计算PSNR和SSIM指标
    Args:
        sr_img: 超分辨率图像 tensor
        hr_img: 高分辨率图像 tensor
    Returns:
        psnr_value, ssim_value
    """
    # 这里简单实现PSNR，实际使用时应该用完整的实现
    mse = torch.mean((sr_img - hr_img) ** 2)
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    
    # SSIM需要更复杂的实现，这里暂时返回0
    ssim = 0.0
    
    return psnr.item(), ssim

def validate_model(model_path, lr_dir, hr_dir, output_dir, num_images=5, up_scale=8, device='cuda'):
    """
    验证模型主函数
    Args:
        model_path: 模型权重路径
        lr_dir: 低分辨率图像目录
        hr_dir: 高分辨率图像目录
        output_dir: 输出目录
        num_images: 要验证的图像数量
        up_scale: 上采样倍数
        device: 设备
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载模型
    print("加载模型...")
    model = UNetSA(
        up_scale=up_scale,
        img_channel=7,
        width=64,
        use_attention=True,
        dropout_p=0.0  # 测试时不使用dropout
    ).to(device)
    
    # 加载权重
    load_model(model, model_path, device)
    model.eval()
    print("模型加载完成！")
    
    # 加载测试图像
    print("加载测试图像...")
    lr_paths, hr_paths = load_test_images(lr_dir, hr_dir, num_images)
    
    if len(lr_paths) == 0:
        print("错误：未找到测试图像！")
        return
    
    print(f"找到 {len(lr_paths)} 张测试图像")
    
    # 存储所有图像的指标
    all_psnr = []
    all_ssim = []
    
    # 处理每张图像
    with torch.no_grad():
        for idx, (lr_path, hr_path) in enumerate(tqdm(zip(lr_paths, hr_paths), desc="处理图像")):
            # 读取图像
            lr_img = read_multi_channel_image(lr_path)
            hr_img = read_multi_channel_image(hr_path)
            
            # 转换为tensor
            lr_tensor = torch.from_numpy(lr_img).unsqueeze(0).to(device)  # [1, C, H, W]
            hr_tensor = torch.from_numpy(hr_img).unsqueeze(0).to(device)
            
            # 归一化（如果训练时有归一化）
            # lr_tensor = lr_tensor / 255.0  # 根据实际情况调整
            # hr_tensor = hr_tensor / 255.0
            
            # 推理
            sr_tensor = model(lr_tensor)
            
            # 去除batch维度
            lr_tensor = lr_tensor.squeeze(0)
            sr_tensor = sr_tensor.squeeze(0)
            hr_tensor = hr_tensor.squeeze(0)
            
            # 计算指标
            psnr_val, ssim_val = calculate_metrics(sr_tensor, hr_tensor)
            all_psnr.append(psnr_val)
            all_ssim.append(ssim_val)
            
            # 获取图像名称
            image_name = os.path.splitext(os.path.basename(lr_path))[0]
            
            # 可视化
            visualize_single_image_all_bands(
                lr_tensor, sr_tensor, hr_tensor, 
                output_dir, image_name, up_scale
            )
            
            print(f"图像 {idx+1}/{len(lr_paths)}: {image_name}, PSNR: {psnr_val:.2f} dB")
    
    # 计算平均指标
    avg_psnr = np.mean(all_psnr)
    avg_ssim = np.mean(all_ssim)
    
    print(f"\n验证完成！")
    print(f"平均 PSNR: {avg_psnr:.2f} dB")
    print(f"平均 SSIM: {avg_ssim:.4f}")
    print(f"结果保存在: {output_dir}")
    
    # 保存指标到文件
    metrics_file = os.path.join(output_dir, 'validation_metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write(f"验证结果统计\n")
        f.write(f"===============\n")
        f.write(f"模型路径: {model_path}\n")
        f.write(f"测试图像数量: {len(lr_paths)}\n")
        f.write(f"平均 PSNR: {avg_psnr:.2f} dB\n")
        f.write(f"平均 SSIM: {avg_ssim:.4f}\n")
        f.write(f"\n详细结果:\n")
        for idx, (lr_path, psnr, ssim) in enumerate(zip(lr_paths, all_psnr, all_ssim)):
            image_name = os.path.splitext(os.path.basename(lr_path))[0]
            f.write(f"{idx+1}. {image_name}: PSNR={psnr:.2f} dB\n")

def main():
    """主函数"""
    # 配置参数
    experiment_name = 'drop_lr_first'
    learning_rate = 0.00043
    
    # 路径设置
    base_dir = './outputs'
    experiment_dir = os.path.join(base_dir, f"{experiment_name}_{learning_rate}")
    model_dir = os.path.join(experiment_dir, 'models')
    
    # 模型路径（可以选择best_model.pth或final_model.pth）
    model_path = os.path.join(model_dir, 'best_model.pth')
    
    # 数据路径
    lr_dir = r"D:\Py_Code\Unet_SR\SR_zy\Imagey\Imagery_WaterLand\WaterLand_TOA_tiles_lr"
    hr_dir = r"D:\Py_Code\Unet_SR\SR_zy\Imagey\Imagery_WaterLand\WaterLand_TOA_tiles_hr"
    
    # 输出路径
    output_dir = os.path.join(experiment_dir, 'validation_results')
    
    # 验证参数
    num_images = 10  # 要验证的图像数量
    up_scale = 8    # 上采样倍数
    device = get_device()
    
    # 运行验证
    validate_model(
        model_path=model_path,
        lr_dir=lr_dir,
        hr_dir=hr_dir,
        output_dir=output_dir,
        num_images=num_images,
        up_scale=up_scale,
        device=device
    )

if __name__ == "__main__":
    main()