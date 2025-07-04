import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import rasterio

def load_image(image_path):
    """
    从路径加载图像
    参数:
        image_path: 图像文件路径
    返回:
        形状为(C, H, W)的张量
    """
    with rasterio.open(image_path) as src:
        image = src.read().astype(np.float32)
    return torch.tensor(image)

def bilinear_interpolation(image, target_size):
    """
    对图像进行双线性插值
    参数:
        image: 输入张量，形状为(B, C, H, W)
        target_size: 目标大小(H, W)
        
    返回:
        插值后的张量，形状为(B, C, target_size[0], target_size[1])
    """
    return F.interpolate(image, size=target_size, mode='bilinear', align_corners=False)

def visualize_results(lr_img, bilinear_img, sr_img, hr_img, metrics, save_path, rgb_channels=[3, 2, 1]):
    """
    Save comparison visualization of different images
    Args:
        lr_img: Low-resolution input image
        bilinear_img: Bilinear interpolated image
        sr_img: Super-resolved output image
        hr_img: High-resolution ground truth image
        metrics: Dictionary containing metrics
        save_path: Path to save the visualization
        rgb_channels: RGB channel indices
    """
    imgs = [lr_img, bilinear_img, sr_img, hr_img]
    titles = [
        'LR Input',
        f'Bilinear\nPSNR: {metrics["bilinear_psnr"]:.2f}dB\nSSIM: {metrics["bilinear_ssim"]:.4f}',
        f'SR Output\nPSNR: {metrics["sr_psnr"]:.2f}dB\nSSIM: {metrics["sr_ssim"]:.4f}',
        'HR Ground Truth'
    ]
    
    def normalize(img):
        rgb = img.squeeze().cpu().numpy()[rgb_channels, :, :]
        norm = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-6)
        return np.clip(norm, 0, 1).transpose(1, 2, 0)
    
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    for ax, img, title in zip(axes, imgs, titles):
        ax.imshow(normalize(img))
        ax.set_title(title)
        ax.axis('off')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

def save_plots(train_metrics, val_metrics, output_dir):
    """
    保存训练和验证指标图表
    
    参数:
        train_metrics: 包含训练指标的字典
        val_metrics: 包含验证指标的字典
        output_dir: 保存图表的目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 损失图表
    plt.figure(figsize=(10, 7))
    plt.plot(train_metrics['loss'], color='orange', label='Train Loss')
    plt.plot(val_metrics['loss'], color='red', label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'loss.png'))
    plt.close()
    
    # PSNR图表
    plt.figure(figsize=(10, 7))
    plt.plot(train_metrics['psnr'], color='orange', label='Train PSNR (dB)')
    plt.plot(val_metrics['psnr'], color='red', label='Test PSNR (dB)')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR (dB)')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'psnr.png'))
    plt.close()
    
    # SSIM图表
    plt.figure(figsize=(10, 7))
    plt.plot(train_metrics['ssim'], color='orange', label='Train SSIM')
    plt.plot(val_metrics['ssim'], color='red', label='Test SSIM')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'ssim.png'))
    plt.close()

def get_device():
    """
    获取可用设备(CUDA或CPU)
    
    返回:
        torch.device: 可用设备
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

def save_metrics_to_file(metrics, file_path):
    """
    将指标保存到文本文件
    
    参数:
        metrics: 包含指标的字典
        file_path: 保存指标的文件路径
    """
    with open(file_path, "w") as f:
        for k, v in metrics.items():
            if isinstance(v, list):
                f.write(f"{k}: {np.mean(v):.4f}\n")
            else:
                f.write(f"{k}: {v:.4f}\n")
