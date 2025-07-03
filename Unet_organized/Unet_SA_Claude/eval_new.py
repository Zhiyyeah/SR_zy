#!/usr/bin/env python3
# validate_improved.py

import os
import glob
import time
import numpy as np
import rasterio
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional

# --- 用户配置参数 ---

# 数据路径
MODEL_PATH    = r"D:\Py_Code\Unet_SR\SR_zy\Unet_organized\Unet_SA_Claude\outputs\improved_model_v2_lr0.001_wd0.0001\models\best_psnr_model.pth"
OUTPUT_DIR    = r"D:\Py_Code\Unet_SR\SR_zy\Unet_organized\Unet_SA_Claude\outputs\improved_model_v2_lr0.001_wd0.0001\test_results"

LR_DIR        = r"D:\Py_Code\Unet_SR\SR_zy\Imagey\Small_Dataset\lr"
HR_DIR        = r"D:\Py_Code\Unet_SR\SR_zy\Imagey\Small_Dataset\hr"


# 验证参数
IDX = None              # 指定景序号(0-based)，None表示验证所有
BATCH_SIZE = 1          # 验证时的批次大小
NUM_WORKERS = 0         # 数据加载线程数

# 模型参数（需要与训练时一致）
UP_SCALE = 8
WIDTH = 64
DROPOUT_RATE = 0.15
USE_ATTENTION = True
USE_DEEP_SUPERVISION = True

# 可视化参数
RGB_CHANNELS = [3, 2, 1]    # 用于RGB可视化的波段（1-based）
SAVE_OUTPUTS = True         # 是否保存SR结果的TIFF文件
# -----------------------------------

# 导入模型
from model_attention_improved import UNetSAImproved

def load_model(path: str, device: torch.device, config: Dict) -> torch.nn.Module:
    """加载训练好的模型"""
    print(f"加载模型: {path}")
    
    # 创建模型
    model = UNetSAImproved(
        up_scale=config['up_scale'],
        img_channel=config['num_channels'],
        width=config['width'],
        use_attention=config['use_attention'],
        dropout_rate=config['dropout_rate'],
        deep_supervision=config['deep_supervision']
    ).to(device)
    
    # 加载权重
    checkpoint = torch.load(path, map_location=device)
    
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            # 新格式checkpoint
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ 模型加载成功 (Epoch: {checkpoint.get('epoch', 'unknown')})")
            
            # 显示训练时的指标
            if 'test_metrics' in checkpoint:
                metrics = checkpoint['test_metrics']
                print(f"训练时的测试指标:")
                print(f"  - Loss: {metrics.get('loss', 'N/A'):.4f}")
                print(f"  - PSNR: {metrics.get('psnr', 'N/A'):.2f} dB")
                print(f"  - SSIM: {metrics.get('ssim', 'N/A'):.4f}")
        else:
            # 尝试直接加载state_dict
            model.load_state_dict(checkpoint)
            print("✓ 模型加载成功")
    else:
        # 直接是state_dict
        model.load_state_dict(checkpoint)
        print("✓ 模型加载成功")
    
    model.eval()
    return model

def read_image(path: str, normalize: bool = True) -> Tuple[np.ndarray, rasterio.profiles.Profile]:
    """读取遥感图像和元数据"""
    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)
        profile = src.profile.copy()
        
        if normalize:
            # 假设TOA反射率范围在[0, 10000]
            data = data / 10000.0
            data = np.clip(data, 0, 1)
        
        return data, profile

def save_image(data: np.ndarray, profile: rasterio.profiles.Profile, path: str, denormalize: bool = True):
    """保存遥感图像"""
    if denormalize:
        # 反归一化
        data = data * 10000.0
        data = np.clip(data, 0, 10000)
    
    # 更新profile
    profile.update(
        height=data.shape[1],
        width=data.shape[2],
        count=data.shape[0],
        dtype='float32'
    )
    
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(data.astype(np.float32))

def calculate_metrics(hr: np.ndarray, sr: np.ndarray) -> Dict[str, float]:
    """计算PSNR和SSIM等评估指标"""
    # 确保数据在相同范围
    hr = np.clip(hr, 0, 1)
    sr = np.clip(sr, 0, 1)
    
    # 计算MSE和PSNR
    mse = np.mean((hr - sr) ** 2)
    if mse == 0:
        psnr = 100
    else:
        psnr = 20 * np.log10(1.0 / np.sqrt(mse))
    
    # 计算SSIM（简化版本）
    # 对于多波段图像，计算每个波段的SSIM然后取平均
    ssim_bands = []
    for i in range(hr.shape[0]):
        hr_band = hr[i]
        sr_band = sr[i]
        
        # 计算统计量
        mu1 = np.mean(hr_band)
        mu2 = np.mean(sr_band)
        sigma1_sq = np.var(hr_band)
        sigma2_sq = np.var(sr_band)
        sigma12 = np.mean((hr_band - mu1) * (sr_band - mu2))
        
        # SSIM常数
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        # 计算SSIM
        numerator = (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)
        denominator = (mu1 ** 2 + mu2 ** 2 + c1) * (sigma1_sq + sigma2_sq + c2)
        ssim_band = numerator / denominator
        ssim_bands.append(ssim_band)
    
    ssim = np.mean(ssim_bands)
    
    # 计算RMSE
    rmse = np.sqrt(mse)
    
    # 计算MAE
    mae = np.mean(np.abs(hr - sr))
    
    return {
        'psnr': psnr,
        'ssim': ssim,
        'rmse': rmse,
        'mae': mae,
        'mse': mse
    }

def validate_scene(
    lr_path: str, 
    hr_path: str, 
    model: torch.nn.Module, 
    device: torch.device,
    save_sr: bool = False,
    output_dir: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """验证单个场景"""
    # 读取图像
    lr_np, lr_profile = read_image(lr_path)
    hr_np, hr_profile = read_image(hr_path)
    
    print(f"  LR: {os.path.basename(lr_path)} - shape: {lr_np.shape}")
    print(f"  HR: {os.path.basename(hr_path)} - shape: {hr_np.shape}")
    
    # 转换为tensor
    lr_tensor = torch.from_numpy(lr_np).unsqueeze(0).to(device)
    hr_tensor = torch.from_numpy(hr_np).unsqueeze(0).to(device)
    
    # 模型推理
    with torch.no_grad():
        sr_tensor = model(lr_tensor)
        # 处理深度监督的情况
        if isinstance(sr_tensor, tuple):
            sr_tensor = sr_tensor[0]
    
    # 双线性插值基准
    interp_tensor = F.interpolate(
        lr_tensor,
        size=hr_tensor.shape[-2:],
        mode='bilinear',
        align_corners=False
    )
    
    # 转回numpy
    sr_np = sr_tensor.cpu().squeeze(0).numpy()
    interp_np = interp_tensor.cpu().squeeze(0).numpy()
    
    # 计算指标
    metrics_sr = calculate_metrics(hr_np, sr_np)
    metrics_interp = calculate_metrics(hr_np, interp_np)
    
    print(f"  SR模型 - PSNR: {metrics_sr['psnr']:.2f} dB, SSIM: {metrics_sr['ssim']:.4f}")
    print(f"  双线性 - PSNR: {metrics_interp['psnr']:.2f} dB, SSIM: {metrics_interp['ssim']:.4f}")
    print(f"  改善量 - PSNR: +{metrics_sr['psnr'] - metrics_interp['psnr']:.2f} dB")
    
    # 保存SR结果
    if save_sr and output_dir:
        sr_filename = os.path.basename(lr_path).replace('_lr', '_sr')
        sr_path = os.path.join(output_dir, 'sr_outputs', sr_filename)
        os.makedirs(os.path.dirname(sr_path), exist_ok=True)
        save_image(sr_np, hr_profile, sr_path)
        print(f"  已保存SR结果: {sr_path}")
    
    # 合并指标
    metrics = {
        'psnr_sr': metrics_sr['psnr'],
        'ssim_sr': metrics_sr['ssim'],
        'rmse_sr': metrics_sr['rmse'],
        'mae_sr': metrics_sr['mae'],
        'psnr_interp': metrics_interp['psnr'],
        'ssim_interp': metrics_interp['ssim'],
        'rmse_interp': metrics_interp['rmse'],
        'mae_interp': metrics_interp['mae']
    }
    
    return lr_np, interp_np, sr_np, hr_np, metrics

def plot_comparison(
    lr: np.ndarray,
    interp: np.ndarray,
    sr: np.ndarray,
    hr: np.ndarray,
    metrics: Dict,
    save_path: str,
    rgb_channels: List[int]
):
    """绘制对比图"""
    # 准备RGB显示
    rgb_idx = [ch - 1 for ch in rgb_channels]
    
    def to_rgb(arr):
        rgb = arr[rgb_idx, :, :].transpose(1, 2, 0)
        # 增强对比度
        vmin, vmax = np.percentile(rgb, (2, 98))
        rgb_scaled = np.clip((rgb - vmin) / (vmax - vmin + 1e-8), 0, 1)
        return rgb_scaled
    
    # 创建图形
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 第一行：原始图像
    axes[0, 0].imshow(to_rgb(lr))
    axes[0, 0].set_title(f'Low Resolution\n{lr.shape[1]}×{lr.shape[2]}', fontsize=12)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(to_rgb(interp))
    axes[0, 1].set_title(f'Bilinear Interpolation\nPSNR: {metrics["psnr_interp"]:.2f} dB', fontsize=12)
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(to_rgb(sr))
    axes[0, 2].set_title(f'Super Resolution (Ours)\nPSNR: {metrics["psnr_sr"]:.2f} dB', fontsize=12)
    axes[0, 2].axis('off')
    
    # 第二行：差异图和真值
    axes[1, 0].imshow(to_rgb(hr))
    axes[1, 0].set_title(f'Ground Truth\n{hr.shape[1]}×{hr.shape[2]}', fontsize=12)
    axes[1, 0].axis('off')
    
    # 误差图（SR vs GT）
    error_sr = np.abs(hr - sr)
    error_rgb = error_sr[rgb_idx, :, :].transpose(1, 2, 0)
    im = axes[1, 1].imshow(error_rgb, cmap='hot')
    axes[1, 1].set_title(f'Absolute Error (SR)\nRMSE: {metrics["rmse_sr"]:.4f}', fontsize=12)
    axes[1, 1].axis('off')
    plt.colorbar(im, ax=axes[1, 1], fraction=0.046)
    
    # 误差图（Bilinear vs GT）
    error_interp = np.abs(hr - interp)
    error_rgb_interp = error_interp[rgb_idx, :, :].transpose(1, 2, 0)
    im2 = axes[1, 2].imshow(error_rgb_interp, cmap='hot')
    axes[1, 2].set_title(f'Absolute Error (Bilinear)\nRMSE: {metrics["rmse_interp"]:.4f}', fontsize=12)
    axes[1, 2].axis('off')
    plt.colorbar(im2, ax=axes[1, 2], fraction=0.046)
    
    # 总标题
    fig.suptitle(
        f'Super-Resolution Results\n'
        f'PSNR Improvement: +{metrics["psnr_sr"] - metrics["psnr_interp"]:.2f} dB | '
        f'SSIM: {metrics["ssim_sr"]:.4f} vs {metrics["ssim_interp"]:.4f}',
        fontsize=16
    )
    
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

def plot_metrics_distribution(all_metrics: List[Dict], save_path: str):
    """绘制所有场景的指标分布图"""
    # 提取指标
    psnr_sr = [m['psnr_sr'] for m in all_metrics]
    psnr_interp = [m['psnr_interp'] for m in all_metrics]
    ssim_sr = [m['ssim_sr'] for m in all_metrics]
    ssim_interp = [m['ssim_interp'] for m in all_metrics]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # PSNR对比
    x = np.arange(len(all_metrics))
    width = 0.35
    
    ax1.bar(x - width/2, psnr_sr, width, label='Super Resolution', alpha=0.8)
    ax1.bar(x + width/2, psnr_interp, width, label='Bilinear', alpha=0.8)
    ax1.set_xlabel('Scene Index')
    ax1.set_ylabel('PSNR (dB)')
    ax1.set_title('PSNR Comparison Across Scenes')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # SSIM对比
    ax2.bar(x - width/2, ssim_sr, width, label='Super Resolution', alpha=0.8)
    ax2.bar(x + width/2, ssim_interp, width, label='Bilinear', alpha=0.8)
    ax2.set_xlabel('Scene Index')
    ax2.set_ylabel('SSIM')
    ax2.set_title('SSIM Comparison Across Scenes')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

def main():
    """主验证函数"""
    print("="*70)
    print("改进模型的超分辨率验证")
    print("="*70)
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    comparison_dir = os.path.join(OUTPUT_DIR, 'comparisons')
    os.makedirs(comparison_dir, exist_ok=True)
    
    # 获取文件列表
    lr_files = sorted(glob.glob(os.path.join(LR_DIR, "*.tif")))
    hr_files = sorted(glob.glob(os.path.join(HR_DIR, "*.tif")))
    
    print(f"\n数据集信息:")
    print(f"  LR图像数量: {len(lr_files)}")
    print(f"  HR图像数量: {len(hr_files)}")
    
    if len(lr_files) != len(hr_files):
        raise ValueError("LR和HR图像数量不匹配！")
    
    if len(lr_files) == 0:
        raise ValueError("未找到任何图像文件！")
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n计算设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU型号: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 读取第一个图像获取通道数
    sample_img, _ = read_image(lr_files[0])
    num_channels = sample_img.shape[0]
    print(f"\n图像信息:")
    print(f"  通道数: {num_channels}")
    print(f"  LR尺寸: {sample_img.shape[1]}×{sample_img.shape[2]}")
    
    # 模型配置
    model_config = {
        'up_scale': UP_SCALE,
        'num_channels': num_channels,
        'width': WIDTH,
        'use_attention': USE_ATTENTION,
        'dropout_rate': DROPOUT_RATE,
        'deep_supervision': USE_DEEP_SUPERVISION
    }
    
    # 加载模型
    print(f"\n加载模型中...")
    model = load_model(MODEL_PATH, device, model_config)
    
    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")
    
    # 确定要处理的图像
    if IDX is not None:
        indices = [IDX]
        print(f"\n仅验证场景 #{IDX}")
    else:
        indices = list(range(len(lr_files)))
        print(f"\n将验证所有 {len(indices)} 个场景")
    
    # 开始验证
    all_metrics = []
    start_time = time.time()
    
    print("\n开始验证...")
    print("-" * 70)
    
    for idx in tqdm(indices, desc="验证进度"):
        print(f"\n场景 #{idx}:")
        
        try:
            lr, interp, sr, hr, metrics = validate_scene(
                lr_files[idx],
                hr_files[idx],
                model,
                device,
                save_sr=SAVE_OUTPUTS,
                output_dir=OUTPUT_DIR
            )
            
            all_metrics.append(metrics)
            
            # 保存对比图
            comparison_path = os.path.join(comparison_dir, f'scene_{idx:04d}_comparison.png')
            plot_comparison(
                lr, interp, sr, hr, metrics,
                save_path=comparison_path,
                rgb_channels=RGB_CHANNELS
            )
            
        except Exception as e:
            print(f"  ❌ 处理失败: {str(e)}")
            continue
    
    # 计算总体统计
    if len(all_metrics) > 0:
        avg_psnr_sr = np.mean([m['psnr_sr'] for m in all_metrics])
        avg_ssim_sr = np.mean([m['ssim_sr'] for m in all_metrics])
        avg_rmse_sr = np.mean([m['rmse_sr'] for m in all_metrics])
        avg_mae_sr = np.mean([m['mae_sr'] for m in all_metrics])
        
        avg_psnr_interp = np.mean([m['psnr_interp'] for m in all_metrics])
        avg_ssim_interp = np.mean([m['ssim_interp'] for m in all_metrics])
        avg_rmse_interp = np.mean([m['rmse_interp'] for m in all_metrics])
        avg_mae_interp = np.mean([m['mae_interp'] for m in all_metrics])
        
        # 计算标准差
        std_psnr_sr = np.std([m['psnr_sr'] for m in all_metrics])
        std_ssim_sr = np.std([m['ssim_sr'] for m in all_metrics])
        
        # 打印结果
        print("\n" + "="*70)
        print("验证完成！")
        print("="*70)
        print(f"\n验证统计 ({len(all_metrics)} 个场景):")
        print(f"\n超分辨率模型:")
        print(f"  PSNR: {avg_psnr_sr:.2f} ± {std_psnr_sr:.2f} dB")
        print(f"  SSIM: {avg_ssim_sr:.4f} ± {std_ssim_sr:.4f}")
        print(f"  RMSE: {avg_rmse_sr:.4f}")
        print(f"  MAE:  {avg_mae_sr:.4f}")
        
        print(f"\n双线性插值:")
        print(f"  PSNR: {avg_psnr_interp:.2f} dB")
        print(f"  SSIM: {avg_ssim_interp:.4f}")
        print(f"  RMSE: {avg_rmse_interp:.4f}")
        print(f"  MAE:  {avg_mae_interp:.4f}")
        
        print(f"\n改善量:")
        print(f"  PSNR: +{avg_psnr_sr - avg_psnr_interp:.2f} dB")
        print(f"  SSIM: +{avg_ssim_sr - avg_ssim_interp:.4f}")
        print(f"  RMSE: -{avg_rmse_interp - avg_rmse_sr:.4f}")
        
        # 保存统计结果
        stats_path = os.path.join(OUTPUT_DIR, 'validation_statistics.txt')
        with open(stats_path, 'w') as f:
            f.write("超分辨率模型验证报告\n")
            f.write("="*70 + "\n\n")
            f.write(f"模型路径: {MODEL_PATH}\n")
            f.write(f"验证时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"验证场景数: {len(all_metrics)}\n")
            f.write(f"计算设备: {device}\n")
            f.write(f"\n模型配置:\n")
            for k, v in model_config.items():
                f.write(f"  {k}: {v}\n")
            
            f.write(f"\n总体指标:\n")
            f.write(f"\n超分辨率模型:\n")
            f.write(f"  PSNR: {avg_psnr_sr:.2f} ± {std_psnr_sr:.2f} dB\n")
            f.write(f"  SSIM: {avg_ssim_sr:.4f} ± {std_ssim_sr:.4f}\n")
            f.write(f"  RMSE: {avg_rmse_sr:.4f}\n")
            f.write(f"  MAE:  {avg_mae_sr:.4f}\n")
            
            f.write(f"\n双线性插值:\n")
            f.write(f"  PSNR: {avg_psnr_interp:.2f} dB\n")
            f.write(f"  SSIM: {avg_ssim_interp:.4f}\n")
            f.write(f"  RMSE: {avg_rmse_interp:.4f}\n")
            f.write(f"  MAE:  {avg_mae_interp:.4f}\n")
            
            f.write(f"\n相对改善:\n")
            f.write(f"  PSNR: +{avg_psnr_sr - avg_psnr_interp:.2f} dB ({(avg_psnr_sr/avg_psnr_interp - 1)*100:.1f}%)\n")
            f.write(f"  SSIM: +{avg_ssim_sr - avg_ssim_interp:.4f} ({(avg_ssim_sr/avg_ssim_interp - 1)*100:.1f}%)\n")
            f.write(f"  RMSE: -{avg_rmse_interp - avg_rmse_sr:.4f} ({(1 - avg_rmse_sr/avg_rmse_interp)*100:.1f}%)\n")
            
            f.write(f"\n\n详细结果:\n")
            f.write("-"*70 + "\n")
            for i, (idx, m) in enumerate(zip(indices, all_metrics)):
                f.write(f"\n场景 #{idx}:\n")
                f.write(f"  SR   - PSNR: {m['psnr_sr']:.2f} dB, SSIM: {m['ssim_sr']:.4f}, RMSE: {m['rmse_sr']:.4f}\n")
                f.write(f"  双线性 - PSNR: {m['psnr_interp']:.2f} dB, SSIM: {m['ssim_interp']:.4f}, RMSE: {m['rmse_interp']:.4f}\n")
                f.write(f"  改善  - PSNR: +{m['psnr_sr'] - m['psnr_interp']:.2f} dB\n")
        
        # 绘制指标分布图
        if len(all_metrics) > 1:
            dist_path = os.path.join(OUTPUT_DIR, 'metrics_distribution.png')
            plot_metrics_distribution(all_metrics, dist_path)
            print(f"\n已保存指标分布图: {dist_path}")
        
        print(f"\n验证结果已保存至: {OUTPUT_DIR}")
        
    else:
        print("\n❌ 没有成功验证任何场景！")
    
    elapsed_time = time.time() - start_time
    print(f"\n总耗时: {elapsed_time//60:.0f}分{elapsed_time%60:.0f}秒")

if __name__ == "__main__":
    main()