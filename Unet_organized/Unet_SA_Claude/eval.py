#!/usr/bin/env python3
# validate_sr_fixed.py

import os
import glob
import time
import numpy as np
import rasterio
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- 用户请在此处修改路径和参数 ---
LR_DIR        = "../../Imagey/Imagery_WaterLand/WaterLand_TOA_tiles_lr"
HR_DIR        = "../../Imagey/Imagery_WaterLand/WaterLand_TOA_tiles_hr"
MODEL_PATH    = r"D:\Py_Code\Unet_SR\SR_zy\Unet_organized\Unet_SA_Claude\outputs\improved_model_v2_lr0.001_wd0.0001\models\best_psnr_model.pth"
OUTPUT_DIR    = r"D:\Py_Code\Unet_SR\SR_zy\Unet_organized\Unet_SA_Claude\outputs\improved_model_v2_lr0.001_wd0.0001\test_results"
IDX           = 68           # 指定景序号(0-based)，不想遍历所有则设为 int，否则设为 None
UP_SCALE      = 8
WIDTH         = 32            # 如果你使用了改进的模型，这里应该是32
DROPOUT_RATE  = 0.1          # 改进模型的dropout率
RGB_CHANNELS  = [3, 2, 1]    # 1-based 波段序号，用于 RGB 可视化
# -----------------------------------
from model_attention import UNetSA

def load_model(path, device, up_scale, num_channels, width, dropout_rate=0.1):
    """加载模型，兼容新旧checkpoint格式"""
    # 创建模型 - 如果使用了改进的模型，需要添加dropout_rate参数
    try:
        # 尝试使用新的模型参数
        model = UNetSA(
            up_scale=up_scale, 
            img_channel=num_channels, 
            width=width,
            dropout_rate=dropout_rate,
            use_attention=True
        ).to(device)
    except TypeError:
        # 如果是旧版本模型，不需要dropout_rate
        print("使用旧版本模型结构")
        model = UNetSA(
            up_scale=up_scale, 
            img_channel=num_channels, 
            width=width
        ).to(device)
    
    # 加载checkpoint
    print(f"加载模型: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    
    # 检查checkpoint格式
    if isinstance(ckpt, dict):
        if 'model_state_dict' in ckpt:
            # 新格式的checkpoint
            print(f"检测到新格式checkpoint，Epoch: {ckpt.get('epoch', 'unknown')}")
            model.load_state_dict(ckpt['model_state_dict'])
            
            # 打印保存的训练指标
            if 'test_metrics' in ckpt:
                metrics = ckpt['test_metrics']
                print(f"保存时的测试指标:")
                print(f"  - Loss: {metrics.get('loss', 'N/A'):.4f}")
                print(f"  - PSNR: {metrics.get('psnr', 'N/A'):.2f} dB")
                print(f"  - SSIM: {metrics.get('ssim', 'N/A'):.4f}")
        else:
            # 可能是旧格式，尝试直接加载
            model.load_state_dict(ckpt)
            print("加载模型成功（旧格式）")
    else:
        # 直接是state_dict
        model.load_state_dict(ckpt)
        print("加载模型成功（纯state_dict）")
    
    model.eval()
    return model

def read_image(path):
    """读取遥感图像"""
    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)
        # 归一化到[0,1]范围（根据你的数据范围调整）
        # 假设数据是TOA反射率，范围大概在[0, 10000]
        data = data / 10000.0  # 根据实际情况调整
        return data

def calculate_metrics(hr, sr):
    """计算PSNR和SSIM"""
    # 计算MSE和PSNR
    mse = np.mean((hr - sr) ** 2)
    if mse == 0:
        psnr = 100
    else:
        max_pixel = 1.0  # 假设已归一化到[0,1]
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    
    # 简单的SSIM计算（可以用skimage.metrics.structural_similarity代替）
    # 这里只计算一个简化版本
    mean_hr = np.mean(hr)
    mean_sr = np.mean(sr)
    var_hr = np.var(hr)
    var_sr = np.var(sr)
    cov = np.mean((hr - mean_hr) * (sr - mean_sr))
    
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    
    ssim = ((2 * mean_hr * mean_sr + c1) * (2 * cov + c2)) / \
           ((mean_hr ** 2 + mean_sr ** 2 + c1) * (var_hr + var_sr + c2))
    
    return psnr, ssim

def validate_scene(lr_path, hr_path, model, device):
    """验证单个场景"""
    print(f"  读取LR: {os.path.basename(lr_path)}")
    lr_np = read_image(lr_path)
    print(f"  读取HR: {os.path.basename(hr_path)}")
    hr_np = read_image(hr_path)
    
    print(f"  LR shape: {lr_np.shape}, range: [{lr_np.min():.3f}, {lr_np.max():.3f}]")
    print(f"  HR shape: {hr_np.shape}, range: [{hr_np.min():.3f}, {hr_np.max():.3f}]")

    # 转换为tensor
    lr_t = torch.from_numpy(lr_np).unsqueeze(0).to(device)
    hr_t = torch.from_numpy(hr_np).unsqueeze(0).to(device)
    
    # 模型推理
    with torch.no_grad():
        sr_t = model(lr_t)
    
    # 双线性插值基准
    interp_t = F.interpolate(
        lr_t,
        size=hr_t.shape[-2:],
        mode="bilinear",
        align_corners=False
    )
    
    # 转回numpy
    sr_np = sr_t.cpu().squeeze(0).numpy()
    interp_np = interp_t.cpu().squeeze(0).numpy()
    
    # 计算指标
    psnr_sr, ssim_sr = calculate_metrics(hr_np, sr_np)
    psnr_interp, ssim_interp = calculate_metrics(hr_np, interp_np)
    
    print(f"  SR结果 - PSNR: {psnr_sr:.2f} dB, SSIM: {ssim_sr:.4f}")
    print(f"  双线性 - PSNR: {psnr_interp:.2f} dB, SSIM: {ssim_interp:.4f}")
    print(f"  改善: PSNR +{psnr_sr - psnr_interp:.2f} dB")
    
    return lr_np, interp_np, sr_np, hr_np, {
        'psnr_sr': psnr_sr,
        'ssim_sr': ssim_sr,
        'psnr_interp': psnr_interp,
        'ssim_interp': ssim_interp
    }

def plot_comparison(
    lr: np.ndarray,
    interp: np.ndarray,
    sr: np.ndarray,
    hr: np.ndarray,
    metrics: dict,
    save_path: str,
    rgb_channels: list[int]
):
    """绘制对比图"""
    idx = [b-1 for b in rgb_channels]
    def to_rgb(arr):
        rgb = arr[idx,:,:].transpose(1,2,0)
        # 增强对比度以便更好地可视化
        vmin, vmax = np.percentile(rgb, (2, 98))
        return np.clip((rgb-vmin)/(vmax-vmin), 0, 1)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()

    titles = [
        f"LowRes {lr.shape[1]}×{lr.shape[2]}",
        f"Bilinear {hr.shape[1]}×{hr.shape[2]}\nPSNR: {metrics['psnr_interp']:.2f} dB",
        f"Super-Res {sr.shape[1]}×{sr.shape[2]}\nPSNR: {metrics['psnr_sr']:.2f} dB",
        f"GroundTruth {hr.shape[1]}×{hr.shape[2]}"
    ]
    imgs = [lr, interp, sr, hr]

    for ax, img, ttl in zip(axes, imgs, titles):
        ax.imshow(to_rgb(img))
        ax.set_title(ttl, fontsize=10)
        ax.axis("off")

    fig.suptitle(
        f"Super-Resolution Results Comparison\n"
        f"PSNR Improvement: +{metrics['psnr_sr'] - metrics['psnr_interp']:.2f} dB", 
        fontsize=14
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

def plot_density_scatter_linear(hr: np.ndarray, sr: np.ndarray, save_path: str):
    """绘制密度散点图"""
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    import numpy as np

    C = hr.shape[0]
    ncol = 4
    nrow = int(np.ceil(C / ncol))
    cell = 3.2
    cbar_ratio = 0.2

    fig = plt.figure(figsize=(ncol*cell + cbar_ratio*cell, nrow*cell))
    gs = gridspec.GridSpec(
        nrow, ncol+1,
        width_ratios=[*([1]*ncol), cbar_ratio],
        height_ratios=[1]*nrow,
        wspace=0.6, hspace=0.6,
        figure=fig
    )

    hb0 = None
    axes = []

    for b in range(C):
        r, c = divmod(b, ncol)
        ax = fig.add_subplot(gs[r, c])
        axes.append(ax)

        x = hr[b].ravel()
        y = sr[b].ravel()

        # 计算 RMSE 和 R²
        mse  = np.mean((y - x)**2)
        rmse = np.sqrt(mse)
        ssr  = np.sum((y - x)**2)
        sst  = np.sum((x - x.mean())**2)
        r2   = 1 - ssr / sst if sst > 0 else np.nan

        # 使用hexbin绘制密度图
        hb = ax.hexbin(
            x, y,
            gridsize=100,
            cmap='YlOrRd',
            mincnt=1
        )
        if hb0 is None:
            hb0 = hb

        # 设置坐标轴范围
        data_min = min(x.min(), y.min())
        data_max = max(x.max(), y.max())
        margin = (data_max - data_min) * 0.05
        
        ax.set_xlim(data_min - margin, data_max + margin)
        ax.set_ylim(data_min - margin, data_max + margin)

        # 绘制1:1参考线
        ax.plot([data_min, data_max], [data_min, data_max], 'k--', lw=1, alpha=0.5)

        # 添加统计信息
        txt = (
            f"RMSE: {rmse:.3e}\n"
            f"R²: {r2:.3f}"
        )
        ax.text(
            0.05, 0.95, txt,
            transform=ax.transAxes,
            va='top', ha='left',
            fontsize=8,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8)
        )

        ax.set_title(f"Band {b+1}", fontsize=10)
        ax.set_xlabel("Ground Truth", fontsize=9)
        ax.set_ylabel("Super Resolution", fontsize=9)
        ax.grid(True, alpha=0.3)

    # 隐藏多余的子图
    for b in range(C, nrow*ncol):
        r, c = divmod(b, ncol)
        fig.add_subplot(gs[r, c]).set_visible(False)

    # 添加颜色条
    cax = fig.add_subplot(gs[:, -1])
    cb = fig.colorbar(hb0, cax=cax, orientation='vertical')
    cb.set_label('Point Density', fontsize=10)

    fig.suptitle("Band-wise SR vs GT Density Scatter", fontsize=14)
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

def main():
    """主函数"""
    print("="*60)
    print("超分辨率模型验证")
    print("="*60)
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 获取文件列表
    lr_list = sorted(glob.glob(os.path.join(LR_DIR, "*.tif")))
    hr_list = sorted(glob.glob(os.path.join(HR_DIR, "*.tif")))
    
    print(f"\n找到 {len(lr_list)} 个LR文件")
    print(f"找到 {len(hr_list)} 个HR文件")
    
    assert len(lr_list) == len(hr_list), "LR/HR 数量不匹配！"
    assert len(lr_list) > 0, "没有找到图像文件！"
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")
    
    # 读取第一个图像以获取波段数
    tmp = read_image(lr_list[0])
    num_bands = tmp.shape[0]
    print(f"检测到 {num_bands} 个波段")
    
    # 加载模型
    print(f"\n加载模型...")
    model = load_model(MODEL_PATH, device, UP_SCALE, num_bands, WIDTH, DROPOUT_RATE)
    
    # 确定要处理的图像索引
    if IDX is not None:
        indices = [IDX]
        print(f"\n只处理场景 #{IDX}")
    else:
        indices = list(range(len(lr_list)))
        print(f"\n将处理所有 {len(indices)} 个场景")
    
    # 收集所有指标
    all_metrics = []
    
    # 处理每个场景
    for i in tqdm(indices, desc="处理场景"):
        print(f"\n>>> 场景 #{i}")
        
        lr, interp, sr, hr, metrics = validate_scene(
            lr_list[i], hr_list[i], model, device
        )
        
        all_metrics.append(metrics)
        
        # 保存对比图
        cmp_path = os.path.join(OUTPUT_DIR, f"scene_{i:03d}_comparison.png")
        plot_comparison(
            lr, interp, sr, hr, metrics,
            save_path=cmp_path,
            rgb_channels=RGB_CHANNELS
        )
        
        # 保存散点图
        sca_path = os.path.join(OUTPUT_DIR, f"scene_{i:03d}_scatter.png")
        plot_density_scatter_linear(hr, sr, sca_path)
    
    # 计算平均指标
    if len(all_metrics) > 0:
        avg_psnr_sr = np.mean([m['psnr_sr'] for m in all_metrics])
        avg_ssim_sr = np.mean([m['ssim_sr'] for m in all_metrics])
        avg_psnr_interp = np.mean([m['psnr_interp'] for m in all_metrics])
        avg_ssim_interp = np.mean([m['ssim_interp'] for m in all_metrics])
        
        print("\n" + "="*60)
        print("验证完成！")
        print(f"\n平均指标 ({len(all_metrics)} 个场景):")
        print(f"  超分辨率 - PSNR: {avg_psnr_sr:.2f} dB, SSIM: {avg_ssim_sr:.4f}")
        print(f"  双线性   - PSNR: {avg_psnr_interp:.2f} dB, SSIM: {avg_ssim_interp:.4f}")
        print(f"  平均改善 - PSNR: +{avg_psnr_sr - avg_psnr_interp:.2f} dB")
        print("="*60)
        
        # 保存指标到文件
        metrics_path = os.path.join(OUTPUT_DIR, "validation_metrics.txt")
        with open(metrics_path, 'w') as f:
            f.write("验证结果汇总\n")
            f.write("="*60 + "\n")
            f.write(f"模型路径: {MODEL_PATH}\n")
            f.write(f"处理场景数: {len(all_metrics)}\n")
            f.write(f"\n平均指标:\n")
            f.write(f"  超分辨率 - PSNR: {avg_psnr_sr:.2f} dB, SSIM: {avg_ssim_sr:.4f}\n")
            f.write(f"  双线性   - PSNR: {avg_psnr_interp:.2f} dB, SSIM: {avg_ssim_interp:.4f}\n")
            f.write(f"  平均改善 - PSNR: +{avg_psnr_sr - avg_psnr_interp:.2f} dB\n")
            f.write("\n各场景详细指标:\n")
            for i, m in enumerate(all_metrics):
                f.write(f"\n场景 #{indices[i]}:\n")
                f.write(f"  SR - PSNR: {m['psnr_sr']:.2f} dB, SSIM: {m['ssim_sr']:.4f}\n")
                f.write(f"  双线性 - PSNR: {m['psnr_interp']:.2f} dB, SSIM: {m['ssim_interp']:.4f}\n")

if __name__ == "__main__":
    main()