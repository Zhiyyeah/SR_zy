import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from tqdm import tqdm
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

# 导入本地模块
from model_attention import UNetSA
from data_loader import create_train_val_test_dataloaders
from metrics import psnr, compute_ssim
from utils import visualize_results, save_plots, save_metrics_to_file, get_device
from model_io import save_model, load_model

# ====================== 配置参数 ======================
experiment_name = 'zy_first_optimized'
# 数据设置
lr_dir = r"D:\Py_Code\Unet_SR\SR_zy\Imagey\Imagery_WaterLand\WaterLand_TOA_tiles_lr"
hr_dir = r"D:\Py_Code\Unet_SR\SR_zy\Imagey\Imagery_WaterLand\WaterLand_TOA_tiles_hr"
train_ratio = 0.8

# 模型设置
up_scale = 8
width = 64
dropout_rate = 0.05  # Dropout率（模型内部使用）

# 训练设置
batch_size = 16
num_workers = 4
pin_memory = True
seed = 42
epochs = 100
learning_rate = 0.00043
weight_decay = 0.0001  # 添加L2正则化

# 学习率调度器设置
lr_scheduler_type = 'ReduceLROnPlateau'  # 可选: 'ReduceLROnPlateau', 'CosineAnnealing'
lr_patience = 5  # ReduceLROnPlateau的patience
lr_factor = 0.5  # ReduceLROnPlateau的factor
lr_min = 1e-6  # 最小学习率

# 设备设置
device = get_device()

# 输出设置
output_dir = './outputs'
save_interval = 5

# 可视化设置
rgb_channels = [3, 2, 1]

# ====================== 辅助函数 ======================

def save_training_history(history_df, save_path):
    """保存训练历史为CSV文件"""
    history_df.to_csv(save_path, index=False)
    print(f"训练历史已保存到: {save_path}")

def train_epoch(model, dataloader, criterion, optimizer):
    """训练一个轮次"""
    model.train()
    running_loss = 0.0
    running_psnr = 0.0
    running_ssim = 0.0
    total_samples = 0

    loop = tqdm(enumerate(dataloader), total=len(dataloader), ncols=100)

    for i, data in loop:
        lr_imgs = data[0].to(device)
        hr_imgs = data[1].to(device)
        current_batch_size = lr_imgs.size(0)
        total_samples += current_batch_size

        optimizer.zero_grad()
        sr_imgs = model(lr_imgs)
        loss = criterion(sr_imgs, hr_imgs)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * current_batch_size
        batch_psnr = psnr(hr_imgs, sr_imgs)
        running_psnr += batch_psnr * current_batch_size
        batch_ssim = compute_ssim(hr_imgs, sr_imgs)
        running_ssim += batch_ssim * current_batch_size

        loop.set_description(f"训练批次 {i+1}/{len(dataloader)}")
        loop.set_postfix(loss=loss.item(), psnr=batch_psnr, ssim=batch_ssim)

    epoch_loss = running_loss / total_samples if total_samples > 0 else 0
    epoch_psnr = running_psnr / total_samples if total_samples > 0 else 0
    epoch_ssim = running_ssim / total_samples if total_samples > 0 else 0

    avg_metrics = {
        'loss': epoch_loss,
        'psnr': epoch_psnr,
        'ssim': epoch_ssim
    }

    return avg_metrics

def validate_model(model, val_loader, criterion, device):
    """验证模型"""
    model.eval()
    running_loss = 0.0
    running_psnr = 0.0
    running_ssim = 0.0
    total_samples = 0

    with torch.no_grad():
        for i, data in enumerate(tqdm(val_loader, desc="验证中", ncols=100)):
            lr_imgs, hr_imgs = data
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            batch_size = lr_imgs.size(0)
            total_samples += batch_size

            sr_imgs = model(lr_imgs)
            batch_loss = criterion(sr_imgs, hr_imgs)
            running_loss += batch_loss.item() * batch_size

            batch_psnr = psnr(hr_imgs, sr_imgs)
            running_psnr += batch_psnr * batch_size

            batch_ssim = compute_ssim(hr_imgs, sr_imgs)
            running_ssim += batch_ssim * batch_size

    epoch_loss = running_loss / total_samples if total_samples > 0 else 0.0
    epoch_psnr = running_psnr / total_samples if total_samples > 0 else 0.0
    epoch_ssim = running_ssim / total_samples if total_samples > 0 else 0.0

    avg_metrics = {
        'loss': epoch_loss,
        'psnr': epoch_psnr,
        'ssim': epoch_ssim
    }

    return avg_metrics

def test_model(model, test_loader, criterion, device):
    """测试模型"""
    model.eval()
    running_loss = 0.0
    running_psnr = 0.0
    running_ssim = 0.0
    total_samples = 0

    print(f"测试模型，共 {len(test_loader.dataset)} 张图像...")
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader, desc="测试中", ncols=100)):
            lr_imgs, hr_imgs = data
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            batch_size = lr_imgs.size(0)
            total_samples += batch_size

            sr_imgs = model(lr_imgs)
            batch_loss = criterion(sr_imgs, hr_imgs)
            running_loss += batch_loss.item() * batch_size

            batch_psnr = psnr(hr_imgs, sr_imgs)
            running_psnr += batch_psnr * batch_size

            batch_ssim = compute_ssim(hr_imgs, sr_imgs)
            running_ssim += batch_ssim * batch_size

    epoch_loss = running_loss / total_samples if total_samples > 0 else 0.0
    epoch_psnr = running_psnr / total_samples if total_samples > 0 else 0.0
    epoch_ssim = running_ssim / total_samples if total_samples > 0 else 0.0

    avg_metrics = {
        'loss': epoch_loss,
        'psnr': epoch_psnr,
        'ssim': epoch_ssim
    }

    return avg_metrics

def plot_learning_curves(history_df, plots_dir):
    """绘制学习曲线"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss曲线
    axes[0, 0].plot(history_df['epoch'], history_df['train_loss'], label='Train Loss')
    axes[0, 0].plot(history_df['epoch'], history_df['val_loss'], label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # PSNR曲线
    axes[0, 1].plot(history_df['epoch'], history_df['train_psnr'], label='Train PSNR')
    axes[0, 1].plot(history_df['epoch'], history_df['val_psnr'], label='Val PSNR')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('PSNR (dB)')
    axes[0, 1].set_title('Training and Validation PSNR')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # SSIM曲线
    axes[1, 0].plot(history_df['epoch'], history_df['train_ssim'], label='Train SSIM')
    axes[1, 0].plot(history_df['epoch'], history_df['val_ssim'], label='Val SSIM')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('SSIM')
    axes[1, 0].set_title('Training and Validation SSIM')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # 学习率曲线
    axes[1, 1].plot(history_df['epoch'], history_df['learning_rate'], label='Learning Rate')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].set_yscale('log')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

def train_and_test():
    """训练并测试模型"""
    
    # 保存路径
    experiment_dir = os.path.join(output_dir, f"{experiment_name}_{learning_rate}")
    model_dir = os.path.join(experiment_dir, 'models')
    test_results_dir = os.path.join(experiment_dir, 'test_results')
    plots_dir = os.path.join(experiment_dir, 'plots')
    
    # 创建保存路径
    for dir_path in [experiment_dir, model_dir, test_results_dir, plots_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = create_train_val_test_dataloaders(
        lr_dir, hr_dir, batch_size, train_ratio, seed, num_workers, pin_memory
    )
    print(f"训练集样本数: {len(train_loader.dataset)}")
    print(f"验证集样本数: {len(val_loader.dataset)}")
    print(f"测试集样本数: {len(test_loader.dataset)}")
    
    num_channels = 7
    
    # 创建模型（模型内部已实现Dropout）
    model = UNetSA(up_scale=up_scale, img_channel=num_channels, width=width, 
                   use_attention=True, dropout_rate=dropout_rate).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"模型创建完成，共 {num_params} 个参数")
    print(f"模型内部Dropout率: {dropout_rate}")
    
    # 创建优化器（添加权重衰减）
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    
    # 创建学习率调度器
    if lr_scheduler_type == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=lr_factor, 
                                    patience=lr_patience, min_lr=lr_min, verbose=True)
    elif lr_scheduler_type == 'CosineAnnealing':
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr_min)
    
    # 训练历史记录
    history_data = []
    best_val_psnr = 0.0
    
    print(f"\n开始训练，共 {epochs} 个轮次...")
    print(f"使用学习率调度器: {lr_scheduler_type}")
    
    total_training_start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\n📘 轮次 {epoch+1}/{epochs} (学习率: {current_lr:.2e})")
        
        # 训练
        train_metrics = train_epoch(model, train_loader, criterion, optimizer)
        print(f"训练 - Loss: {train_metrics['loss']:.4f}, PSNR: {train_metrics['psnr']:.4f}, SSIM: {train_metrics['ssim']:.4f}")
        
        # 验证
        val_metrics = validate_model(model, val_loader, criterion, device)
        print(f"验证 - Loss: {val_metrics['loss']:.4f}, PSNR: {val_metrics['psnr']:.4f}, SSIM: {val_metrics['ssim']:.4f}")
        
        # 学习率调度
        if lr_scheduler_type == 'ReduceLROnPlateau':
            scheduler.step(val_metrics['psnr'])
        elif lr_scheduler_type == 'CosineAnnealing':
            scheduler.step()
        
        # 记录历史
        epoch_data = {
            'epoch': epoch + 1,
            'learning_rate': current_lr,
            'train_loss': train_metrics['loss'],
            'train_psnr': train_metrics['psnr'],
            'train_ssim': train_metrics['ssim'],
            'val_loss': val_metrics['loss'],
            'val_psnr': val_metrics['psnr'],
            'val_ssim': val_metrics['ssim'],
            'epoch_time': time.time() - epoch_start_time,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        history_data.append(epoch_data)
        
        # 保存历史数据为CSV
        history_df = pd.DataFrame(history_data)
        history_df.to_csv(os.path.join(experiment_dir, 'training_history.csv'), index=False)
        
        # 绘制学习曲线
        if len(history_data) > 1:
            plot_learning_curves(history_df, plots_dir)
        
        # 保存模型
        if (epoch + 1) % save_interval == 0:
            save_model(model, model_dir, f"epoch_{epoch+1}.pth", val_metrics)
            print(f"已保存轮次 {epoch+1} 的模型")
        
        # 保存最佳模型（基于验证集PSNR）
        if val_metrics['psnr'] > best_val_psnr:
            best_val_psnr = val_metrics['psnr']
            save_model(model, model_dir, "best_model.pth", val_metrics)
            print(f"已保存最佳模型 (验证PSNR: {best_val_psnr:.4f})")
        
        print(f"轮次 {epoch+1} 耗时: {epoch_data['epoch_time']:.2f} 秒")
    
    total_training_time = time.time() - total_training_start_time
    print(f"\n🎉 训练完成! 总耗时: {total_training_time // 3600:.0f}小时 {(total_training_time % 3600) // 60:.0f}分钟 {total_training_time % 60:.2f}秒")
    
    # 保存最终模型
    save_model(model, model_dir, "final_model.pth", val_metrics)
    print(f"已保存最终模型")
    
    # 保存训练配置
    config_data = {
        'experiment_name': experiment_name,
        'up_scale': up_scale,
        'width': width,
        'dropout_rate': dropout_rate,
        'batch_size': batch_size,
        'epochs': epoch + 1,  # 实际训练的轮次数
        'initial_learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'lr_scheduler_type': lr_scheduler_type,
        'best_val_psnr': best_val_psnr,
        'total_training_time': total_training_time,
        'num_parameters': num_params
    }
    config_df = pd.DataFrame([config_data])
    config_df.to_csv(os.path.join(experiment_dir, 'training_config.csv'), index=False)
    
    # 使用最佳模型进行最终测试
    print("\n📊 使用最佳模型在测试集上进行最终评估...")
    try:
        best_model_path = os.path.join(model_dir, "best_model.pth")
        if os.path.exists(best_model_path):
            load_model(model, best_model_path, device)
            print(f"已加载最佳模型: {best_model_path}")
        else:
            print(f"警告: best_model.pth 未找到，使用最终模型进行测试。")
    except Exception as e:
        print(f"加载最佳模型失败: {e}。使用最终模型进行测试。")
    
    # 最终测试
    final_test_metrics = test_model(model, test_loader, criterion, device)
    
    # 保存测试结果
    test_results = {
        'test_loss': final_test_metrics['loss'],
        'test_psnr': final_test_metrics['psnr'],
        'test_ssim': final_test_metrics['ssim'],
        'test_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    test_df = pd.DataFrame([test_results])
    test_df.to_csv(os.path.join(experiment_dir, 'test_results.csv'), index=False)
    
    # 创建并保存完整的训练总结
    summary_data = {
        '实验名称': experiment_name,
        '总轮次': epoch + 1,
        '最佳验证PSNR': best_val_psnr,
        '最终测试Loss': final_test_metrics['loss'],
        '最终测试PSNR': final_test_metrics['psnr'],
        '最终测试SSIM': final_test_metrics['ssim'],
        '总训练时间(小时)': total_training_time / 3600,
        '模型参数量': num_params,
        '初始学习率': learning_rate,
        '权重衰减': weight_decay,
        'Dropout率': dropout_rate,
        '批次大小': batch_size,
        '上采样倍数': up_scale,
        '模型宽度': width
    }
    summary_df = pd.DataFrame([summary_data])
    summary_df.to_csv(os.path.join(experiment_dir, 'training_summary.csv'), index=False)
    print(f"\n训练总结已保存到: {os.path.join(experiment_dir, 'training_summary.csv')}")
    
    return model, final_test_metrics

# ====================== 主函数 ======================
def main():
    """主函数"""
    
    print("🚀 GPU 可用:", torch.cuda.is_available())
    
    if torch.cuda.is_available():
        print("🧠 当前使用的 GPU 数量:", torch.cuda.device_count())
        print("📍 当前默认设备:", torch.cuda.current_device())
        print("💻 当前设备名称:", torch.cuda.get_device_name(torch.cuda.current_device()))
        
        device = torch.cuda.current_device()
        total_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        used_gb = torch.cuda.memory_allocated(device) / (1024**3)
        print(f"💾 GPU 内存: {used_gb:.2f}/{total_gb:.2f} GB ({used_gb/total_gb*100:.1f}%)")
    else:
        print("⚠️ 当前使用 CPU")
    
    print("\n📋 配置:")
    print(f"  实验名称: {experiment_name}")
    print(f"  数据目录: LR='{lr_dir}', HR='{hr_dir}'")
    print(f"  模型参数: up_scale={up_scale}, width={width}, dropout_rate={dropout_rate}")
    print(f"  训练参数: batch_size={batch_size}, epochs={epochs}, lr={learning_rate}, weight_decay={weight_decay}")
    print(f"  学习率调度: {lr_scheduler_type}")
    print(f"  输出目录: {os.path.join(output_dir, f'{experiment_name}_{learning_rate}')}")
    
    print("\n🚀 开始训练和测试...")
    _, final_metrics = train_and_test()
    
    print("\n✅ 训练和测试完成!")
    print(f"最终测试集结果: Loss={final_metrics['loss']:.4f}, PSNR={final_metrics['psnr']:.4f}, SSIM={final_metrics['ssim']:.4f}")

if __name__ == "__main__":
    main()