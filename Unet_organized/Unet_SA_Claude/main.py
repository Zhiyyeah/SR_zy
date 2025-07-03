import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import time
import json
import matplotlib.pyplot as plt

# 导入本地模块
from model_attention import UNetSA
from data_loader import create_train_test_dataloaders
from metrics import psnr, compute_ssim
from utils import visualize_results, save_plots, get_device

# ====================== 配置参数 ======================
class Config:
    """训练配置类"""
    # 实验设置
    experiment_name = 'zy_computer_1'
    
    # 数据路径
    lr_dir = 'SR_zy/Imagey/Imagery_WaterLand/WaterLand_TOA_tiles_lr'
    hr_dir = 'SR_zy/Imagey/Imagery_WaterLand/WaterLand_TOA_tiles_hr'
    train_ratio = 0.8
    
    # 模型参数
    up_scale = 8
    width = 32
    dropout_rate = 0.1
    num_channels = 7
    
    # 训练参数
    batch_size = 8
    epochs = 200
    learning_rate = 1e-4
    weight_decay = 1e-5
    
    # 系统设置
    num_workers = 4
    pin_memory = True
    seed = 42
    device = get_device()
    
    # 输出设置
    output_dir = './outputs'
    save_interval = 1
    
    # 可视化设置
    rgb_channels = [3, 2, 1]


# ====================== 训练函数 ======================
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    
    # 初始化指标
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    num_batches = len(dataloader)
    
    # 进度条
    progress_bar = tqdm(dataloader, desc="训练中", ncols=100)
    
    for batch_idx, (lr_imgs, hr_imgs) in enumerate(progress_bar):
        # 数据移到GPU
        lr_imgs = lr_imgs.to(device)
        hr_imgs = hr_imgs.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        sr_imgs = model(lr_imgs)
        loss = criterion(sr_imgs, hr_imgs)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 计算指标
        batch_psnr = psnr(hr_imgs, sr_imgs)
        batch_ssim = compute_ssim(hr_imgs, sr_imgs)
        
        # 累积指标
        total_loss += loss.item()
        total_psnr += batch_psnr
        total_ssim += batch_ssim
        
        # 更新进度条
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'psnr': f'{batch_psnr:.2f}',
            'ssim': f'{batch_ssim:.4f}'
        })
    
    # 计算平均指标
    avg_metrics = {
        'loss': total_loss / num_batches,
        'psnr': total_psnr / num_batches,
        'ssim': total_ssim / num_batches
    }
    
    return avg_metrics


def evaluate_model(model, dataloader, criterion, device):
    """评估模型"""
    model.eval()
    
    # 初始化指标
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    num_batches = len(dataloader)
    
    # 进度条
    progress_bar = tqdm(dataloader, desc="测试中", ncols=100)
    
    with torch.no_grad():
        for lr_imgs, hr_imgs in progress_bar:
            # 数据移到GPU
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            
            # 前向传播
            sr_imgs = model(lr_imgs)
            loss = criterion(sr_imgs, hr_imgs)
            
            # 计算指标
            batch_psnr = psnr(hr_imgs, sr_imgs)
            batch_ssim = compute_ssim(hr_imgs, sr_imgs)
            
            # 累积指标
            total_loss += loss.item()
            total_psnr += batch_psnr
            total_ssim += batch_ssim
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'psnr': f'{batch_psnr:.2f}',
                'ssim': f'{batch_ssim:.4f}'
            })
    
    # 计算平均指标
    avg_metrics = {
        'loss': total_loss / num_batches,
        'psnr': total_psnr / num_batches,
        'ssim': total_ssim / num_batches
    }
    
    return avg_metrics


def save_checkpoint(model, optimizer, epoch, metrics, save_path):
    """保存模型检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    torch.save(checkpoint, save_path)


def create_loss_function():
    """创建混合损失函数"""
    l1_criterion = nn.L1Loss()
    mse_criterion = nn.MSELoss()
    
    def mixed_loss(pred, target):
        return 0.7 * l1_criterion(pred, target) + 0.3 * mse_criterion(pred, target)
    
    return mixed_loss


def print_training_info(config):
    """打印训练信息"""
    print("="*60)
    print("超分辨率模型训练")
    print("="*60)
    
    print("\n📊 训练配置:")
    print(f"  • 实验名称: {config.experiment_name}")
    print(f"  • 上采样倍数: {config.up_scale}x")
    print(f"  • 模型宽度: {config.width}")
    print(f"  • 批次大小: {config.batch_size}")
    print(f"  • 训练轮数: {config.epochs}")
    print(f"  • 学习率: {config.learning_rate}")
    print(f"  • 权重衰减: {config.weight_decay}")
    
    print("\n💻 系统信息:")
    if torch.cuda.is_available():
        print(f"  • 使用设备: {torch.cuda.get_device_name(0)}")
        device_props = torch.cuda.get_device_properties(0)
        print(f"  • GPU内存: {device_props.total_memory / 1024**3:.1f} GB")
    else:
        print("  • 使用设备: CPU")


# ====================== 主训练函数 ======================
def train():
    """主训练函数"""
    # 加载配置
    config = Config()
    
    # 设置随机种子
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
    
    # 打印训练信息
    print_training_info(config)
    
    # 创建输出目录
    experiment_dir = os.path.join(config.output_dir, config.experiment_name)
    model_dir = os.path.join(experiment_dir, 'models')
    plots_dir = os.path.join(experiment_dir, 'plots')
    
    for dir_path in [experiment_dir, model_dir, plots_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # 保存配置
    config_dict = {k: v for k, v in vars(config).items() if not k.startswith('_')}
    with open(os.path.join(experiment_dir, 'config.json'), 'w') as f:
        json.dump(config_dict, f, indent=4, default=str)
    
    # 创建数据加载器
    print("\n📁 加载数据...")
    train_loader, test_loader = create_train_test_dataloaders(
        config.lr_dir, 
        config.hr_dir, 
        config.batch_size, 
        config.train_ratio, 
        config.seed, 
        config.num_workers, 
        config.pin_memory
    )
    print(f"  • 训练集: {len(train_loader.dataset)} 张图像")
    print(f"  • 测试集: {len(test_loader.dataset)} 张图像")
    
    # 创建模型
    print("\n🔧 创建模型...")
    model = UNetSA(
        up_scale=config.up_scale,
        img_channel=config.num_channels,
        width=config.width,
        dropout_rate=config.dropout_rate,
        use_attention=True
    ).to(config.device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  • 模型参数量: {num_params:,}")
    
    # 创建优化器和损失函数
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config.learning_rate, 
        weight_decay=config.weight_decay
    )
    criterion = create_loss_function()
    
    # 训练历史记录
    train_history = {'loss': [], 'psnr': [], 'ssim': []}
    test_history = {'loss': [], 'psnr': [], 'ssim': []}
    best_test_psnr = 0.0
    best_epoch = 0
    
    # 开始训练
    print("\n🚀 开始训练...")
    start_time = time.time()
    
    for epoch in range(1, config.epochs + 1):
        print(f"\n轮次 {epoch}/{config.epochs}")
        
        # 训练一个epoch
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, config.device)
        print(f"训练 - Loss: {train_metrics['loss']:.4f}, "
              f"PSNR: {train_metrics['psnr']:.2f} dB, "
              f"SSIM: {train_metrics['ssim']:.4f}")
        
        # 评估模型
        test_metrics = evaluate_model(model, test_loader, criterion, config.device)
        print(f"测试 - Loss: {test_metrics['loss']:.4f}, "
              f"PSNR: {test_metrics['psnr']:.2f} dB, "
              f"SSIM: {test_metrics['ssim']:.4f}")
        
        # 记录历史
        for key in ['loss', 'psnr', 'ssim']:
            train_history[key].append(train_metrics[key])
            test_history[key].append(test_metrics[key])
        
        # 保存最佳模型
        if test_metrics['psnr'] > best_test_psnr:
            best_test_psnr = test_metrics['psnr']
            best_epoch = epoch
            save_checkpoint(
                model, optimizer, epoch, test_metrics,
                os.path.join(model_dir, 'best_model.pth')
            )
            print(f"✨ 保存最佳模型 (PSNR: {best_test_psnr:.2f} dB)")
        
        # 定期保存检查点
        if epoch % config.save_interval == 0:
            save_checkpoint(
                model, optimizer, epoch, test_metrics,
                os.path.join(model_dir, f'checkpoint_epoch_{epoch}.pth')
            )
            
            # 保存训练曲线
            save_plots(train_history, test_history, plots_dir)
    
    # 训练完成
    end_time = time.time()
    training_time = end_time - start_time
    hours = int(training_time // 3600)
    minutes = int((training_time % 3600) // 60)
    seconds = training_time % 60
    
    print("\n✅ 训练完成!")
    print(f"  • 总耗时: {hours}小时 {minutes}分钟 {seconds:.0f}秒")
    print(f"  • 最佳模型: 第 {best_epoch} 轮 (PSNR: {best_test_psnr:.2f} dB)")
    
    # 保存最终模型
    save_checkpoint(
        model, optimizer, config.epochs, test_metrics,
        os.path.join(model_dir, 'final_model.pth')
    )
    
    # 保存训练总结
    summary = {
        'total_epochs': config.epochs,
        'best_epoch': best_epoch,
        'best_test_psnr': best_test_psnr,
        'final_test_metrics': test_metrics,
        'training_time_seconds': training_time,
        'model_parameters': num_params
    }
    
    with open(os.path.join(experiment_dir, 'training_summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)
    
    return model, test_metrics


# ====================== 程序入口 ======================
if __name__ == "__main__":
    model, final_metrics = train()