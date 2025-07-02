import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from tqdm import tqdm
import numpy as np
import time
import matplotlib.pyplot as plt
from collections import deque

# 导入本地模块
from model_attention import UNetSA  # 使用改进后的模型
from data_loader import create_train_test_dataloaders
from metrics import psnr, compute_ssim
from utils import visualize_results, save_plots, save_metrics_to_file, get_device
from model_io import save_model, load_model

# ====================== 配置参数 ======================
experiment_name = 'improved_training_no_overfit_2'  # 更新实验名
# 数据设置
lr_dir = '/public/home/zyye/SR_backup/Imagery/Water_TOA_tiles_lr'  # 低分辨率图像路径
hr_dir = '/public/home/zyye/SR_backup/Imagery/Water_TOA_tiles'  # 高分辨率图像路径
train_ratio = 0.8

# 模型设置
up_scale = 8
width = 32  # 降低模型容量
dropout_rate = 0.1  # 添加dropout率

# 训练设置
batch_size = 32  # 适当减小batch size可以增加噪声，有助于正则化
num_workers = 4
pin_memory = True
seed = 42
epochs = 200  # 增加epochs，因为会使用早停
learning_rate = 3e-4  # 适中的初始学习率
weight_decay = 1e-5  # 更小的权重衰减

# 早停设置
early_stopping_patience = 50  # 大幅增加patience，给模型充分的训练时间
min_delta = 0  # 设为0，只要有改善就继续
use_early_stopping = True  # 可以选择是否使用早停

# 学习率调度器设置
lr_scheduler_type = 'cosine_warmup'  # 使用带预热的余弦退火
lr_patience = 20  # 大幅增加patience
lr_factor = 0.8  # 更温和的衰减因子（从0.5改为0.8）
lr_min = 5e-5  # 提高最小学习率（从1e-6改为5e-5）
warmup_epochs = 5  # 添加预热期

# 梯度裁剪
gradient_clip_norm = 1.0  # 梯度裁剪的最大范数

# 设备设置
device = get_device()

# 输出设置
output_dir = './outputs'
save_interval = 5  # 每5个epoch保存一次

# 可视化设置
rgb_channels = [3, 2, 1]


# ====================== 早停类 ======================
class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=10, min_delta=0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.mode == 'min':
            score = -score
            
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            
        return self.early_stop


# ====================== 数据增强（如果data_loader中没有） ======================
def get_augmentation_transforms():
    """获取数据增强变换"""
    import torchvision.transforms as transforms
    
    # 注意：这里假设你的数据是多通道遥感图像
    # 如果data_loader已经包含增强，可以跳过这部分
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        # 对于多通道图像，可能需要自定义旋转和颜色增强
    ])
    
    test_transform = None  # 测试时不使用数据增强
    
    return train_transform, test_transform


# ====================== 辅助函数 ======================
def train_epoch(model, dataloader, criterion, optimizer, gradient_clip_norm=None):
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
        
        # 梯度裁剪
        if gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        
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

            # 前向推理
            sr_imgs = model(lr_imgs)

            # 损失计算
            batch_loss = criterion(sr_imgs, hr_imgs)
            running_loss += batch_loss.item() * batch_size

            # 指标计算
            batch_psnr = psnr(hr_imgs, sr_imgs)
            running_psnr += batch_psnr * batch_size

            batch_ssim = compute_ssim(hr_imgs, sr_imgs)
            running_ssim += batch_ssim * batch_size

    # 计算平均指标
    epoch_loss = running_loss / total_samples if total_samples > 0 else 0.0
    epoch_psnr = running_psnr / total_samples if total_samples > 0 else 0.0
    epoch_ssim = running_ssim / total_samples if total_samples > 0 else 0.0

    avg_metrics = {
        'loss': epoch_loss,
        'psnr': epoch_psnr,
        'ssim': epoch_ssim
    }

    return avg_metrics


def train_and_test():
    """训练并测试模型"""

    # 保存路径
    experiment_dir = os.path.join(output_dir, f"{experiment_name}_lr{learning_rate}_wd{weight_decay}")
    model_dir = os.path.join(experiment_dir, 'models')
    test_results_dir = os.path.join(experiment_dir, 'test_results')
    plots_dir = os.path.join(experiment_dir, 'plots')
    
    # 创建保存路径
    for dir_path in [experiment_dir, model_dir, test_results_dir, plots_dir]:
        os.makedirs(dir_path, exist_ok=True)

    # 保存配置
    config = {
        'experiment_name': experiment_name,
        'up_scale': up_scale,
        'width': width,
        'dropout_rate': dropout_rate,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'epochs': epochs,
        'early_stopping_patience': early_stopping_patience,
        'gradient_clip_norm': gradient_clip_norm,
        'lr_scheduler_type': lr_scheduler_type
    }
    
    import json
    with open(os.path.join(experiment_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    # 创建dataloader
    train_loader, test_loader = create_train_test_dataloaders(
        lr_dir, hr_dir, batch_size, train_ratio, seed, num_workers, pin_memory
    )
    print(f"训练集样本数: {len(train_loader.dataset)}")
    print(f"测试集样本数: {len(test_loader.dataset)}")

    num_channels = 7

    # 创建模型（使用改进的模型）
    model = UNetSA(
        up_scale=up_scale, 
        img_channel=num_channels, 
        width=width,
        dropout_rate=dropout_rate,
        use_attention=True
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"模型创建完成，共 {num_params:,} 个参数")

    # 创建优化器（使用AdamW）
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # 损失函数 - 使用混合损失
    l1_criterion = nn.L1Loss()
    mse_criterion = nn.MSELoss()
    
    # 定义混合损失函数
    def mixed_loss(pred, target):
        return 0.7 * l1_criterion(pred, target) + 0.3 * mse_criterion(pred, target)
    
    criterion = mixed_loss
    
    # 学习率调度器 - 添加更多选项
    if lr_scheduler_type == 'reduce_on_plateau':
        scheduler = ReduceLROnPlateau(
            optimizer, 
            mode='max',  # 改为max，因为我们要最大化PSNR
            factor=lr_factor, 
            patience=lr_patience,
            min_lr=lr_min,
            verbose=True,
            threshold=0.01  # 只有改善超过0.01才算有效改善
        )
    elif lr_scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=lr_min
        )
    elif lr_scheduler_type == 'cosine_warmup':
        # 自定义带预热的余弦退火
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return epoch / warmup_epochs
            else:
                progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
                return lr_min / learning_rate + (1 - lr_min / learning_rate) * 0.5 * (1 + np.cos(np.pi * progress))
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:  # 'none' - 不使用学习率调度
        scheduler = None
    
    # 早停机制
    early_stopping = EarlyStopping(patience=early_stopping_patience, min_delta=min_delta, mode='min')

    # 训练历史
    train_history = {'loss': [], 'psnr': [], 'ssim': []}
    test_history = {'loss': [], 'psnr': [], 'ssim': []}
    best_test_psnr = 0.0
    best_test_loss = float('inf')

    print(f"开始训练，共 {epochs} 个轮次...")
    print(f"使用早停机制，patience={early_stopping_patience}")
    print(f"使用学习率调度器: {lr_scheduler_type}")
    print(f"使用梯度裁剪: max_norm={gradient_clip_norm}")
    
    total_training_start_time = time.time()

    for epoch in range(epochs):
        epoch_start_time = time.time()
        print(f"\n📘 轮次 {epoch+1}/{epochs}")
        print(f"当前学习率: {optimizer.param_groups[0]['lr']:.2e}")

        # 训练
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, gradient_clip_norm)
        for k, v in train_metrics.items():
            train_history[k].append(v)
        print(f"训练 - Loss: {train_metrics['loss']:.4f}, PSNR: {train_metrics['psnr']:.2f} dB, SSIM: {train_metrics['ssim']:.4f}")

        # 测试
        test_metrics = test_model(model, test_loader, criterion, device)
        for k, v in test_metrics.items():
            test_history[k].append(v)
        print(f"测试 - Loss: {test_metrics['loss']:.4f}, PSNR: {test_metrics['psnr']:.2f} dB, SSIM: {test_metrics['ssim']:.4f}")

        # 学习率调度
        if scheduler is not None:
            if lr_scheduler_type == 'reduce_on_plateau':
                scheduler.step(test_metrics['psnr'])  # 基于PSNR调整
            else:
                scheduler.step()

        # 早停检查 - 基于PSNR而不是损失
        if use_early_stopping and early_stopping(-test_metrics['psnr']):  # 负号因为我们要最大化PSNR
            print(f"\n早停触发！测试PSNR在 {early_stopping_patience} 个epochs内没有改善。")
            break

        # 保存图表
        save_plots(train_history, test_history, plots_dir)

        # 定期保存模型
        if (epoch + 1) % save_interval == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_metrics': train_metrics,
                'test_metrics': test_metrics
            }
            torch.save(checkpoint, os.path.join(model_dir, f"checkpoint_epoch_{epoch+1}.pth"))
            print(f"已保存检查点 (epoch {epoch+1})")

        # 保存最佳模型
        if test_metrics['loss'] < best_test_loss:
            best_test_loss = test_metrics['loss']
            best_test_psnr = test_metrics['psnr']
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_metrics': train_metrics,
                'test_metrics': test_metrics
            }
            torch.save(checkpoint, os.path.join(model_dir, "best_model.pth"))
            print(f"✨ 保存最佳模型 - Loss: {best_test_loss:.4f}, PSNR: {best_test_psnr:.2f} dB")

        epoch_time = time.time() - epoch_start_time
        print(f"轮次耗时: {epoch_time:.2f} 秒")

    total_training_time = time.time() - total_training_start_time
    hours = int(total_training_time // 3600)
    minutes = int((total_training_time % 3600) // 60)
    seconds = total_training_time % 60
    print(f"\n🎉 训练完成! 总耗时: {hours}小时 {minutes}分钟 {seconds:.2f}秒")

    # 保存最终模型
    final_checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'train_history': train_history,
        'test_history': test_history
    }
    torch.save(final_checkpoint, os.path.join(model_dir, "final_model.pth"))

    # 加载最佳模型进行最终测试
    print("\n📊 加载最佳模型进行最终测试...")
    best_checkpoint = torch.load(os.path.join(model_dir, "best_model.pth"))
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    final_test_metrics = test_model(model, test_loader, criterion, device)

    # 保存训练总结
    summary = {
        'total_epochs': epoch + 1,
        'best_epoch': best_checkpoint['epoch'],
        'best_test_metrics': best_checkpoint['test_metrics'],
        'final_test_metrics': final_test_metrics,
        'total_training_time': total_training_time,
        'model_parameters': num_params
    }
    
    with open(os.path.join(experiment_dir, 'training_summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)

    return model, final_test_metrics


# ====================== 主函数 ======================+
def main():
    """主函数"""
    print("="*60)
    print("超分辨率模型训练 - 改进版")
    print("="*60)
    
    print("\n🚀 GPU 信息:")
    if torch.cuda.is_available():
        print(f"  ✅ GPU 可用")
        print(f"  📍 设备数量: {torch.cuda.device_count()}")
        print(f"  💻 当前设备: {torch.cuda.get_device_name(0)}")
        
        device = torch.cuda.current_device()
        total_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        used_gb = torch.cuda.memory_allocated(device) / (1024**3)
        print(f"  💾 GPU 内存: {used_gb:.2f}/{total_gb:.2f} GB")
    else:
        print("  ⚠️ GPU 不可用，使用 CPU")
    
    print("\n📋 训练配置:")
    print(f"  实验名称: {experiment_name}")
    print(f"  模型宽度: {width} (通道数)")
    print(f"  Dropout率: {dropout_rate}")
    print(f"  批次大小: {batch_size}")
    print(f"  学习率: {learning_rate}")
    print(f"  权重衰减: {weight_decay}")
    print(f"  梯度裁剪: {gradient_clip_norm}")
    print(f"  早停patience: {early_stopping_patience}")
    print(f"  LR调度器: {lr_scheduler_type}")
    
    print("\n🚀 开始训练...")
    _, final_metrics = train_and_test()

    print("\n✅ 训练完成!")
    print(f"最终测试结果:")
    print(f"  Loss: {final_metrics['loss']:.4f}")
    print(f"  PSNR: {final_metrics['psnr']:.2f} dB")
    print(f"  SSIM: {final_metrics['ssim']:.4f}")


if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    main()