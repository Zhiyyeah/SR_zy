import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from tqdm import tqdm
import numpy as np
import time
import matplotlib.pyplot as plt

# 导入本地模块
from model_attention_improved import UNetSAImproved  # 使用改进的模型
from data_loader import create_train_test_dataloaders
from metrics import psnr, compute_ssim
from utils import visualize_results, save_plots, save_metrics_to_file, get_device
from model_io import save_model, load_model

# ====================== 配置参数 ======================
experiment_name = 'improved_model_v2'

# 数据设置
lr_dir = r'D:\Py_Code\Unet_SR\SR_zy\Imagey\Small_Dataset\lr'
hr_dir = r'D:\Py_Code\Unet_SR\SR_zy\Imagey\Small_Dataset\hr'
train_ratio = 0.8

# 模型设置
up_scale = 8
width = 64
dropout_rate = 0.15  # Dropout率
use_deep_supervision = True  # 是否使用深度监督

# 训练设置
batch_size = 16
num_workers = 4
pin_memory = True
seed = 42
epochs = 100
learning_rate = 0.001  # 初始学习率（会使用学习率调度器）
weight_decay = 1e-4  # L2正则化
gradient_clip = 1.0  # 梯度裁剪

# 学习率调度器设置
scheduler_type = 'cosine'  # 'plateau' or 'cosine'
warmup_epochs = 5  # 预热轮次

# 设备设置
device = get_device()

# 输出设置
output_dir = './outputs'
save_interval = 5  # 每5轮保存一次
patience = 15  # 早停耐心值

# 可视化设置
rgb_channels = [3, 2, 1]

# ====================== 辅助函数 ======================

class CombinedLoss(nn.Module):
    """组合损失函数：MSE + Perceptual-like loss"""
    def __init__(self, alpha=0.9, beta=0.1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
    
    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        l1_loss = self.l1(pred, target)
        return self.alpha * mse_loss + self.beta * l1_loss

def get_lr(optimizer):
    """获取当前学习率"""
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train_epoch(model, dataloader, criterion, optimizer, epoch, warmup_epochs):
    """训练一个轮次"""
    model.train()
    running_loss = 0.0
    running_psnr = 0.0
    running_ssim = 0.0
    total_samples = 0

    loop = tqdm(enumerate(dataloader), total=len(dataloader), ncols=100)
    
    # 学习率预热
    if epoch < warmup_epochs:
        warmup_factor = (epoch + 1) / warmup_epochs
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate * warmup_factor

    for i, data in loop:
        lr_imgs = data[0].to(device)
        hr_imgs = data[1].to(device)
        current_batch_size = lr_imgs.size(0)
        total_samples += current_batch_size

        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(lr_imgs)
        
        # 处理深度监督
        if use_deep_supervision and isinstance(outputs, tuple):
            sr_imgs, aux_output = outputs
            # 主损失 + 辅助损失
            main_loss = criterion(sr_imgs, hr_imgs)
            aux_loss = criterion(aux_output, hr_imgs)
            loss = 0.8 * main_loss + 0.2 * aux_loss
        else:
            sr_imgs = outputs
            loss = criterion(sr_imgs, hr_imgs)
        
        loss.backward()
        
        # 梯度裁剪
        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        
        optimizer.step()

        running_loss += loss.item() * current_batch_size
        batch_psnr = psnr(hr_imgs, sr_imgs)
        running_psnr += batch_psnr * current_batch_size
        batch_ssim = compute_ssim(hr_imgs, sr_imgs)
        running_ssim += batch_ssim * current_batch_size

        loop.set_description(f"训练轮次 {epoch+1}")
        loop.set_postfix(loss=loss.item(), psnr=batch_psnr, ssim=batch_ssim, lr=get_lr(optimizer))

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
            
            # 测试时不使用深度监督
            if isinstance(sr_imgs, tuple):
                sr_imgs = sr_imgs[0]

            # 计算损失和指标
            batch_loss = criterion(sr_imgs, hr_imgs)
            running_loss += batch_loss.item() * batch_size

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
    config_path = os.path.join(experiment_dir, 'config.txt')
    with open(config_path, 'w') as f:
        f.write(f"实验名称: {experiment_name}\n")
        f.write(f"学习率: {learning_rate}\n")
        f.write(f"权重衰减: {weight_decay}\n")
        f.write(f"Dropout率: {dropout_rate}\n")
        f.write(f"批次大小: {batch_size}\n")
        f.write(f"轮次数: {epochs}\n")
        f.write(f"梯度裁剪: {gradient_clip}\n")
        f.write(f"调度器类型: {scheduler_type}\n")
        f.write(f"深度监督: {use_deep_supervision}\n")

    # 创建数据加载器
    train_loader, test_loader = create_train_test_dataloaders(
        lr_dir, hr_dir, batch_size, train_ratio, seed, num_workers, pin_memory
    )
    print(f"训练集样本数: {len(train_loader.dataset)}")
    print(f"测试集样本数: {len(test_loader.dataset)}")

    num_channels = 7

    # 创建改进的模型
    model = UNetSAImproved(
        up_scale=up_scale, 
        img_channel=num_channels, 
        width=width,
        dropout_rate=dropout_rate,
        deep_supervision=use_deep_supervision
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"模型创建完成，共 {num_params:,} 个参数")

    # 创建优化器和损失函数
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.999))
    criterion = CombinedLoss(alpha=0.9, beta=0.1)  # 组合损失
    
    # 创建学习率调度器
    if scheduler_type == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True, min_lr=1e-6)
    elif scheduler_type == 'cosine':
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    
    # 训练前准备
    train_history = {'loss': [], 'psnr': [], 'ssim': [], 'lr': []}
    test_history = {'loss': [], 'psnr': [], 'ssim': []}
    best_test_psnr = 0.0
    best_test_ssim = 0.0
    no_improve_count = 0

    print(f"\n开始训练，共 {epochs} 个轮次...")
    total_training_start_time = time.time()

    for epoch in range(epochs):
        epoch_start_time = time.time()
        print(f"\n📘 轮次 {epoch+1}/{epochs}")
        
        # 训练
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, epoch, warmup_epochs)
        for k, v in train_metrics.items():
            train_history[k].append(v)
        train_history['lr'].append(get_lr(optimizer))
        
        print(f"训练 - Loss: {train_metrics['loss']:.4f}, PSNR: {train_metrics['psnr']:.2f}, SSIM: {train_metrics['ssim']:.4f}")

        # 测试
        test_metrics = test_model(model, test_loader, criterion, device)
        for k, v in test_metrics.items():
            test_history[k].append(v)
        
        print(f"测试 - Loss: {test_metrics['loss']:.4f}, PSNR: {test_metrics['psnr']:.2f}, SSIM: {test_metrics['ssim']:.4f}")

        # 学习率调度
        if scheduler_type == 'plateau':
            scheduler.step(test_metrics['psnr'])
        elif scheduler_type == 'cosine':
            scheduler.step()

        # 保存训练曲线
        save_plots(train_history, test_history, plots_dir)

        # 定期保存模型
        if (epoch + 1) % save_interval == 0:
            save_model(model, model_dir, f"epoch_{epoch+1}.pth", train_metrics)
            print(f"已保存轮次 {epoch+1} 的模型")

        # 保存最佳模型
        if test_metrics['psnr'] > best_test_psnr:
            best_test_psnr = test_metrics['psnr']
            save_model(model, model_dir, "best_psnr_model.pth", test_metrics)
            print(f"✨ 新的最佳PSNR: {best_test_psnr:.4f}")
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        if test_metrics['ssim'] > best_test_ssim:
            best_test_ssim = test_metrics['ssim']
            save_model(model, model_dir, "best_ssim_model.pth", test_metrics)
            print(f"✨ 新的最佳SSIM: {best_test_ssim:.4f}")

        # 早停检查
        if no_improve_count >= patience:
            print(f"\n⚠️ 早停触发：{patience}轮未改善")
            break

        epoch_time = time.time() - epoch_start_time
        print(f"轮次耗时: {epoch_time:.2f}秒")

    total_time = time.time() - total_training_start_time
    print(f"\n🎉 训练完成! 总耗时: {total_time//3600:.0f}h {(total_time%3600)//60:.0f}m {total_time%60:.0f}s")

    # 保存最终模型
    save_model(model, model_dir, "final_model.pth", train_metrics)
    
    # 加载最佳模型进行最终测试
    print("\n📊 加载最佳模型进行最终测试...")
    best_model_path = os.path.join(model_dir, "best_psnr_model.pth")
    if os.path.exists(best_model_path):
        load_model(model, best_model_path, device)
        print(f"已加载最佳PSNR模型")
    
    final_test_metrics = test_model(model, test_loader, criterion, device)
    
    # 保存最终结果
    results_path = os.path.join(experiment_dir, 'final_results.txt')
    with open(results_path, 'w') as f:
        f.write(f"最终测试结果:\n")
        f.write(f"Loss: {final_test_metrics['loss']:.6f}\n")
        f.write(f"PSNR: {final_test_metrics['psnr']:.4f}\n")
        f.write(f"SSIM: {final_test_metrics['ssim']:.4f}\n")
        f.write(f"最佳PSNR: {best_test_psnr:.4f}\n")
        f.write(f"最佳SSIM: {best_test_ssim:.4f}\n")
        f.write(f"总训练时间: {total_time//3600:.0f}h {(total_time%3600)//60:.0f}m\n")

    return model, final_test_metrics

# ====================== 主函数 ======================
def main():
    """主函数"""
    print("🚀 GPU 可用:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("🧠 当前使用的 GPU 数量:", torch.cuda.device_count())
        print("📍 当前默认设备:", torch.cuda.current_device())
        print("💻 当前设备名称:", torch.cuda.get_device_name(torch.cuda.current_device()))
        
        # 简洁版内存信息
        device = torch.cuda.current_device()
        total_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        used_gb = torch.cuda.memory_allocated(device) / (1024**3)
        print(f"💾 GPU 内存: {used_gb:.2f}/{total_gb:.2f} GB ({used_gb/total_gb*100:.1f}%)")
    else:
        print("⚠️ 当前使用 CPU")
    
    print("\n📋 配置:")
    print(f"  实验名称: {experiment_name}")
    print(f"  数据目录: LR='{lr_dir}', HR='{hr_dir}'")
    print(f"  模型参数: width={width}, up_scale={up_scale}, dropout={dropout_rate}")
    print(f"  训练参数: batch_size={batch_size}, epochs={epochs}")
    print(f"  优化器: AdamW(lr={learning_rate}, weight_decay={weight_decay})")
    print(f"  调度器: {scheduler_type}")
    print(f"  深度监督: {use_deep_supervision}")
    print(f"  梯度裁剪: {gradient_clip}")
    print(f"  输出目录: {os.path.join(output_dir, f'{experiment_name}_lr{learning_rate}_wd{weight_decay}')}")

    print("\n🚀 开始训练和测试...")
    _, final_metrics = train_and_test()

    print("\n✅ 训练和测试完成!")
    print(f"最终测试集结果: Loss={final_metrics['loss']:.4f}, PSNR={final_metrics['psnr']:.4f}, SSIM={final_metrics['ssim']:.4f}")


if __name__ == "__main__":
    main()