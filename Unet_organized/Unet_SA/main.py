import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import time
import matplotlib.pyplot as plt

# 导入本地模块
from model_attention import UNetSA # 假设这些模块存在且路径正确
from data_loader import create_train_test_dataloaders
from metrics import psnr, compute_ssim
from utils import visualize_results, save_plots, save_metrics_to_file, get_device # bilinear_interpolation 不再直接从这里导入到test_model
from model_io import save_model, load_model

# ====================== 配置参数 ======================
# (配置参数部分保持不变，此处省略以保持简洁)
experiment_name = 'best_of_500_bs_256' # 可以更新实验名
# 数据设置
lr_dir = '/public/home/zyye/SR_backup/Imagery/Water_TOA_tiles_lr'  # 低分辨率图像路径
hr_dir = '/public/home/zyye/SR_backup/Imagery/Water_TOA_tiles'  # 高分辨率图像路径
train_ratio = 0.8

# 模型设置
up_scale = 8
width = 64

# 训练设置
batch_size = 256
num_workers = 4
pin_memory = True
seed = 42
epochs = 100
learning_rate = 0.00043
weight_decay = 0

# 设备设置
device = get_device()

# 输出设置
output_dir = './outputs'
save_interval = 1


# 可视化设置
rgb_channels = [3, 2, 1]


# ====================== 辅助函数 ======================

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
        loss = criterion(sr_imgs, hr_imgs)#torch.sqrt() # RMSE 损失
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * current_batch_size
        batch_psnr = psnr(hr_imgs, sr_imgs) # 重命名避免与函数名冲突
        running_psnr += batch_psnr * current_batch_size
        batch_ssim = compute_ssim(hr_imgs, sr_imgs) # 重命名
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
    """
    测试模型，计算整体的 RMSE 损失、PSNR 和 SSIM，返回格式与 train_epoch 一致。
    可选地将可视化结果保存到 results_dir。
    """
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

            # RMSE 损失（与 train_epoch 保持一致）
            batch_loss = criterion(sr_imgs, hr_imgs)#torch.sqrt()
            running_loss += batch_loss.item() * batch_size

            # 批次 PSNR 和 SSIM （对整批取平均，再加权累加）
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

    #保存路径
    experiment_dir = os.path.join(output_dir, f"{experiment_name}_{learning_rate}")
    model_dir = os.path.join(experiment_dir, 'models')
    # 所有测试结果将保存到此目录
    test_results_dir = os.path.join(experiment_dir, 'test_results')
    plots_dir = os.path.join(experiment_dir, 'plots')
    # 创建保存路径
    for dir_path in [experiment_dir, model_dir, test_results_dir, plots_dir]:
        os.makedirs(dir_path, exist_ok=True)

    # 创建dataloader
    train_loader, test_loader = create_train_test_dataloaders(
        lr_dir, hr_dir, batch_size, train_ratio, seed, num_workers, pin_memory
    )
    print(f"训练集样本数: {len(train_loader.dataset)}")
    print(f"测试集样本数: {len(test_loader.dataset)}")

    num_channels = 7

    # 创建模型
    model = UNetSA(up_scale=up_scale, img_channel=num_channels, width=width).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"模型创建完成，共 {num_params} 个参数")

    # 创建优化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss() # MSE损失函数，train_epoch 和 test_model 内部会取sqrt得到RMSE

    #训练前准备
    train_history = {'loss': [], 'psnr': [], 'ssim': []}
    test_history = {'loss': [], 'psnr': [], 'ssim': []}
    best_test_psnr = 0.0

    print(f"开始训练，共 {epochs} 个轮次...")
    total_training_start_time = time.time()

    for epoch in range(epochs):
        epoch_start_time = time.time()
        print(f"\n📘 轮次 {epoch+1}/{epochs}")

        # 训练、数据保存
        train_metrics = train_epoch(model, train_loader, criterion, optimizer)
        for k, v in train_metrics.items():
            train_history[k].append(v)
        print(f"训练损失: {train_metrics['loss']:.4f}, 训练PSNR: {train_metrics['psnr']:.4f}, 训练SSIM: {train_metrics['ssim']:.4f}")

        # 测试、数据保存
        test_metrics = test_model(model, test_loader, criterion, device)
        for k, v in test_metrics.items():
            test_history[k].append(v)
        print(f"测试损失: {test_metrics['loss']:.4f}, 测试PSNR: {test_metrics['psnr']:.4f}, 测试SSIM: {test_metrics['ssim']:.4f}")

        # 画曲线图
        save_plots(train_history, test_history, plots_dir)

        # 保存模型
        if (epoch + 1) % save_interval == 0:
            save_model(model, model_dir, f"epoch_{epoch+1}.pth", train_metrics)
            print(f"已保存轮次 {epoch+1} 的模型")

        # 保存最佳模型
        if test_metrics['psnr'] > best_test_psnr:
            best_test_psnr = test_metrics['psnr']
            save_model(model, model_dir, "best_model.pth", test_metrics)
            print(f"已保存最佳模型 (基于测试PSNR): {best_test_psnr:.4f}")

        epoch_time_taken = time.time() - epoch_start_time
        print(f"轮次 {epoch+1} 耗时: {epoch_time_taken:.2f} 秒")

    total_training_time = time.time() - total_training_start_time
    print(f"\n🎉 训练完成! 总耗时: {total_training_time // 3600:.0f}小时 {(total_training_time % 3600) // 60:.0f}分钟 {total_training_time % 60:.2f}秒")

    save_model(model, model_dir, "final_model.pth", train_metrics)
    print(f"已保存最终模型")

    print("\n📊 使用测试集评估最终模型...")
    print("加载最佳模型 (基于测试PSNR) 进行测试...")
    try:
        best_model_path = os.path.join(model_dir, "best_model.pth")
        if os.path.exists(best_model_path):
            # 假设 load_model 修改传入的 model 对象的状态
            load_model(model, best_model_path, device)
            print(f"已加载最佳模型: {best_model_path}")
        else:
            print(f"警告: best_model.pth 未找到 ({best_model_path})，将使用当前轮次的最终模型进行测试。")
    except Exception as e:
        print(f"加载最佳模型失败: {e}。将使用当前轮次的最终模型进行测试。")

    # 将 criterion 和 test_results_dir 传递给 test_model
    final_test_metrics = test_model(model, test_loader, criterion, device)

    return model, final_test_metrics


# ====================== 主函数 ======================
def main():
    """主函数"""

    print("🚀 GPU 可用:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("🧠 当前使用的 GPU 数量:", torch.cuda.device_count())
        print("📍 当前默认设备:", torch.cuda.current_device())
        print("💻 当前设备名称:", torch.cuda.get_device_name(torch.cuda.current_device()))
        
        # ✅ 简洁版内存信息
        device = torch.cuda.current_device()
        total_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        used_gb = torch.cuda.memory_allocated(device) / (1024**3)
        print(f"💾 GPU 内存: {used_gb:.2f}/{total_gb:.2f} GB ({used_gb/total_gb*100:.1f}%)")
    else:
        print("⚠️ 当前使用 CPU")
    print("\n📋 配置:")
    print(f"  数据目录: LR='{lr_dir}', HR='{hr_dir}'")
    # ... 其他配置打印 ...
    print(f"  输出目录: {os.path.join(output_dir, f'{experiment_name}_{learning_rate}')}")
    print(f"  测试结果将保存在: {os.path.join(output_dir, f'{experiment_name}_{learning_rate}', 'test_results')}")


    print("\n🚀 开始训练和测试...")
    _, final_metrics = train_and_test() # final_metrics 现在包含 'loss', 'psnr', 'ssim'

    print("\n✅ 训练和测试完成!")
    # final_metrics 的键名已更新
    print(f"最终测试集结果: Loss={final_metrics['loss']:.4f}, PSNR={final_metrics['psnr']:.4f}, SSIM={final_metrics['ssim']:.4f}")


if __name__ == "__main__":
    main()