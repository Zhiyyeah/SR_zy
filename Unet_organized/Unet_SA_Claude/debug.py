import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from data_loader import create_train_val_test_dataloaders, SRDataset
import rasterio

def check_data_quality():
    """检查数据质量和范围"""
    
    # 配置路径
    lr_dir = r"D:\Py_Code\Unet_SR\SR_zy\Imagey\Imagery_WaterLand\WaterLand_TOA_tiles_lr"
    hr_dir = r"D:\Py_Code\Unet_SR\SR_zy\Imagey\Imagery_WaterLand\WaterLand_TOA_tiles_hr"
    
    print("🔍 开始数据质量检查...")
    
    # 1. 检查数据集基本信息
    dataset = SRDataset(lr_dir, hr_dir)
    print(f"\n📊 数据集信息:")
    print(f"总样本数: {len(dataset)}")
    
    # 2. 检查前几个样本的原始数据
    print(f"\n🔬 检查前3个样本的原始数据:")
    for i in range(min(3, len(dataset))):
        lr_path = dataset.lr_paths[i]
        hr_path = dataset.hr_paths[i]
        
        print(f"\n--- 样本 {i} ---")
        print(f"LR: {os.path.basename(lr_path)}")
        print(f"HR: {os.path.basename(hr_path)}")
        
        # 直接读取原始文件
        with rasterio.open(lr_path) as src:
            lr_raw = src.read()
            print(f"LR 原始形状: {lr_raw.shape}")
            print(f"LR 原始数据类型: {lr_raw.dtype}")
            print(f"LR 原始值范围: [{lr_raw.min():.6f}, {lr_raw.max():.6f}]")
            print(f"LR 原始均值: {lr_raw.mean():.6f}")
            print(f"LR 是否有NaN: {np.isnan(lr_raw).any()}")
            print(f"LR 是否有Inf: {np.isinf(lr_raw).any()}")
        
        with rasterio.open(hr_path) as src:
            hr_raw = src.read()
            print(f"HR 原始形状: {hr_raw.shape}")
            print(f"HR 原始数据类型: {hr_raw.dtype}")
            print(f"HR 原始值范围: [{hr_raw.min():.6f}, {hr_raw.max():.6f}]")
            print(f"HR 原始均值: {hr_raw.mean():.6f}")
            print(f"HR 是否有NaN: {np.isnan(hr_raw).any()}")
            print(f"HR 是否有Inf: {np.isinf(hr_raw).any()}")
        
        # 检查通过Dataset类加载的数据
        lr_tensor, hr_tensor = dataset[i]
        print(f"\nDataset加载后:")
        print(f"LR tensor形状: {lr_tensor.shape}")
        print(f"LR tensor值范围: [{lr_tensor.min():.6f}, {lr_tensor.max():.6f}]")
        print(f"HR tensor形状: {hr_tensor.shape}")
        print(f"HR tensor值范围: [{hr_tensor.min():.6f}, {hr_tensor.max():.6f}]")

def visualize_samples():
    """可视化样本数据"""
    
    lr_dir = r"D:\Py_Code\Unet_SR\SR_zy\Imagey\Imagery_WaterLand\WaterLand_TOA_tiles_lr"
    hr_dir = r"D:\Py_Code\Unet_SR\SR_zy\Imagey\Imagery_WaterLand\WaterLand_TOA_tiles_hr"
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = create_train_val_test_dataloaders(
        lr_dir, hr_dir, batch_size=1, 
        train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
        seed=42, num_workers=0, pin_memory=False
    )
    
    # 获取一个batch
    lr_batch, hr_batch = next(iter(train_loader))
    lr_img = lr_batch[0]  # [C, H, W]
    hr_img = hr_batch[0]  # [C, H, W]
    
    print(f"\n📸 可视化样本:")
    print(f"LR图像形状: {lr_img.shape}")
    print(f"HR图像形状: {hr_img.shape}")
    
    # 选择RGB通道进行可视化 (假设通道3,2,1对应RGB)
    if lr_img.shape[0] >= 4:  # 确保有足够的通道
        # 选择前3个通道作为RGB
        lr_rgb = lr_img[:3].permute(1, 2, 0).numpy()
        hr_rgb = hr_img[:3].permute(1, 2, 0).numpy()
        
        # 归一化到0-1范围用于显示
        def normalize_for_display(img):
            img_min, img_max = img.min(), img.max()
            if img_max > img_min:
                return (img - img_min) / (img_max - img_min)
            return img
        
        lr_rgb_norm = normalize_for_display(lr_rgb)
        hr_rgb_norm = normalize_for_display(hr_rgb)
        
        # 创建图像显示
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # LR图像
        axes[0, 0].imshow(lr_rgb_norm)
        axes[0, 0].set_title(f'LR图像 {lr_img.shape[1]}x{lr_img.shape[2]}')
        axes[0, 0].axis('off')
        
        # HR图像  
        axes[0, 1].imshow(hr_rgb_norm)
        axes[0, 1].set_title(f'HR图像 {hr_img.shape[1]}x{hr_img.shape[2]}')
        axes[0, 1].axis('off')
        
        # LR第一个通道的直方图
        axes[1, 0].hist(lr_img[0].flatten().numpy(), bins=50, alpha=0.7)
        axes[1, 0].set_title('LR第一通道值分布')
        axes[1, 0].set_xlabel('像素值')
        axes[1, 0].set_ylabel('频次')
        
        # HR第一个通道的直方图
        axes[1, 1].hist(hr_img[0].flatten().numpy(), bins=50, alpha=0.7)
        axes[1, 1].set_title('HR第一通道值分布')
        axes[1, 1].set_xlabel('像素值')
        axes[1, 1].set_ylabel('频次')
        
        plt.tight_layout()
        plt.savefig('data_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ 图像已保存为 'data_visualization.png'")

def check_model_output():
    """检查模型输出"""
    from model_attention import UNetSA
    
    print(f"\n🤖 检查模型输出:")
    
    # 创建模型
    model = UNetSA(up_scale=8, img_channel=7, width=64, dropout_rate=0.1)
    model.eval()
    
    # 创建测试输入
    test_input = torch.randn(1, 7, 64, 64)
    print(f"测试输入形状: {test_input.shape}")
    print(f"测试输入值范围: [{test_input.min():.6f}, {test_input.max():.6f}]")
    
    # 前向传播
    with torch.no_grad():
        output = model(test_input)
    
    print(f"模型输出形状: {output.shape}")
    print(f"模型输出值范围: [{output.min():.6f}, {output.max():.6f}]")
    print(f"模型输出均值: {output.mean():.6f}")
    print(f"模型输出标准差: {output.std():.6f}")
    
    # 检查是否有异常值
    if torch.isnan(output).any():
        print("⚠️ 警告: 模型输出包含NaN值!")
    if torch.isinf(output).any():
        print("⚠️ 警告: 模型输出包含Inf值!")

def check_loss_function():
    """检查损失函数行为"""
    print(f"\n📉 检查损失函数:")
    
    # 创建测试数据
    hr_test = torch.randn(1, 7, 512, 512)
    sr_test = torch.randn(1, 7, 512, 512) 
    
    # 测试MSE损失
    mse_loss = torch.nn.MSELoss()
    loss_value = mse_loss(sr_test, hr_test)
    
    print(f"HR测试数据范围: [{hr_test.min():.6f}, {hr_test.max():.6f}]")
    print(f"SR测试数据范围: [{sr_test.min():.6f}, {sr_test.max():.6f}]")
    print(f"MSE损失值: {loss_value.item():.6f}")

def main():
    """主函数"""
    print("🚀 开始超分辨率模型诊断...")
    
    try:
        # 1. 检查数据质量
        check_data_quality()
        
        # 2. 可视化样本
        print(f"\n" + "="*50)
        visualize_samples()
        
        # 3. 检查模型输出
        print(f"\n" + "="*50)
        check_model_output()
        
        # 4. 检查损失函数
        print(f"\n" + "="*50)
        check_loss_function()
        
        print(f"\n✅ 诊断完成！请查看上述输出来识别问题。")
        
    except Exception as e:
        print(f"❌ 诊断过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()