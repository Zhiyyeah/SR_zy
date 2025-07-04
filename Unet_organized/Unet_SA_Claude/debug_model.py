import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from model_attention import UNetSA
from data_loader import create_train_val_test_dataloaders

def test_model_with_real_data():
    """使用真实数据测试模型"""
    print("🔬 使用真实数据测试模型...")
    
    # 加载真实数据
    lr_dir = r"D:\Py_Code\Unet_SR\SR_zy\Imagey\Imagery_WaterLand\WaterLand_TOA_tiles_lr"
    hr_dir = r"D:\Py_Code\Unet_SR\SR_zy\Imagey\Imagery_WaterLand\WaterLand_TOA_tiles_hr"
    
    train_loader, _, _ = create_train_val_test_dataloaders(
        lr_dir, hr_dir, batch_size=1, 
        train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
        seed=42, num_workers=0, pin_memory=False
    )
    
    # 获取一个真实样本
    lr_real, hr_real = next(iter(train_loader))
    
    print(f"真实LR范围: [{lr_real.min():.6f}, {lr_real.max():.6f}]")
    print(f"真实HR范围: [{hr_real.min():.6f}, {hr_real.max():.6f}]")
    
    # 创建未训练的模型
    model = UNetSA(up_scale=8, img_channel=7, width=64, dropout_rate=0.1)
    model.eval()
    
    with torch.no_grad():
        # 测试模型各个组件
        print("\n🔍 分析模型各个组件:")
        
        # 1. 双线性插值基线
        up_input = F.interpolate(lr_real, scale_factor=8, mode='bilinear', align_corners=False)
        print(f"双线性插值结果范围: [{up_input.min():.6f}, {up_input.max():.6f}]")
        
        # 2. 完整模型输出
        sr_output = model(lr_real)
        print(f"模型完整输出范围: [{sr_output.min():.6f}, {sr_output.max():.6f}]")
        
        # 3. 分析残差连接
        # 我们需要修改模型来获取残差部分
        # 先获取模型内部的残差预测
        x = model._check_image_size(lr_real)
        up_input_internal = F.interpolate(x, scale_factor=model.up_scale, mode='bilinear', align_corners=False)
        
        # 编码部分
        x1 = model.input_conv(x)
        x2 = model.down1(x1)
        x3 = model.down2(x2)
        x4 = model.down3(x3)
        x5 = model.down4(x4)
        
        if model.use_attention:
            x5 = model.bottleneck_att(x5)
        
        # 解码部分
        x = model.up1(x5, x4)
        x = model.up2(x, x3)
        x = model.up3(x, x2)
        x = model.up4(x, x1)
        
        # 超分辨率部分
        x = model.sr_up1(x)
        x = model.sr_up2(x)
        x = model.sr_up3(x)
        x = model.output_conv(x)
        if model.use_attention:
            x = model.final_att(x)
        
        # 这是残差部分（模型学习的增量）
        residual = x
        print(f"模型学习的残差范围: [{residual.min():.6f}, {residual.max():.6f}]")
        print(f"残差均值: {residual.mean():.6f}")
        print(f"残差标准差: {residual.std():.6f}")
        
        # 最终输出 = 残差 + 双线性插值
        final_output = residual + up_input_internal
        print(f"最终输出范围: [{final_output.min():.6f}, {final_output.max():.6f}]")
        
        # 检查是否有异常值
        if torch.isnan(final_output).any():
            print("⚠️ 警告: 输出包含NaN!")
        if torch.isinf(final_output).any():
            print("⚠️ 警告: 输出包含Inf!")

def test_loss_scale():
    """测试损失函数的尺度"""
    print("\n📏 测试损失函数尺度:")
    
    # 模拟真实数据范围的损失
    hr_sim = torch.rand(1, 7, 512, 512) * 0.18 + 0.003  # 0.003-0.18范围
    sr_sim = torch.rand(1, 7, 512, 512) * 0.18 + 0.003
    
    mse_loss = torch.nn.MSELoss()
    loss_real_scale = mse_loss(sr_sim, hr_sim)
    
    print(f"真实数据尺度的MSE损失: {loss_real_scale.item():.8f}")
    
    # 如果损失很小，可能需要调整学习率或损失函数
    if loss_real_scale.item() < 1e-4:
        print("⚠️ 警告: 损失值非常小，可能需要调整学习率或使用不同的损失函数")

def analyze_gradient_flow():
    """分析梯度流"""
    print("\n🌊 分析梯度流:")
    
    # 创建模型
    model = UNetSA(up_scale=8, img_channel=7, width=64, dropout_rate=0.1)
    
    # 创建真实范围的输入和目标
    lr_input = torch.rand(1, 7, 64, 64) * 0.18 + 0.003
    hr_target = torch.rand(1, 7, 512, 512) * 0.18 + 0.003
    
    lr_input.requires_grad_(True)
    
    # 前向传播
    sr_output = model(lr_input)
    
    # 计算损失
    criterion = torch.nn.MSELoss()
    loss = criterion(sr_output, hr_target)
    
    print(f"输入范围: [{lr_input.min():.6f}, {lr_input.max():.6f}]")
    print(f"输出范围: [{sr_output.min():.6f}, {sr_output.max():.6f}]")
    print(f"目标范围: [{hr_target.min():.6f}, {hr_target.max():.6f}]")
    print(f"损失值: {loss.item():.8f}")
    
    # 反向传播
    loss.backward()
    
    # 检查梯度
    if lr_input.grad is not None:
        grad_norm = lr_input.grad.norm().item()
        print(f"输入梯度范数: {grad_norm:.8f}")
        
        if grad_norm > 100:
            print("⚠️ 警告: 梯度可能过大，可能导致梯度爆炸")
        elif grad_norm < 1e-8:
            print("⚠️ 警告: 梯度过小，可能导致梯度消失")

def visualize_model_behavior():
    """可视化模型行为"""
    print("\n📊 可视化模型行为:")
    
    # 加载真实数据
    lr_dir = r"D:\Py_Code\Unet_SR\SR_zy\Imagey\Imagery_WaterLand\WaterLand_TOA_tiles_lr"
    hr_dir = r"D:\Py_Code\Unet_SR\SR_zy\Imagey\Imagery_WaterLand\WaterLand_TOA_tiles_hr"
    
    train_loader, _, _ = create_train_val_test_dataloaders(
        lr_dir, hr_dir, batch_size=1, 
        train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
        seed=42, num_workers=0, pin_memory=False
    )
    
    lr_real, hr_real = next(iter(train_loader))
    
    # 未训练模型
    model = UNetSA(up_scale=8, img_channel=7, width=64, dropout_rate=0.1)
    model.eval()
    
    with torch.no_grad():
        sr_output = model(lr_real)
        
        # 双线性插值作为基线
        bilinear = F.interpolate(lr_real, scale_factor=8, mode='bilinear', align_corners=False)
        
        # 选择第一个通道进行可视化
        lr_img = lr_real[0, 0].numpy()
        hr_img = hr_real[0, 0].numpy()
        sr_img = sr_output[0, 0].numpy()
        bi_img = bilinear[0, 0].numpy()
        
        # 创建对比图
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # LR
        im1 = axes[0, 0].imshow(lr_img, cmap='gray')
        axes[0, 0].set_title(f'LR Input\nRange: [{lr_img.min():.4f}, {lr_img.max():.4f}]')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # HR Ground Truth
        im2 = axes[0, 1].imshow(hr_img, cmap='gray')
        axes[0, 1].set_title(f'HR Ground Truth\nRange: [{hr_img.min():.4f}, {hr_img.max():.4f}]')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # Bilinear Interpolation
        im3 = axes[1, 0].imshow(bi_img, cmap='gray')
        axes[1, 0].set_title(f'Bilinear Interpolation\nRange: [{bi_img.min():.4f}, {bi_img.max():.4f}]')
        plt.colorbar(im3, ax=axes[1, 0])
        
        # Model Output
        im4 = axes[1, 1].imshow(sr_img, cmap='gray')
        axes[1, 1].set_title(f'Model Output (Untrained)\nRange: [{sr_img.min():.4f}, {sr_img.max():.4f}]')
        plt.colorbar(im4, ax=axes[1, 1])
        
        plt.tight_layout()
        plt.savefig('model_behavior_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ 模型行为分析图已保存为 'model_behavior_analysis.png'")

def main():
    """主函数"""
    print("🚀 开始模型残差连接和行为分析...")
    
    try:
        # 1. 使用真实数据测试模型
        test_model_with_real_data()
        
        # 2. 测试损失函数尺度
        test_loss_scale()
        
        # 3. 分析梯度流
        analyze_gradient_flow()
        
        # 4. 可视化模型行为
        visualize_model_behavior()
        
        print(f"\n✅ 分析完成！")
        
    except Exception as e:
        print(f"❌ 分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()