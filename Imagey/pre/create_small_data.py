import os
import shutil
import random
from pathlib import Path

def create_small_dataset(
    hr_source_dir, 
    lr_source_dir, 
    hr_dest_dir, 
    lr_dest_dir, 
    num_samples=100,
    seed=42
):
    """
    从原始数据集中随机选择指定数量的影像对，创建小数据集用于测试
    
    Args:
        hr_source_dir: 高分辨率图像源目录
        lr_source_dir: 低分辨率图像源目录  
        hr_dest_dir: 高分辨率图像目标目录
        lr_dest_dir: 低分辨率图像目标目录
        num_samples: 要选择的样本数量
        seed: 随机种子
    """
    
    # 设置随机种子
    random.seed(seed)
    
    # 检查源目录是否存在
    if not os.path.exists(hr_source_dir):
        raise FileNotFoundError(f"HR源目录不存在: {hr_source_dir}")
    if not os.path.exists(lr_source_dir):
        raise FileNotFoundError(f"LR源目录不存在: {lr_source_dir}")
    
    # 获取所有图像文件
    hr_files = []
    lr_files = []
    
    # 支持的图像格式
    image_extensions = {'.tif', '.tiff', '.png', '.jpg', '.jpeg'}
    
    print("📁 扫描源目录...")
    
    # 获取HR文件列表
    for file in os.listdir(hr_source_dir):
        if Path(file).suffix.lower() in image_extensions:
            hr_files.append(file)
    
    # 获取LR文件列表
    for file in os.listdir(lr_source_dir):
        if Path(file).suffix.lower() in image_extensions:
            lr_files.append(file)
    
    print(f"  • HR目录中找到 {len(hr_files)} 个图像文件")
    print(f"  • LR目录中找到 {len(lr_files)} 个图像文件")
    
    # 找到匹配的文件对
    hr_set = set(hr_files)
    lr_set = set(lr_files)
    common_files = list(hr_set.intersection(lr_set))
    
    print(f"  • 找到 {len(common_files)} 对匹配的图像")
    
    if len(common_files) == 0:
        raise ValueError("没有找到匹配的图像对！请检查文件名是否一致。")
    
    # 检查请求的样本数量
    if num_samples > len(common_files):
        print(f"⚠️  请求的样本数量({num_samples})超过可用数量({len(common_files)})，将使用全部可用样本")
        num_samples = len(common_files)
    
    # 随机选择样本
    selected_files = random.sample(common_files, num_samples)
    
    print(f"\n🎯 随机选择了 {len(selected_files)} 个样本")
    
    # 创建目标目录
    os.makedirs(hr_dest_dir, exist_ok=True)
    os.makedirs(lr_dest_dir, exist_ok=True)
    
    print(f"\n📋 创建目标目录:")
    print(f"  • HR目标目录: {hr_dest_dir}")
    print(f"  • LR目标目录: {lr_dest_dir}")
    
    # 复制选中的文件
    print("\n📂 开始复制文件...")
    success_count = 0
    
    for i, filename in enumerate(selected_files, 1):
        try:
            # 复制HR文件
            hr_src = os.path.join(hr_source_dir, filename)
            hr_dst = os.path.join(hr_dest_dir, filename)
            shutil.copy2(hr_src, hr_dst)
            
            # 复制LR文件
            lr_src = os.path.join(lr_source_dir, filename)
            lr_dst = os.path.join(lr_dest_dir, filename)
            shutil.copy2(lr_src, lr_dst)
            
            success_count += 1
            
            # 显示进度
            if i % 10 == 0 or i == len(selected_files):
                print(f"  进度: {i}/{len(selected_files)} ({i/len(selected_files)*100:.1f}%)")
                
        except Exception as e:
            print(f"  ❌ 复制文件失败 {filename}: {e}")
    
    print(f"\n✅ 完成！成功复制了 {success_count} 对图像")
    
    # 验证结果
    hr_result_count = len([f for f in os.listdir(hr_dest_dir) 
                          if Path(f).suffix.lower() in image_extensions])
    lr_result_count = len([f for f in os.listdir(lr_dest_dir) 
                          if Path(f).suffix.lower() in image_extensions])
    
    print(f"\n📊 结果验证:")
    print(f"  • HR目标目录中有 {hr_result_count} 个文件")
    print(f"  • LR目标目录中有 {lr_result_count} 个文件")
    
    if hr_result_count == lr_result_count == success_count:
        print("  ✅ 数据集创建成功！")
    else:
        print("  ⚠️  文件数量不匹配，请检查！")
    
    return selected_files


def main():
    """主函数"""
    # 源数据路径
    hr_source = r'D:\Py_Code\Unet_SR\SR_zy\Imagey\Imagery_WaterLand\WaterLand_TOA_tiles_hr'
    lr_source = r'D:\Py_Code\Unet_SR\SR_zy\Imagey\Imagery_WaterLand\WaterLand_TOA_tiles_lr'
    
    # 目标数据路径（小数据集）
    hr_dest = r'D:\Py_Code\Unet_SR\SR_zy\Imagey\Small_Dataset\hr'
    lr_dest = r'D:\Py_Code\Unet_SR\SR_zy\Imagey\Small_Dataset\lr'
    
    # 样本数量
    num_samples = 100
    
    print("="*60)
    print("创建小数据集用于模型测试")
    print("="*60)
    
    print(f"\n🔧 配置信息:")
    print(f"  • HR源目录: {hr_source}")
    print(f"  • LR源目录: {lr_source}")
    print(f"  • HR目标目录: {hr_dest}")
    print(f"  • LR目标目录: {lr_dest}")
    print(f"  • 样本数量: {num_samples}")
    
    try:
        selected_files = create_small_dataset(
            hr_source_dir=hr_source,
            lr_source_dir=lr_source,
            hr_dest_dir=hr_dest,
            lr_dest_dir=lr_dest,
            num_samples=num_samples,
            seed=42
        )
        
        print(f"\n🎉 小数据集创建完成！")
        print(f"📝 选中的文件列表（前10个）:")
        for i, filename in enumerate(selected_files[:10]):
            print(f"  {i+1:2d}. {filename}")
        if len(selected_files) > 10:
            print(f"  ... 还有 {len(selected_files)-10} 个文件")
            
        print(f"\n💡 接下来可以修改训练配置使用小数据集:")
        print(f"lr_dir = '{lr_dest.replace(chr(92), '/')}'")
        print(f"hr_dir = '{hr_dest.replace(chr(92), '/')}'")
        
    except Exception as e:
        print(f"\n❌ 创建数据集时出错: {e}")


if __name__ == "__main__":
    main()