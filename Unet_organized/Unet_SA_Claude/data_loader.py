import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import rasterio

class SRDataset(Dataset):
    """
    超分辨率数据集
    """
    def __init__(self, lr_dir, hr_dir, transform=None):
        self.lr_paths = sorted(glob.glob(os.path.join(lr_dir, "*.tif")))
        self.hr_paths = sorted(glob.glob(os.path.join(hr_dir, "*.tif")))
        self.transform = transform

        assert len(self.lr_paths) == len(self.hr_paths), "低分辨率和高分辨率图像数量不匹配！"

    def __len__(self):
        return len(self.lr_paths)

    def __getitem__(self, idx):
        lr_path = self.lr_paths[idx]
        hr_path = self.hr_paths[idx]

        # 读取多通道TIF文件
        with rasterio.open(lr_path) as lr_src:
            lr_img = lr_src.read().astype(np.float32)
        with rasterio.open(hr_path) as hr_src:
            hr_img = hr_src.read().astype(np.float32)

        # 转换为torch张量
        lr_tensor = torch.tensor(lr_img, dtype=torch.float32)
        hr_tensor = torch.tensor(hr_img, dtype=torch.float32)
        
        # 应用变换（如果有）
        if self.transform:
            lr_tensor, hr_tensor = self.transform(lr_tensor, hr_tensor)
            
        return lr_tensor, hr_tensor


def create_train_val_test_dataloaders(lr_dir, hr_dir, batch_size, 
                                      train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                                      seed=42, num_workers=4, pin_memory=True):
    """
    从同一目录创建训练、验证和测试数据加载器
    
    参数:
        lr_dir: 低分辨率图像目录
        hr_dir: 高分辨率图像目录
        batch_size: 批次大小
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子
        num_workers: 数据加载线程数
        pin_memory: 是否使用内存锁定
        
    返回:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        test_loader: 测试数据加载器
    """
    # 验证比例之和为1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        f"数据集比例之和必须为1，当前为{train_ratio + val_ratio + test_ratio}"
    
    # 创建完整数据集
    full_dataset = SRDataset(lr_dir, hr_dir)
    
    # 获取数据集大小
    total_len = len(full_dataset)
    train_len = int(total_len * train_ratio)
    val_len = int(total_len * val_ratio)
    test_len = total_len - train_len - val_len  # 确保总和等于total_len
    
    # 使用随机种子拆分数据集为训练集、验证集和测试集
    train_data, val_data, test_data = random_split(
        full_dataset, 
        [train_len, val_len, test_len], 
        generator=torch.Generator().manual_seed(seed)
    )
    
    print(f"数据集总大小: {total_len}")
    print(f"- 训练集: {train_len} ({train_ratio*100:.1f}%)")
    print(f"- 验证集: {val_len} ({val_ratio*100:.1f}%)")
    print(f"- 测试集: {test_len} ({test_len/total_len*100:.1f}%)")
    
    # 创建训练数据加载器
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    # 创建验证数据加载器
    val_loader = DataLoader(
        dataset=val_data,
        batch_size=batch_size,
        shuffle=False,  # 验证集不需要打乱
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    # 创建测试数据加载器
    test_loader = DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=False,  # 测试集不需要打乱
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # 设置路径和参数
    lr_dir = 'Imagey/WaterLand_TOA_tiles_lr'
    hr_dir = 'Imagey/WaterLand_TOA_tiles_hr'
    batch_size = 4
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15
    seed = 42
    num_workers = 4
    pin_memory = True

    # 创建数据加载器
    train_loader, val_loader, test_loader = create_train_val_test_dataloaders(
        lr_dir, hr_dir, batch_size, 
        train_ratio, val_ratio, test_ratio,
        seed, num_workers, pin_memory
    )

    # 验证路径对应关系
    dataset = SRDataset(lr_dir, hr_dir)
    print("\n=== 路径对应验证（前3组）===")
    for i in range(min(3, len(dataset))):
        print(f"[{i}] LR: {os.path.basename(dataset.lr_paths[i])}")
        print(f"[{i}] HR: {os.path.basename(dataset.hr_paths[i])}")

    # 验证各数据集的张量形状和值范围
    print("\n=== 数据集验证 ===")
    
    # 训练集
    if len(train_loader) > 0:
        lr_batch, hr_batch = next(iter(train_loader))
        print(f"\n训练集：")
        print(f"- 批次数: {len(train_loader)}")
        print(f"- LR 图像尺寸: {lr_batch.shape}")
        print(f"- HR 图像尺寸: {hr_batch.shape}")
        print(f"- LR 值范围: min={lr_batch.min().item():.2f}, max={lr_batch.max().item():.2f}")
        print(f"- HR 值范围: min={hr_batch.min().item():.2f}, max={hr_batch.max().item():.2f}")
    
    # 验证集
    if len(val_loader) > 0:
        lr_batch, hr_batch = next(iter(val_loader))
        print(f"\n验证集：")
        print(f"- 批次数: {len(val_loader)}")
        print(f"- LR 图像尺寸: {lr_batch.shape}")
        print(f"- HR 图像尺寸: {hr_batch.shape}")
        print(f"- LR 值范围: min={lr_batch.min().item():.2f}, max={lr_batch.max().item():.2f}")
        print(f"- HR 值范围: min={hr_batch.min().item():.2f}, max={hr_batch.max().item():.2f}")
    
    # 测试集
    if len(test_loader) > 0:
        lr_batch, hr_batch = next(iter(test_loader))
        print(f"\n测试集：")
        print(f"- 批次数: {len(test_loader)}")
        print(f"- LR 图像尺寸: {lr_batch.shape}")
        print(f"- HR 图像尺寸: {hr_batch.shape}")
        print(f"- LR 值范围: min={lr_batch.min().item():.2f}, max={lr_batch.max().item():.2f}")
        print(f"- HR 值范围: min={hr_batch.min().item():.2f}, max={hr_batch.max().item():.2f}")
    
