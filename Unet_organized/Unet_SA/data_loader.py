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
        """
        初始化数据集
        
        参数:
            lr_dir: 低分辨率图像目录
            hr_dir: 高分辨率图像目录
            transform: 可选的数据变换
        """
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


def create_train_test_dataloaders(lr_dir, hr_dir, batch_size, train_ratio=0.8, seed=42, num_workers=4, pin_memory=True):
    """
    从同一目录创建训练和测试数据加载器
    
    参数:
        lr_dir: 低分辨率图像目录
        hr_dir: 高分辨率图像目录
        batch_size: 批次大小
        train_ratio: 训练集比例
        seed: 随机种子
        num_workers: 数据加载线程数
        pin_memory: 是否使用内存锁定
        
    返回:
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
    """
    # 创建完整数据集
    full_dataset = SRDataset(lr_dir, hr_dir)
    
    # 获取数据集大小
    total_len = len(full_dataset)
    train_len = int(total_len * train_ratio)
    test_len = total_len - train_len
    
    # 使用随机种子拆分数据集为训练集和测试集
    train_data, test_data = random_split(
        full_dataset, 
        [train_len, test_len], 
        generator=torch.Generator().manual_seed(seed)
    )
    
    print(f"数据集总大小: {total_len}，训练集: {train_len}，测试集: {test_len}")
    
    # 创建训练数据加载器
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    # 创建测试数据加载器
    test_loader = DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    return train_loader, test_loader


if __name__ == "__main__":

    # 设置路径和参数
    lr_dir = '/home/zyye/SR_backup/Imagery/Water_TOA_tiles_lr'
    hr_dir = '/home/zyye/SR_backup/Imagery/Water_TOA_tiles'
    batch_size = 4
    train_ratio = 0.8
    seed = 42
    num_workers = 4
    pin_memory = True

    # 创建数据加载器
    train_loader, test_loader = create_train_test_dataloaders(
        lr_dir, hr_dir, batch_size, train_ratio, seed, num_workers, pin_memory
    )

    # 验证路径对应关系
    dataset = SRDataset(lr_dir, hr_dir)
    print("=== 路径对应验证（前3组）===")
    for i in range(3):
        print(f"[{i}] LR: {dataset.lr_paths[i]}")
        print(f"[{i}] HR: {dataset.hr_paths[i]}")

    # 验证张量形状和值范围（训练集）
    lr_batch, hr_batch = next(iter(train_loader))
    print(f"\n训练集：")
    print(f"- LR 图像尺寸: {lr_batch.shape}")
    print(f"- HR 图像尺寸: {hr_batch.shape}")
    print(f"- LR 值范围: min={lr_batch.min().item():.2f}, max={lr_batch.max().item():.2f}")
    print(f"- HR 值范围: min={hr_batch.min().item():.2f}, max={hr_batch.max().item():.2f}")


