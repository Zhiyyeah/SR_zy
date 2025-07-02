import os
import torch

def save_model(model, save_dir, filename, metrics=None):
    """
    保存模型到磁盘
    
    参数:
        model: 要保存的模型
        save_dir: 保存目录
        filename: 文件名
        metrics: 可选的指标信息
    
    返回:
        保存的模型路径
    """
    # 确保目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建模型路径
    model_path = os.path.join(save_dir, filename)
    
    # 保存模型
    torch.save(model.state_dict(), model_path)
    
    # 如果提供了指标，保存到同名的txt文件
    if metrics:
        metrics_path = os.path.join(save_dir, f"{os.path.splitext(filename)[0]}_metrics.txt")
        with open(metrics_path, 'w') as f:
            for k, v in metrics.items():
                if isinstance(v, dict):
                    f.write(f"{k}:\n")
                    for sub_k, sub_v in v.items():
                        f.write(f"  {sub_k}: {sub_v:.4f}\n")
                else:
                    f.write(f"{k}: {v:.4f}\n")
    
    return model_path


def load_model(model, model_path, device):
    """
    从磁盘加载模型
    
    参数:
        model: 模型实例
        model_path: 模型文件路径
        device: 设备(CPU或CUDA)
    
    返回:
        加载了权重的模型
    """
    # 加载模型权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model
