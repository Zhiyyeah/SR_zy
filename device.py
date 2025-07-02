import torch
print("CUDA 是否可用:", torch.cuda.is_available())
print("GPU 数量:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"GPU {i} 名称:", torch.cuda.get_device_name(i))

# 当前 GPU ID
gpu_id = 0

# 查看当前 GPU 显存占用（单位：MB）
allocated = torch.cuda.memory_allocated(gpu_id) / 1024 / 1024
reserved = torch.cuda.memory_reserved(gpu_id) / 1024 / 1024

print(f"当前显存使用（已分配）：{allocated:.2f} MB")
print(f"当前显存使用（已保留）：{reserved:.2f} MB")
