import os
import tarfile

# 设置.tar文件所在路径
tar_dir = '/public/home/zyye/SR_organized/Imagey'

# 遍历文件夹中的所有文件
for filename in os.listdir(tar_dir):
    if filename.endswith('.tar'):
        tar_path = os.path.join(tar_dir, filename)
        
        # 创建解压目标文件夹（与.tar文件同名）
        folder_name = filename.replace('.tar', '')
        extract_path = os.path.join(tar_dir, folder_name)
        os.makedirs(extract_path, exist_ok=True)

        # 解压，---------核心代码--------
        with tarfile.open(tar_path, 'r') as tar:
            tar.extractall(path=extract_path)
        
        print(f"已解压：{filename} → {extract_path}")

