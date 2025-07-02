#!/bin/bash -l
#SBATCH --job-name=SR_Unet_SA
#SBATCH --output=main_log.out
#SBATCH --error=main_err.out
#SBATCH --partition=cpu               # 🔄 改为 CPU 分区
# #SBATCH --gres=gpu:1                # 🔄 注释掉 GPU 申请
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16            # 🔄 增加 CPU 核数（根据需要调整）
#SBATCH --time=10:00:00
#SBATCH --account=leezhongpinggroup
#SBATCH --qos=normal

source ~/.bashrc
conda activate pytorch

python -u VIS_ALL.py > VIS_ALL.log 2>&1