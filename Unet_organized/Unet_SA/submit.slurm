#!/bin/bash -l
#SBATCH --job-name=SR_Unet_SA
#SBATCH --output=main_log.out
#SBATCH --error=main_err.out
#SBATCH --partition=gpu               # ✅ 使用 GPU 分区
#SBATCH --gres=gpu:1                  # ✅ 申请 1 块 GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=10:00:00
#SBATCH --account=leezhongpinggroup
#SBATCH --qos=normal                  # ✅ 可用的 QOS（也可改为 long）

source ~/.bashrc
conda activate pytorch                  # ⚠️ 替换为你实际的环境名

python -u main.py > main_log_bs_256.log 2>&1                       # 你的代码入口
