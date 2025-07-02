import optuna
import torch
import torch.nn as nn
import torch.optim as optim

from model_attention import UNetSA
from data_loader import create_train_test_dataloaders
from metrics import psnr
from utils import get_device

# ========== 固定参数 ==========
max_epochs = 10
up_scale = 8
device = get_device()

optuna.logging.set_verbosity(optuna.logging.INFO)

# ========== 训练函数 ==========
def train_one_trial(model, train_loader, test_loader, optimizer, criterion, device, trial):
    model = model.to(device)
    model.train()

    for epoch in range(max_epochs):
        for lr_imgs, hr_imgs in train_loader:
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)

            optimizer.zero_grad()
            sr_imgs = model(lr_imgs)
            loss = criterion(sr_imgs, hr_imgs)
            loss.backward()
            optimizer.step()

        # 计算验证集 PSNR
        model.eval()
        total_psnr = 0.0
        count = 0
        with torch.no_grad():
            for lr_imgs, hr_imgs in test_loader:
                lr_imgs = lr_imgs.to(device)
                hr_imgs = hr_imgs.to(device)
                sr_imgs = model(lr_imgs)
                batch_psnr = psnr(hr_imgs, sr_imgs)
                total_psnr += batch_psnr
                count += 1
        avg_psnr = total_psnr / count

        # 输出 PSNR
        print(f"🟦 Trial {trial.number} | Epoch {epoch+1}/{max_epochs} | Val PSNR: {avg_psnr:.2f} dB", flush=True)

        # 第一轮表现太差直接跳过
        if epoch == 0 and avg_psnr < 35:
            print(f"⛔ Trial {trial.number} 被跳过：第1轮 PSNR={avg_psnr:.2f} < 35", flush=True)
            raise optuna.exceptions.TrialPruned()

        # Optuna 剪枝机制
        trial.report(-avg_psnr, step=epoch)
        if trial.should_prune():
            print(f"❌ Trial {trial.number} 被剪枝于 Epoch {epoch+1}", flush=True)
            raise optuna.exceptions.TrialPruned()

        model.train()

    return avg_psnr

# ========== 目标函数 ==========
def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    width = trial.suggest_categorical("width", [32, 64, 96])
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    loss_type = trial.suggest_categorical("loss_type", ["MSE", "L1"])

    print(f"\n📋 Trial {trial.number} | lr={lr:.5e}, width={width}, batch_size={batch_size}, loss={loss_type}", flush=True)

    train_loader, test_loader = create_train_test_dataloaders(
        lr_dir='/home/zyye/SR_backup/Imagery/Water_TOA_tiles_lr',
        hr_dir='/home/zyye/SR_backup/Imagery/Water_TOA_tiles',
        batch_size=batch_size,
        train_ratio=0.8
    )

    model = UNetSA(up_scale=up_scale, width=width)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    if loss_type == "MSE":
        criterion = nn.MSELoss()
    else:
        criterion = nn.L1Loss()

    val_psnr = train_one_trial(model, train_loader, test_loader, optimizer, criterion, device, trial)
    print(f"✅ Trial {trial.number} 完成 | Final Val PSNR: {val_psnr:.2f} dB", flush=True)
    return -val_psnr

# ========== 主流程 ==========
if __name__ == "__main__":
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(objective, n_trials=500)

    print("\n🎯 最佳参数：", study.best_trial.params, flush=True)
    print("🏆 最佳验证 PSNR：", -study.best_trial.value, flush=True)
