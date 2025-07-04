import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils import load_image, bilinear_interpolation, visualize_results, get_device, save_metrics_to_file
from model_attention import UNetSA

# ==== 用户可自定义参数 ====
LR_DIR        = r"D:\Py_Code\Unet_SR\SR_zy\Imagey\Small_Dataset\lr"
HR_DIR        = r"D:\Py_Code\Unet_SR\SR_zy\Imagey\Small_Dataset\hr"
MODEL_PATH    = r"D:\Py_Code\Unet_SR\SR_zy\Unet_organized\Unet_SA_Claude\outputs\zy_first_optimized_0.00043\models\best_model.pth"
OUTPUT_DIR    = r"D:\Py_Code\Unet_SR\SR_zy\Unet_organized\Unet_SA_Claude\outputs\zy_first_optimized_0.00043\test_results"
IDX           = 68           # 指定景序号(0-based)，不想遍历所有则设为 int，否则设为 None
UP_SCALE      = 8
WIDTH         = 64
DROPOUT_RATE  = 0.1
RGB_CHANNELS  = [3, 2, 1]    # 1-based 波段序号
# ==========================

def load_model(path, device, up_scale, num_channels, width, dropout_rate=0.1):
    model = UNetSA(
        up_scale=up_scale,
        img_channel=num_channels,
        width=width,
        dropout_rate=dropout_rate,
        use_attention=True
    ).to(device)
    print(f"加载模型: {path}")
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"检测到新格式checkpoint，Epoch: {ckpt.get('epoch', 'unknown')}")
        if 'test_metrics' in ckpt:
            metrics = ckpt['test_metrics']
            print(f"保存时的测试指标:")
            print(f"  - Loss: {metrics.get('loss', 'N/A'):.4f}")
            print(f"  - PSNR: {metrics.get('psnr', 'N/A'):.2f} dB")
            print(f"  - SSIM: {metrics.get('ssim', 'N/A'):.4f}")
    else:
        model.load_state_dict(ckpt)
        print("加载模型成功（旧格式）")
    model.eval()
    return model

def calculate_metrics(hr, sr):
    mse = torch.mean((hr - sr) ** 2).item()
    if mse == 0:
        psnr = 100
    else:
        max_pixel = 1.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    # 简单SSIM实现
    mean_hr = torch.mean(hr).item()
    mean_sr = torch.mean(sr).item()
    var_hr = torch.var(hr).item()
    var_sr = torch.var(sr).item()
    cov = torch.mean((hr - mean_hr) * (sr - mean_sr)).item()
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim = ((2 * mean_hr * mean_sr + c1) * (2 * cov + c2)) / \
           ((mean_hr ** 2 + mean_sr ** 2 + c1) * (var_hr + var_sr + c2))
    return psnr, ssim

def validate_scene(lr_path, hr_path, model, device):
    print(f"  读取LR: {os.path.basename(lr_path)}")
    lr = load_image(lr_path).to(device) / 10000.0
    print(f"  读取HR: {os.path.basename(hr_path)}")
    hr = load_image(hr_path).to(device) / 10000.0
    print(f"  LR shape: {tuple(lr.shape)}, range: [{lr.min():.3f}, {lr.max():.3f}]")
    print(f"  HR shape: {tuple(hr.shape)}, range: [{hr.min():.3f}, {hr.max():.3f}]")
    lr = lr.unsqueeze(0)
    hr = hr.unsqueeze(0)
    with torch.no_grad():
        sr = model(lr)
    bilinear = bilinear_interpolation(lr, hr.shape[-2:])
    sr = sr.squeeze(0)
    bilinear = bilinear.squeeze(0)
    hr = hr.squeeze(0)
    psnr_sr, ssim_sr = calculate_metrics(hr, sr)
    psnr_bi, ssim_bi = calculate_metrics(hr, bilinear)
    print(f"  SR结果 - PSNR: {psnr_sr:.2f} dB, SSIM: {ssim_sr:.4f}")
    print(f"  双线性 - PSNR: {psnr_bi:.2f} dB, SSIM: {ssim_bi:.4f}")
    print(f"  改善: PSNR +{psnr_sr - psnr_bi:.2f} dB")
    metrics = {
        "sr_psnr": psnr_sr,
        "sr_ssim": ssim_sr,
        "bilinear_psnr": psnr_bi,
        "bilinear_ssim": ssim_bi
    }
    return lr.squeeze(0).cpu(), bilinear.cpu(), sr.cpu(), hr.cpu(), metrics

def main():
    print("="*60)
    print("超分辨率模型验证 (new_eval.py)")
    print("="*60)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    lr_list = sorted(glob.glob(os.path.join(LR_DIR, "*.tif")))
    hr_list = sorted(glob.glob(os.path.join(HR_DIR, "*.tif")))
    print(f"\n找到 {len(lr_list)} 个LR文件")
    print(f"找到 {len(hr_list)} 个HR文件")
    assert len(lr_list) == len(hr_list), "LR/HR 数量不匹配！"
    assert len(lr_list) > 0, "没有找到图像文件！"
    device = get_device()
    print(f"\n使用设备: {device}")
    tmp = load_image(lr_list[0])
    num_bands = tmp.shape[0]
    print(f"检测到 {num_bands} 个波段")
    print(f"\n加载模型...")
    model = load_model(MODEL_PATH, device, UP_SCALE, num_bands, WIDTH, DROPOUT_RATE)
    if IDX is not None:
        indices = [IDX]
        print(f"\n只处理场景 #{IDX}")
    else:
        indices = list(range(len(lr_list)))
        print(f"\n将处理所有 {len(indices)} 个场景")
    all_metrics = []
    for i in tqdm(indices, desc="处理场景"):
        print(f"\n>>> 场景 #{i}")
        lr, bilinear, sr, hr, metrics = validate_scene(
            lr_list[i], hr_list[i], model, device
        )
        all_metrics.append(metrics)
        cmp_path = os.path.join(OUTPUT_DIR, f"scene_{i:03d}_comparison.png")
        visualize_results(lr, bilinear, sr, hr, metrics, cmp_path, rgb_channels=RGB_CHANNELS)
    if len(all_metrics) > 0:
        avg_psnr_sr = np.mean([m['sr_psnr'] for m in all_metrics])
        avg_ssim_sr = np.mean([m['sr_ssim'] for m in all_metrics])
        avg_psnr_bi = np.mean([m['bilinear_psnr'] for m in all_metrics])
        avg_ssim_bi = np.mean([m['bilinear_ssim'] for m in all_metrics])
        print("\n" + "="*60)
        print("验证完成！")
        print(f"\n平均指标 ({len(all_metrics)} 个场景):")
        print(f"  超分辨率 - PSNR: {avg_psnr_sr:.2f} dB, SSIM: {avg_ssim_sr:.4f}")
        print(f"  双线性   - PSNR: {avg_psnr_bi:.2f} dB, SSIM: {avg_ssim_bi:.4f}")
        print(f"  平均改善 - PSNR: +{avg_psnr_sr - avg_psnr_bi:.2f} dB")
        print("="*60)
        metrics_path = os.path.join(OUTPUT_DIR, "validation_metrics.txt")
        summary = {
            "avg_sr_psnr": avg_psnr_sr,
            "avg_sr_ssim": avg_ssim_sr,
            "avg_bilinear_psnr": avg_psnr_bi,
            "avg_bilinear_ssim": avg_ssim_bi,
            "psnr_improvement": avg_psnr_sr - avg_psnr_bi
        }
        save_metrics_to_file(summary, metrics_path)

if __name__ == "__main__":
    main()