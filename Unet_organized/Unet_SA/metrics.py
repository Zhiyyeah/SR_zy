import math
import numpy as np
from skimage.metrics import structural_similarity as ssim

def psnr(label, outputs, max_val=1.):
    """
    计算峰值信噪比 (PSNR)
    参数:
        label: 真实图像张量 (B, C, H, W)
        outputs: 预测图像张量 (B, C, H, W)
        max_val: 最大像素值
    返回:
        PSNR值 (float)
    """
    label = label.cpu().detach().numpy()
    outputs = outputs.cpu().detach().numpy()
    diff = outputs - label
    rmse = math.sqrt(np.mean((diff) ** 2))
    if rmse == 0:
        return 100
    else:
        PSNR = 20 * math.log10(max_val / rmse)
        return PSNR


def compute_ssim(label, outputs, max_val=1.0):
    """
    计算结构相似性指数 (SSIM)
    
    参数:
        label: 真实图像张量 (B, C, H, W)
        outputs: 预测图像张量 (B, C, H, W)
        max_val: 最大像素值
        
    返回:
        批次平均SSIM值 (float)
    """
    label = label.cpu().detach().numpy()
    outputs = outputs.cpu().detach().numpy()
    
    ssim_total = 0.0
    batch_size = label.shape[0]
    
    for i in range(batch_size):
        ssim_per_image = 0.0
        for c in range(label.shape[1]):
            # 为每个通道单独计算SSIM
            ssim_val = ssim(label[i, c], outputs[i, c], data_range=max_val)
            ssim_per_image += ssim_val
        ssim_total += ssim_per_image / label.shape[1]  # 通道平均

    return ssim_total / batch_size  # 批次平均

if __name__ == "__main__":
    import torch
    
    # 测试用例1：完全相同的图像
    img1 = torch.ones((1, 3, 32, 32))
    img2 = torch.ones((1, 3, 32, 32))
    psnr_val = psnr(img1, img2)
    print(f"测试1 - 相同图像PSNR: {psnr_val:.2f} (预期接近100)")
    
    # 测试用例2：添加高斯噪声
    noise = torch.randn((1, 3, 32, 32)) * 0.1
    img_noisy = img1 + noise
    psnr_val = psnr(img1, img_noisy)
    print(f"测试2 - 带噪声图像PSNR: {psnr_val:.2f}")
    
    # 测试用例3：不同尺度的图像
    img_scaled = img1 * 0.1
    psnr_val = psnr(img1, img_scaled)
    print(f"测试3 - 不同尺度图像PSNR: {psnr_val:.2f}")
