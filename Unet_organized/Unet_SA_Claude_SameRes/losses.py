"""
仅提供两种可选损失（按你的要求精简）：

- "charbonnier"：纯 Charbonnier（L1 的平滑变体），对异常值更鲁棒。
- "charbonnier+ssim"：加权组合损失 = alpha * Charbonnier + beta * (1 - SSIM)。

权重可调：
- `alpha`（Charbonnier 权重，默认 1.0）
- `beta`（1-SSIM 权重，建议 0.02~0.2，默认 0.05）

使用方式（见 main.py 中示例）：
loss_fn = make_loss("charbonnier+ssim", alpha=1.0, beta=0.05, ssim_data_range=1.0)

说明：
- 如果你的输入/标签已归一化到 [0,1]，`ssim_data_range` 传 1.0 更稳；否则留 None，代码会从 batch 内自动估计。
- 只想用 Charbonnier：make_loss("charbonnier")。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Compute in float32 to avoid FP16 overflow under AMP when values are large
        orig_dtype = pred.dtype
        diff = (pred - target).float()
        eps2 = torch.as_tensor(self.eps, dtype=diff.dtype, device=diff.device) ** 2
        loss = torch.sqrt(diff * diff + eps2)
        return loss.mean().to(orig_dtype)


class SSIM(nn.Module):
    """可微分 SSIM（适用于 BxCxHxW）。

    - 使用高斯窗口逐通道计算，再对所有像素/通道取平均。
    - `data_range` 为像素的动态范围（例如 1.0 或 255.0）。若为 None，则按当前 batch 的 target 动态估计。
    """

    def __init__(self, window_size: int = 11, sigma: float = 1.5, data_range: float = None, reduction: str = "mean"):
        super().__init__()
        assert window_size % 2 == 1, "window_size must be odd"
        self.window_size = window_size
        self.sigma = sigma
        self.data_range = data_range
        self.reduction = reduction

        # Create 2D Gaussian kernel (normalized)
        coords = torch.arange(window_size).float() - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = (g / g.sum()).unsqueeze(1)
        kernel_2d = (g @ g.t())
        self.register_buffer("kernel", kernel_2d)

    def _filter(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        k = self.kernel.to(dtype=x.dtype, device=x.device)
        k = k.expand(C, 1, self.window_size, self.window_size).contiguous()
        pad = self.window_size // 2
        return F.conv2d(x, k, padding=pad, groups=C)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert x.shape == y.shape, "SSIM expects same shape for x and y"
        # 为避免 AMP(半精度) 下的下溢导致 NaN，这里内部用 float32 做计算
        orig_dtype = x.dtype
        x = x.float()
        y = y.float()

        # 均值
        mu_x = self._filter(x)
        mu_y = self._filter(y)
        # 方差与协方差
        sigma_x2 = self._filter(x * x) - mu_x * mu_x
        sigma_y2 = self._filter(y * y) - mu_y * mu_y
        sigma_xy = self._filter(x * y) - mu_x * mu_y

        # 动态范围（避免为 0 导致 C1/C2 下溢）
        if self.data_range is None:
            L = (y.max() - y.min()).detach()
        else:
            L = torch.tensor(self.data_range, dtype=y.dtype, device=y.device)
        L = torch.clamp(L, min=1e-3)  # 放大下界，防止半精度下 C1/C2 变成 0
        C1 = (0.01 * L) ** 2
        C2 = (0.03 * L) ** 2
        C1 = torch.clamp(C1, min=1e-6)
        C2 = torch.clamp(C2, min=1e-6)

        num = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        den = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x2 + sigma_y2 + C2)
        ssim_map = num / (den + 1e-12)
        if self.reduction == "mean":
            return ssim_map.mean().to(orig_dtype)
        if self.reduction == "sum":
            return ssim_map.sum().to(orig_dtype)
        return ssim_map.to(orig_dtype)


class SSIMLoss(nn.Module):
    """SSIM 的互补损失：1 - SSIM。"""

    def __init__(self, window_size: int = 11, sigma: float = 1.5, data_range: float = None):
        super().__init__()
        self.ssim = SSIM(window_size=window_size, sigma=sigma, data_range=data_range)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return 1.0 - self.ssim(pred, target)


class CombinedLoss(nn.Module):
    """组合损失：alpha * Charbonnier + beta * (1 - SSIM)。

    只保留 Charbonnier 作为基础项，权重 `alpha`、`beta` 可调。
    """

    def __init__(self,
                 alpha: float = 1.0,
                 beta: float = 20.0,
                 charbonnier_eps: float = 1e-3,
                 ssim_window_size: int = 11,
                 ssim_sigma: float = 1.5,
                 ssim_data_range: float = None):
        super().__init__()
        self.charb = CharbonnierLoss(eps=charbonnier_eps)
        self.alpha = alpha
        self.beta = beta
        self.ssim_loss = SSIMLoss(window_size=ssim_window_size,
                                  sigma=ssim_sigma,
                                  data_range=ssim_data_range)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.alpha * self.charb(pred, target) + self.beta * self.ssim_loss(pred, target)


def make_loss(name: str = "charbonnier",
              charbonnier_eps: float = 1e-3,
              alpha: float = 1.0,
              beta: float = 0.05,
              ssim_window_size: int = 11,
              ssim_sigma: float = 1.5,
              ssim_data_range: float = None):
    """工厂函数：仅支持两个选项

    - "charbonnier": 仅 CharbonnierLoss
    - "charbonnier+ssim": 组合损失 alpha * Charbonnier + beta * (1 - SSIM)

    参数
    - charbonnier_eps: Charbonnier 平滑项 epsilon
    - alpha: Charbonnier 权重（默认 1.0）
    - beta: 1-SSIM 权重（默认 0.05，建议 0.02~0.2）
    - ssim_window_size/ssim_sigma/ssim_data_range: SSIM 的窗口参数与动态范围
    """
    name = name.lower().strip()
    if name == "charbonnier":
        return CharbonnierLoss(eps=charbonnier_eps)
    if name == "charbonnier+ssim":
        return CombinedLoss(alpha=alpha,
                             beta=beta,
                             charbonnier_eps=charbonnier_eps,
                             ssim_window_size=ssim_window_size,
                             ssim_sigma=ssim_sigma,
                             ssim_data_range=ssim_data_range)
    raise ValueError("loss name 仅支持 'charbonnier' 或 'charbonnier+ssim'")
