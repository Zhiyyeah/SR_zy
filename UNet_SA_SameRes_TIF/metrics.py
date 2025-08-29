import math
import torch


def psnr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    """Compute PSNR for tensors in [0,1]. Supports shape (N,C,H,W)."""
    pred = torch.clamp(pred, 0.0, 1.0)
    target = torch.clamp(target, 0.0, 1.0)
    mse = torch.mean((pred - target) ** 2).item()
    if mse <= eps:
        return 100.0
    return 20.0 * math.log10(1.0 / math.sqrt(mse))


@torch.no_grad()
def batch_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    return psnr(pred, target)

