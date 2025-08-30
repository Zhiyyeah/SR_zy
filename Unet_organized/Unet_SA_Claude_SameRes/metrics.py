import math
import torch


def psnr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    """PSNR without rescaling/clipping.

    - Uses data range from target: peak = (target.max - target.min).
    - If mse ~ 0, returns 100 dB.
    """
    # no clamp, preserve radiance domain
    mse = torch.mean((pred - target) ** 2).item()
    if mse <= eps:
        return 100.0
    t_min = target.amin().item()
    t_max = target.amax().item()
    peak = max(t_max - t_min, eps)
    return 20.0 * math.log10(peak) - 10.0 * math.log10(mse)


@torch.no_grad()
def batch_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    return psnr(pred, target)
