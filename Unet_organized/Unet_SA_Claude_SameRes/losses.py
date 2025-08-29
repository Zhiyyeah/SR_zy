import torch
import torch.nn as nn


class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps ** 2))


def make_loss(name: str = "charbonnier", charbonnier_eps: float = 1e-3):
    name = name.lower()
    if name == "l1":
        return nn.L1Loss()
    if name in ("l2", "mse"):
        return nn.MSELoss()
    if name == "charbonnier":
        return CharbonnierLoss(eps=charbonnier_eps)
    raise ValueError(f"Unknown loss: {name}")

