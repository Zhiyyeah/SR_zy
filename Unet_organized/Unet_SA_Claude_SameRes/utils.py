import os
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(model: torch.nn.Module, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint(model: torch.nn.Module, path: str, map_location: Optional[str] = None):
    sd = torch.load(path, map_location=map_location)
    model.load_state_dict(sd, strict=True)


@dataclass
class AverageMeter:
    name: str
    val: float = 0.0
    sum: float = 0.0
    count: int = 0

    def update(self, v: float, n: int = 1):
        self.val = v
        self.sum += v * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / max(1, self.count)

