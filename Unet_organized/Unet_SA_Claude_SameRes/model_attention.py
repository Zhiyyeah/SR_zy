import torch
import torch.nn as nn


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.cat([avg_out, max_out], dim=1)
        attn = self.conv(attn)
        attn = self.sigmoid(attn)
        return x * attn


def _best_gn_groups(ch: int, preferred: int) -> int:
    g = min(preferred, ch)
    while g > 1 and (ch % g != 0):
        g -= 1
    return max(1, g)


class SAResBlock(nn.Module):
    def __init__(self, ch: int, dilation: int = 1, groups: int = 8, dropout: float = 0.0):
        super().__init__()
        g = _best_gn_groups(ch, groups)
        pad = dilation
        self.sa = SpatialAttention()
        self.conv1 = nn.Conv2d(ch, ch, kernel_size=3, padding=pad, dilation=dilation, bias=False)
        self.gn1 = nn.GroupNorm(g, ch)
        self.act1 = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=3, padding=pad, dilation=dilation, bias=False)
        self.gn2 = nn.GroupNorm(g, ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.sa(x)
        x = self.conv1(x)
        x = self.gn1(x)
        x = self.act1(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = self.gn2(x)
        return torch.relu_(x + identity)


class UNetSASameRes(nn.Module):
    """Same-resolution SR model with Spatial Attention and residual learning."""

    def __init__(
        self,
        in_channels: int = 5,
        out_channels: int = 5,
        base_ch: int = 64,
        groups: int = 8,
        dropout: float = 0.0,
        residual: bool = True,
        n_blocks: int = 8,
        dilations: tuple = (1, 2, 4, 1, 2, 4, 1, 2),
    ):
        super().__init__()
        self.residual = residual
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, base_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_best_gn_groups(base_ch, groups), base_ch),
            nn.ReLU(inplace=True),
        )
        blocks = []
        for i in range(n_blocks):
            d = dilations[i % len(dilations)] if len(dilations) > 0 else 1
            blocks.append(SAResBlock(base_ch, dilation=d, groups=groups, dropout=dropout))
        self.blocks = nn.Sequential(*blocks)
        self.tail = nn.Conv2d(base_ch, out_channels, kernel_size=1)
        self.res_align = nn.Conv2d(in_channels, out_channels, kernel_size=1) if residual else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.head(x)
        y = self.blocks(y)
        y = self.tail(y)
        if self.residual:
            y = y + self.res_align(x)
        return y


def create_model(in_channels: int = 5, out_channels: int = 5) -> nn.Module:
    return UNetSASameRes(in_channels=in_channels, out_channels=out_channels)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    # Minimal self-check
    IN_CH = 5
    OUT_CH = 5
    H = 256
    W = 256
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetSASameRes(in_channels=IN_CH, out_channels=OUT_CH).to(device)
    x = torch.randn(1, IN_CH, H, W, device=device)
    with torch.no_grad():
        y = model(x)
    print(f"Input: {tuple(x.shape)}  Output: {tuple(y.shape)}  Params: {count_parameters(model):,}")

