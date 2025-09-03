import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# 基础模块
# -----------------------------
class SEBlock(nn.Module):
    """Squeeze-and-Excitation 通道注意力"""
    def __init__(self, ch: int, reduction: int = 8):
        super().__init__()
        mid = max(1, ch // reduction)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(ch, mid, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, ch, 1, bias=True),
            nn.Sigmoid()
        )
    def forward(self, x):
        w = self.fc(self.avg(x))
        return x * w


class SpatialAttention(nn.Module):
    """CBAM 风格的空间注意力（与你原模型一致）"""
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        p = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=p, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.cat([avg_out, max_out], dim=1)
        attn = self.sigmoid(self.conv(attn))
        return x * attn


def _best_gn_groups(ch: int, preferred: int = 8) -> int:
    g = min(preferred, ch)
    while g > 1 and (ch % g != 0):
        g -= 1
    return max(1, g)


class DSConv7x7(nn.Module):
    """Depthwise-Separable 7x7（先DW再PW）用于平滑低频"""
    def __init__(self, ch: int):
        super().__init__()
    def forward(self, x):
        return x


# -----------------------------
# 残差块（更稳的 dilation，加入 SE ）
# -----------------------------
class SASEBlock(nn.Module):
    """
    SpatialAttention + (Conv3x3(dilation)->GN->ReLU->Conv3x3(dilation)->GN) + 残差
    """
    def __init__(self, ch: int, dilation: int = 1, groups: int = 8, dropout: float = 0.0):
        super().__init__()
        g = _best_gn_groups(ch, groups)
        pad = dilation
        self.sa = SpatialAttention()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=pad, dilation=dilation, bias=False)
        self.gn1 = nn.GroupNorm(g, ch)
        self.act1 = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=pad, dilation=dilation, bias=False)
        self.gn2 = nn.GroupNorm(g, ch)
        self.acto = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        x = self.sa(x)
        x = self.conv1(x)
        x = self.gn1(x)
        x = self.act1(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = self.gn2(x)
        return self.acto(x + identity)


# -----------------------------
# 低频金字塔（多尺度平均池化 + 上采样 + 平滑）
# -----------------------------
class LowFreqPyramid(nn.Module):
    """
    多尺度 AvgPool -> Conv3x3 -> ReLU -> 上采样回原尺寸，聚合后用 7x7 深度可分离卷积平滑
    """
    def __init__(self, ch: int, scales=(2, 4, 8)):
        super().__init__()
        self.scales = scales
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, ch, 3, padding=1, bias=True),
                nn.ReLU(inplace=True)
            ) for _ in scales
        ])
        

    def forward(self, x):
        H, W = x.shape[-2], x.shape[-1]
        outs = []
        for i, s in enumerate(self.scales):
            # 平均池化获取低频
            pooled = F.avg_pool2d(x, kernel_size=s, stride=s, ceil_mode=True)
            y = self.branches[i](pooled)
            y = F.interpolate(y, size=(H, W), mode='bilinear', align_corners=False)
            outs.append(y)
        out = torch.stack(outs, dim=0).mean(dim=0)  # 多尺度聚合
        
        return out


# -----------------------------
# 门控融合（含低频下限约束）
# -----------------------------
class GatedFusion(nn.Module):
    """
    输入 LF/HF 两个特征，输出融合结果。
    先生成逐像素 gate \in (0,1)，再做：
        g' = lf_floor + (1 - lf_floor) * gate
        out = g' * LF + (1 - g') * HF
    """
    def __init__(self, ch: int, lf_floor: float = 0.6):
        super().__init__()
        assert 0.0 <= lf_floor < 1.0
        self.lf_floor = lf_floor
        self.fuse = nn.Sequential(
            nn.Conv2d(2 * ch, 1, 1, bias=True),
            nn.Sigmoid()
        )
    def forward(self, lf, hf, mask: torch.Tensor = None):
        g = self.fuse(torch.cat([lf, hf], dim=1))  # (N,1,H,W)
        if mask is not None:
            # 可选的水体掩膜：只在水体区域使用门控；陆地处更偏 LF/平滑
            g = g * mask
        # 简化：不再使用 lf_floor 约束，直接用 gate 融合
        return g * lf + (1.0 - g) * hf


# -----------------------------
# 主干网络
# -----------------------------
class UNetSASameRes(nn.Module):
    """
    水体低频友好型 SR 网络：
        Head -> {HF 干路堆叠} & {LF 金字塔} -> 门控融合 -> Tail -> (可选残差输出)
    支持 scale=1（同分辨率细化）或 >1（像素上采样 + 细化）
    """
    def __init__(
        self,
        in_channels: int = 5,
        out_channels: int = 5,
        base_ch: int = 64,
        n_blocks: int = 8,
        dilations: tuple = (1, 2, 1, 2, 1, 2, 1, 2),
        groups: int = 8,
        dropout: float = 0.0,
        residual: bool = True,
        lf_scales=(2, 4, 8),
        lf_floor: float = 0.6,
        scale: int = 1
    ):
        super().__init__()
        assert scale in (1, 2, 4), "scale 仅支持 1/2/4（可按需扩展）"
        self.scale = scale
        self.residual = residual

        # Head: 按需上采样（PixelShuffle），再映射到 base_ch
        head_layers = []
        if scale > 1:
            # 先把输入升到中间通道，再 PixelShuffle 上采样
            head_layers += [
                nn.Conv2d(in_channels, base_ch * (scale ** 2), 3, padding=1, bias=True),
                nn.PixelShuffle(upscale_factor=scale),
                nn.Conv2d(base_ch, base_ch, 3, padding=1, bias=True),
                nn.ReLU(inplace=True),
            ]
            self.res_up = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * (scale ** 2), 3, padding=1, bias=True),
                nn.PixelShuffle(upscale_factor=scale)
            )
        else:
            head_layers += [nn.Conv2d(in_channels, base_ch, 3, padding=1, bias=True),
                            nn.ReLU(inplace=True)]
            self.res_up = nn.Conv2d(in_channels, out_channels, 1, bias=True)

        self.head = nn.Sequential(
            *head_layers,
            nn.GroupNorm(_best_gn_groups(base_ch, groups), base_ch)
        )

        # HF 干路：SASE 残差块堆叠（温和的 dilation）
        blocks = []
        for i in range(n_blocks):
            d = dilations[i % len(dilations)]
            blocks.append(SASEBlock(base_ch, dilation=d, groups=groups, dropout=dropout))
        self.hf_trunk = nn.Sequential(*blocks)

        # LF 分支与门控融合
        self.lf_branch = LowFreqPyramid(base_ch, scales=lf_scales)
        self.fuser = GatedFusion(base_ch, lf_floor=lf_floor)

        # Tail 输出
        self.tail = nn.Conv2d(base_ch, out_channels, 1, bias=True)

        # 是否做输出残差
        self.use_residual_out = residual

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        x: (N, C_in, H, W)
        mask: 可选 (N, 1, H', W')，若 scale>1 需与输出同尺度；用于在水体区域更依赖门控。
        """
        feat = self.head(x)           # -> (N, base_ch, H*scale, W*scale)
        hf = self.hf_trunk(feat)      # 细节/中频
        lf = self.lf_branch(feat)     # 低频/大尺度

        fused = self.fuser(lf, hf, mask)  # 频域偏置式融合

        out = self.tail(fused)        # 通道回归
        if self.use_residual_out:
            out = out + self.res_up(x)  # 与输入（或其上采样）做残差对齐
        return out


# -----------------------------
# 便捷创建与自检
# -----------------------------
def create_model(in_channels: int = 5, out_channels: int = 5, scale: int = 1) -> nn.Module:
    return UNetSASameRes(in_channels=in_channels, out_channels=out_channels, scale=scale)

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

if __name__ == "__main__":
    IN_CH, OUT_CH = 5, 5

    # 1) 同分辨率细化
    model1 = create_model(IN_CH, OUT_CH, scale=1)
    x1 = torch.randn(1, IN_CH, 256, 256)
    with torch.no_grad():
        y1 = model1(x1)
    print("Same-res:", tuple(x1.shape), "->", tuple(y1.shape), "Params:", f"{count_parameters(model1):,}")

    # 2) 2x 上采样 + 细化
    model2 = create_model(IN_CH, OUT_CH, scale=2)
    x2 = torch.randn(1, IN_CH, 128, 128)
    with torch.no_grad():
        y2 = model2(x2)
    print("Scale=2:", tuple(x2.shape), "->", tuple(y2.shape), "Params:", f"{count_parameters(model2):,}")
