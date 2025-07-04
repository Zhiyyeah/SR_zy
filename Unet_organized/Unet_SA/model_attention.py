
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialAttention(nn.Module):
    """仅保留空间注意力模块"""
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 通道维度上的平均和最大池化
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # 拼接后生成注意力图
        attention_input = torch.cat([avg_out, max_out], dim=1)
        weights = self.sigmoid(self.conv(attention_input))
        return x * weights

class SimpleConvBlock(nn.Module):
    """基础卷积块：Conv -> BN -> ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class DoubleConvBlock(nn.Module):
    """双卷积块 + 可选空间注意力"""
    def __init__(self, in_channels, out_channels, use_attention=False):
        super().__init__()
        self.double_conv = nn.Sequential(
            SimpleConvBlock(in_channels, out_channels),
            SimpleConvBlock(out_channels, out_channels)
        )
        self.use_attention = use_attention
        if use_attention:
            self.attention = SpatialAttention()

    def forward(self, x):
        x = self.double_conv(x)
        if self.use_attention:
            x = self.attention(x)
        return x

class DownBlock(nn.Module):
    """下采样 + 双卷积块"""
    def __init__(self, in_channels, out_channels, use_attention=False):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.conv = DoubleConvBlock(in_channels, out_channels, use_attention)

    def forward(self, x):
        x = self.maxpool(x)
        return self.conv(x)

class UpBlock(nn.Module):
    """上采样 + 跳跃连接 + 双卷积块"""
    def __init__(self, in_channels, skip_channels, out_channels, use_attention=False):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConvBlock(in_channels // 2 + skip_channels, out_channels, use_attention)

    def forward(self, x, skip):
        x = self.up(x)
        # pad if size mismatch
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2,
                      diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)

class PixelShuffleBlock(nn.Module):
    """像素重排上采样块"""
    def __init__(self, in_channels, scale_factor=2, activation=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * (scale_factor ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.act = nn.ReLU(inplace=True) if activation else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        return self.act(x)

class AttentionPixelShuffleBlock(nn.Module):
    """带空间注意力的像素重排上采样块"""
    def __init__(self, in_channels, scale_factor=2, activation=True, use_attention=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * (scale_factor ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.act = nn.ReLU(inplace=True) if activation else nn.Identity()
        self.use_attention = use_attention
        if use_attention:
            self.attention = SpatialAttention()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.act(x)
        if self.use_attention:
            x = self.attention(x)
        return x

class UNetSA(nn.Module):
    """UNet架构，仅使用空间注意力机制"""
    def __init__(self, up_scale=8, img_channel=7, width=64, use_attention=True):
        super().__init__()
        self.up_scale = up_scale
        self.use_attention = use_attention
        # 初始卷积
        self.input_conv = DoubleConvBlock(img_channel, width, use_attention)
        # 编码器
        self.down1 = DownBlock(width, width*2, use_attention=use_attention)
        self.down2 = DownBlock(width*2, width*4, use_attention=use_attention)
        self.down3 = DownBlock(width*4, width*8, use_attention=use_attention)
        self.down4 = DownBlock(width*8, width*8, use_attention=use_attention)
        # 瓶颈空间注意力
        if use_attention:
            self.bottleneck_att = SpatialAttention()
        # 解码器
        self.up1 = UpBlock(width*8, width*8, width*4, use_attention=use_attention)
        self.up2 = UpBlock(width*4, width*4, width*2, use_attention=use_attention)
        self.up3 = UpBlock(width*2, width*2, width, use_attention=use_attention)
        self.up4 = UpBlock(width, width, width, use_attention=use_attention)
        # 超分辨率阶段
        self.sr_up1 = AttentionPixelShuffleBlock(width, scale_factor=2, use_attention=use_attention)
        self.sr_up2 = AttentionPixelShuffleBlock(width, scale_factor=2, use_attention=use_attention)
        self.sr_up3 = AttentionPixelShuffleBlock(width, scale_factor=2, activation=False, use_attention=use_attention)
        # 输出
        self.output_conv = nn.Conv2d(width, img_channel, kernel_size=1)
        if use_attention:
            self.final_att = SpatialAttention()
        self.padder_size = 16

    def _check_image_size(self, x):
        _, _, h, w = x.size()
        mod_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_w = (self.padder_size - w % self.padder_size) % self.padder_size
        return F.pad(x, (0, mod_w, 0, mod_h))

    def forward(self, x):
        x = self._check_image_size(x)
        # 基线上采样
        up_input = F.interpolate(x, scale_factor=self.up_scale, mode='bilinear', align_corners=False)
        # 编码
        x1 = self.input_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        if self.use_attention:
            x5 = self.bottleneck_att(x5)
        # 解码
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        # 超分辨率
        x = self.sr_up1(x)
        x = self.sr_up2(x)
        x = self.sr_up3(x)
        x = self.output_conv(x)
        if self.use_attention:
            x = self.final_att(x)
        # 残差连接
        return x + up_input

# 测试示例
if __name__ == '__main__':
    model = UNetSA(up_scale=8, img_channel=7, width=64, use_attention=False)
    x = torch.randn(1, 7, 64, 64)
    out = model(x)
    print('输入:', x.shape, '输出:', out.shape)
    params = sum(p.numel() for p in model.parameters())
    print('参数量:', params)
