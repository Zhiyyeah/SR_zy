import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialAttention(nn.Module):
    """空间注意力模块"""
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention_input = torch.cat([avg_out, max_out], dim=1)
        weights = self.sigmoid(self.conv(attention_input))
        return x * weights

class SimpleConvBlock(nn.Module):
    """基础卷积块：Conv -> BN -> ReLU -> Dropout"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dropout_rate=0.0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class DoubleConvBlock(nn.Module):
    """双卷积块 + 可选空间注意力 + Dropout"""
    def __init__(self, in_channels, out_channels, use_attention=False, dropout_rate=0.0):
        super().__init__()
        mid_channels = out_channels // 2 if out_channels > in_channels else out_channels
        
        self.conv1 = SimpleConvBlock(in_channels, mid_channels, dropout_rate=dropout_rate)
        self.conv2 = SimpleConvBlock(mid_channels, out_channels, dropout_rate=dropout_rate)
        
        self.use_attention = use_attention
        if use_attention:
            self.attention = SpatialAttention()
        
        # 残差连接
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.residual(x)
        x = self.conv1(x)
        x = self.conv2(x)
        if self.use_attention:
            x = self.attention(x)
        return x + residual  # 添加残差连接

class DownBlock(nn.Module):
    """下采样 + 双卷积块"""
    def __init__(self, in_channels, out_channels, use_attention=False, dropout_rate=0.0):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.conv = DoubleConvBlock(in_channels, out_channels, use_attention, dropout_rate)

    def forward(self, x):
        x = self.maxpool(x)
        return self.conv(x)

class UpBlock(nn.Module):
    """上采样 + 跳跃连接 + 双卷积块"""
    def __init__(self, in_channels, skip_channels, out_channels, use_attention=False, dropout_rate=0.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConvBlock(in_channels // 2 + skip_channels, out_channels, use_attention, dropout_rate)

    def forward(self, x, skip):
        x = self.up(x)
        # 处理尺寸不匹配
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2,
                      diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)

class AttentionPixelShuffleBlock(nn.Module):
    """带空间注意力的像素重排上采样块"""
    def __init__(self, in_channels, scale_factor=2, activation=True, use_attention=True, dropout_rate=0.0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * (scale_factor ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.bn = nn.BatchNorm2d(in_channels)  # 添加BN层
        self.act = nn.ReLU(inplace=True) if activation else nn.Identity()
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.use_attention = use_attention
        if use_attention:
            self.attention = SpatialAttention()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.bn(x)  # BN在pixel shuffle之后
        x = self.act(x)
        x = self.dropout(x)
        if self.use_attention:
            x = self.attention(x)
        return x

class UNetSAImproved(nn.Module):
    """改进的UNet架构，增加了Dropout和更好的正则化"""
    def __init__(self, up_scale=8, img_channel=7, width=64, use_attention=True, 
                 dropout_rate=0.1, deep_supervision=True):
        super().__init__()
        self.up_scale = up_scale
        self.use_attention = use_attention
        self.deep_supervision = deep_supervision
        
        # Dropout率递进策略：浅层低，深层高
        dr1, dr2, dr3, dr4 = 0.0, dropout_rate * 0.5, dropout_rate * 0.75, dropout_rate
        
        # 初始卷积
        self.input_conv = DoubleConvBlock(img_channel, width, use_attention, dropout_rate=dr1)
        
        # 编码器（逐渐增加dropout率）
        self.down1 = DownBlock(width, width*2, use_attention=False, dropout_rate=dr1)
        self.down2 = DownBlock(width*2, width*4, use_attention=use_attention, dropout_rate=dr2)
        self.down3 = DownBlock(width*4, width*8, use_attention=use_attention, dropout_rate=dr3)
        self.down4 = DownBlock(width*8, width*8, use_attention=use_attention, dropout_rate=dr4)
        
        # 瓶颈层
        self.bottleneck = DoubleConvBlock(width*8, width*8, use_attention=True, dropout_rate=dr4)
        
        # 解码器（逐渐减少dropout率）
        self.up1 = UpBlock(width*8, width*8, width*4, use_attention=use_attention, dropout_rate=dr3)
        self.up2 = UpBlock(width*4, width*4, width*2, use_attention=use_attention, dropout_rate=dr2)
        self.up3 = UpBlock(width*2, width*2, width, use_attention=False, dropout_rate=dr1)
        self.up4 = UpBlock(width, width, width, use_attention=False, dropout_rate=0.0)
        
        # 超分辨率阶段（渐进式上采样）
        self.sr_up1 = AttentionPixelShuffleBlock(width, scale_factor=2, use_attention=use_attention, dropout_rate=dr1)
        self.sr_up2 = AttentionPixelShuffleBlock(width, scale_factor=2, use_attention=use_attention, dropout_rate=0.0)
        self.sr_up3 = AttentionPixelShuffleBlock(width, scale_factor=2, activation=False, use_attention=use_attention, dropout_rate=0.0)
        
        # 输出卷积
        self.output_conv = nn.Conv2d(width, img_channel, kernel_size=3, padding=1)
        
        # 深度监督输出（可选）
        if self.deep_supervision:
            self.aux_output = nn.Conv2d(width, img_channel, kernel_size=1)
            self.aux_upsample = nn.Upsample(scale_factor=up_scale, mode='bilinear', align_corners=False)
        
        self.padder_size = 16
        
        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """使用He初始化改进权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

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
        
        # 瓶颈处理
        x5 = self.bottleneck(x5)
        
        # 解码
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # 深度监督（可选）
        aux_output = None
        if self.deep_supervision and self.training:
            aux_output = self.aux_upsample(self.aux_output(x))
        
        # 超分辨率
        x = self.sr_up1(x)
        x = self.sr_up2(x)
        x = self.sr_up3(x)
        x = self.output_conv(x)
        
        # 残差连接
        output = x + up_input
        
        if self.deep_supervision and self.training:
            return output, aux_output
        else:
            return output

# 测试
if __name__ == '__main__':
    model = UNetSAImproved(up_scale=8, img_channel=7, width=64, use_attention=True, dropout_rate=0.15)
    x = torch.randn(1, 7, 64, 64)
    
    # 训练模式
    model.train()
    out = model(x)
    if isinstance(out, tuple):
        print('训练模式 - 主输出:', out[0].shape, '辅助输出:', out[1].shape)
    else:
        print('训练模式 - 输出:', out.shape)
    
    # 评估模式
    model.eval()
    out = model(x)
    print('评估模式 - 输出:', out.shape)
    
    params = sum(p.numel() for p in model.parameters())
    print('参数量:', params)