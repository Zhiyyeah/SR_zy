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
    """基础卷积块：Conv -> BN -> ReLU + Dropout"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dropout_rate=0.1):
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
    """双卷积块 + 可选空间注意力 + 残差连接"""
    def __init__(self, in_channels, out_channels, use_attention=False, dropout_rate=0.1):
        super().__init__()
        self.conv1 = SimpleConvBlock(in_channels, out_channels, dropout_rate=dropout_rate)
        self.conv2 = SimpleConvBlock(out_channels, out_channels, dropout_rate=dropout_rate)
        
        # 残差连接
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        
        self.use_attention = use_attention
        if use_attention:
            self.attention = SpatialAttention()

    def forward(self, x):
        identity = self.residual(x)
        x = self.conv1(x)
        x = self.conv2(x)
        if self.use_attention:
            x = self.attention(x)
        # 残差连接
        x = x + identity
        return x

class DownBlock(nn.Module):
    """下采样 + 双卷积块"""
    def __init__(self, in_channels, out_channels, use_attention=False, dropout_rate=0.1):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.conv = DoubleConvBlock(in_channels, out_channels, use_attention, dropout_rate)

    def forward(self, x):
        x = self.maxpool(x)
        return self.conv(x)

class UpBlock(nn.Module):
    """上采样 + 跳跃连接 + 双卷积块"""
    def __init__(self, in_channels, skip_channels, out_channels, use_attention=False, dropout_rate=0.1):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConvBlock(in_channels // 2 + skip_channels, out_channels, use_attention, dropout_rate)

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
    def __init__(self, in_channels, scale_factor=2, activation=True, dropout_rate=0.05):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * (scale_factor ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.act = nn.ReLU(inplace=True) if activation else nn.Identity()
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.act(x)
        x = self.dropout(x)
        return x

class AttentionPixelShuffleBlock(nn.Module):
    """带空间注意力的像素重排上采样块"""
    def __init__(self, in_channels, scale_factor=2, activation=True, use_attention=True, dropout_rate=0.05):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * (scale_factor ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.act = nn.ReLU(inplace=True) if activation else nn.Identity()
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.use_attention = use_attention
        if use_attention:
            self.attention = SpatialAttention()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.act(x)
        x = self.dropout(x)
        if self.use_attention:
            x = self.attention(x)
        return x

class UNetSA(nn.Module):
    """UNet架构，仅使用空间注意力机制，添加正则化技术"""
    def __init__(self, up_scale=8, img_channel=7, width=32, use_attention=True, dropout_rate=0.1):
        super().__init__()
        self.up_scale = up_scale
        self.use_attention = use_attention
        
        # 降低初始通道数
        # 初始卷积
        self.input_conv = DoubleConvBlock(img_channel, width, use_attention=False, dropout_rate=0)
        
        # 编码器 - 逐渐增加dropout率
        self.down1 = DownBlock(width, width*2, use_attention=False, dropout_rate=dropout_rate*0.5)
        self.down2 = DownBlock(width*2, width*4, use_attention=use_attention, dropout_rate=dropout_rate)
        self.down3 = DownBlock(width*4, width*8, use_attention=use_attention, dropout_rate=dropout_rate*1.5)
        self.down4 = DownBlock(width*8, width*8, use_attention=use_attention, dropout_rate=dropout_rate*2)
        
        # 瓶颈空间注意力
        if use_attention:
            self.bottleneck_att = SpatialAttention()
        self.bottleneck_dropout = nn.Dropout2d(dropout_rate*2)
        
        # 解码器 - 逐渐减少dropout率
        self.up1 = UpBlock(width*8, width*8, width*4, use_attention=use_attention, dropout_rate=dropout_rate*1.5)
        self.up2 = UpBlock(width*4, width*4, width*2, use_attention=use_attention, dropout_rate=dropout_rate)
        self.up3 = UpBlock(width*2, width*2, width, use_attention=False, dropout_rate=dropout_rate*0.5)
        self.up4 = UpBlock(width, width, width, use_attention=False, dropout_rate=0)
        
        # 超分辨率阶段 - 较低的dropout率
        self.sr_up1 = AttentionPixelShuffleBlock(width, scale_factor=2, use_attention=use_attention, dropout_rate=dropout_rate*0.3)
        self.sr_up2 = AttentionPixelShuffleBlock(width, scale_factor=2, use_attention=use_attention, dropout_rate=dropout_rate*0.2)
        self.sr_up3 = AttentionPixelShuffleBlock(width, scale_factor=2, activation=False, use_attention=use_attention, dropout_rate=0)
        
        # 输出
        self.output_conv = nn.Conv2d(width, img_channel, kernel_size=1)
        if use_attention:
            self.final_att = SpatialAttention()
        self.padder_size = 16
        
        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """使用He初始化改善权重初始化"""
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
        
        if self.use_attention:
            x5 = self.bottleneck_att(x5)
        x5 = self.bottleneck_dropout(x5)
        
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

# 训练建议函数
def get_training_suggestions():
    """返回训练时的建议配置"""
    suggestions = {
        "optimizer": "AdamW with weight_decay=1e-4",
        "learning_rate": "1e-4 with ReduceLROnPlateau or CosineAnnealingLR",
        "data_augmentation": [
            "RandomHorizontalFlip(p=0.5)",
            "RandomVerticalFlip(p=0.5)",
            "RandomRotation(degrees=30)",
            "ColorJitter(brightness=0.2, contrast=0.2)"
        ],
        "loss_function": "L1Loss + 0.1 * PerceptualLoss (if applicable)",
        "early_stopping": "patience=10-15 epochs",
        "gradient_clipping": "max_norm=1.0",
        "batch_size": "根据GPU内存调整，建议16-32"
    }
    return suggestions

# 测试示例
if __name__ == '__main__':
    # 使用更小的width来减少模型容量
    model = UNetSA(up_scale=8, img_channel=7, width=32, use_attention=True, dropout_rate=0.1)
    x = torch.randn(1, 7, 64, 64)
    
    # 设置为训练模式查看dropout效果
    model.train()
    out = model(x)
    print('输入:', x.shape, '输出:', out.shape)
    params = sum(p.numel() for p in model.parameters())
    print('参数量:', params)
    
    # 打印训练建议
    print("\n训练建议:")
    suggestions = get_training_suggestions()
    for key, value in suggestions.items():
        print(f"{key}: {value}")