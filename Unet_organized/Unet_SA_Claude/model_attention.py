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
        
        # 残差连接 - 修复：只有在通道数相同时才使用残差连接
        self.use_residual = in_channels == out_channels
        if not self.use_residual:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
        self.use_attention = use_attention
        if use_attention:
            self.attention = SpatialAttention()

    def forward(self, x):
        identity = x
        if not self.use_residual:
            identity = self.residual(x)
        
        x = self.conv1(x)
        x = self.conv2(x)
        
        if self.use_attention:
            x = self.attention(x)
        
        # 残差连接 - 使用更小的权重避免梯度爆炸
        x = x + 0.1 * identity
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
        # 处理尺寸不匹配
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

class UNetSA(nn.Module):
    """修复后的UNet超分辨率模型"""
    def __init__(self, up_scale=8, img_channel=7, width=32, use_attention=True, dropout_rate=0.05):
        super().__init__()
        self.up_scale = up_scale
        self.use_attention = use_attention
        
        # 初始卷积 - 减少dropout
        self.input_conv = DoubleConvBlock(img_channel, width, use_attention=False, dropout_rate=0)
        
        # 编码器 - 减少dropout率
        self.down1 = DownBlock(width, width*2, use_attention=False, dropout_rate=dropout_rate*0.5)
        self.down2 = DownBlock(width*2, width*4, use_attention=use_attention, dropout_rate=dropout_rate)
        self.down3 = DownBlock(width*4, width*8, use_attention=use_attention, dropout_rate=dropout_rate)
        self.down4 = DownBlock(width*8, width*8, use_attention=use_attention, dropout_rate=dropout_rate)
        
        # 瓶颈 - 减少dropout
        if use_attention:
            self.bottleneck_att = SpatialAttention()
        self.bottleneck_dropout = nn.Dropout2d(dropout_rate)
        
        # 解码器 - 减少dropout率
        self.up1 = UpBlock(width*8, width*8, width*4, use_attention=use_attention, dropout_rate=dropout_rate)
        self.up2 = UpBlock(width*4, width*4, width*2, use_attention=use_attention, dropout_rate=dropout_rate*0.5)
        self.up3 = UpBlock(width*2, width*2, width, use_attention=False, dropout_rate=dropout_rate*0.5)
        self.up4 = UpBlock(width, width, width, use_attention=False, dropout_rate=0)
        
        # 超分辨率阶段 - 减少dropout
        self.sr_conv = nn.Conv2d(width, width, kernel_size=3, padding=1)
        self.sr_bn = nn.BatchNorm2d(width)  # 添加BN层稳定训练
        self.sr_up1 = PixelShuffleBlock(width, scale_factor=2, dropout_rate=0)
        self.sr_up2 = PixelShuffleBlock(width, scale_factor=2, dropout_rate=0)
        self.sr_up3 = PixelShuffleBlock(width, scale_factor=2, activation=False, dropout_rate=0)
        
        # 输出层
        self.output_conv = nn.Conv2d(width, img_channel, kernel_size=3, padding=1)
        
        # 修复：移除可学习的残差缩放因子，使用固定的小权重
        # self.residual_scale = nn.Parameter(torch.tensor(0.1))  # 删除这行
        
        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """更保守的权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 使用更小的初始化避免梯度爆炸
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, 0, 0.02)  # 更小的初始化
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, 0, 0.02)  # 更小的初始化
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def check_and_pad_input(self, x):
        """检查并填充输入以确保能被16整除"""
        _, _, h, w = x.size()
        pad_h = (16 - h % 16) % 16
        pad_w = (16 - w % 16) % 16
        
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        
        return x, (h, w)

    def forward(self, x):
        # 记录原始尺寸并填充
        x_padded, (orig_h, orig_w) = self.check_and_pad_input(x)
        
        # 双三次插值上采样作为基准
        bicubic_up = F.interpolate(x_padded, scale_factor=self.up_scale, mode='bicubic', align_corners=False)
        
        # 编码路径
        x1 = self.input_conv(x_padded)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # 瓶颈处理
        if self.use_attention:
            x5 = self.bottleneck_att(x5)
        x5 = self.bottleneck_dropout(x5)
        
        # 解码路径
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # 超分辨率路径
        x = self.sr_conv(x)
        x = self.sr_bn(x)  # 添加BN层
        x = F.relu(x)      # 添加激活函数
        x = self.sr_up1(x)
        x = self.sr_up2(x)
        x = self.sr_up3(x)
        
        # 输出卷积
        residual = self.output_conv(x)
        
        # 修复：使用固定的小权重进行残差连接，避免训练不稳定
        output = bicubic_up + 0.1 * residual
        
        # 裁剪回原始尺寸
        output = output[:, :, :orig_h * self.up_scale, :orig_w * self.up_scale]
        
        return output

    def get_num_params(self):
        """获取模型参数数量"""
        return sum(p.numel() for p in self.parameters())


# 测试代码
if __name__ == '__main__':
    # 创建模型
    model = UNetSA(up_scale=8, img_channel=7, width=32, use_attention=True, dropout_rate=0.05)
    
    # 测试不同尺寸的输入
    test_sizes = [(64, 64), (128, 128), (100, 150)]
    
    for h, w in test_sizes:
        x = torch.randn(1, 7, h, w)
        model.eval()  # 评估模式
        with torch.no_grad():
            out = model(x)
        print(f'输入: {x.shape} -> 输出: {out.shape}')
        print(f'期望输出: torch.Size([1, 7, {h*8}, {w*8}])')
        print(f'输出范围: [{out.min().item():.3f}, {out.max().item():.3f}]')
        print('-' * 50)
    
    print(f'模型参数量: {model.get_num_params():,}')
    
    # 检查梯度流
    print('\n检查梯度流:')
    x = torch.randn(1, 7, 64, 64, requires_grad=True)
    model.train()
    out = model(x)
    loss = out.mean()
    loss.backward()
    print(f'输入梯度范数: {x.grad.norm().item():.4f}')