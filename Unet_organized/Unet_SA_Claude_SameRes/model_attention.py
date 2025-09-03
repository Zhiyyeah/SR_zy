import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------#
# 1. 注意力机制模块 (CBAM) - 模型的“水体适应性”核心
# ----------------------------------------------------------------#
class ChannelAttention(nn.Module):
    """通道注意力模块"""
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    """空间注意力模块"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return self.sigmoid(out)

class CBAM(nn.Module):
    """CBAM模块，结合通道和空间注意力"""
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

# ----------------------------------------------------------------#
# 2. 基础模块 (残差卷积块)
# ----------------------------------------------------------------#
class ResidualConvBlock(nn.Module):
    """
    带有残差连接和CBAM注意力的双卷积块
    """
    def __init__(self, in_channels, out_channels):
        super(ResidualConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        
        self.attention = CBAM(out_channels)
        
    def forward(self, x):
        residual = self.residual_conv(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out = self.attention(out) # 应用注意力
        
        out += residual # 添加残差连接
        return F.relu(out)

# ----------------------------------------------------------------#
# 3. U-Net 整体结构
# ----------------------------------------------------------------#
class UNetSASameRes(nn.Module):
    def __init__(self, in_channels=5, out_channels=5):
        super(UNetSASameRes, self).__init__()

        # 编码器 (下采样)
        self.encoder1 = ResidualConvBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.encoder2 = ResidualConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.encoder3 = ResidualConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.encoder4 = ResidualConvBlock(256, 512)
        self.pool4 = nn.MaxPool2d(2, 2)

        # 瓶颈层
        self.bottleneck = ResidualConvBlock(512, 1024)

        # 解码器 (上采样)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = ResidualConvBlock(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = ResidualConvBlock(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = ResidualConvBlock(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = ResidualConvBlock(128, 64)

        # 输出层
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # 编码器
        skip1 = self.encoder1(x)
        skip2 = self.encoder2(self.pool1(skip1))
        skip3 = self.encoder3(self.pool2(skip2))
        skip4 = self.encoder4(self.pool3(skip3))

        # 瓶颈层
        bottleneck = self.bottleneck(self.pool4(skip4))

        # 解码器
        d4 = self.upconv4(bottleneck)
        d4 = torch.cat((skip4, d4), dim=1)
        d4 = self.decoder4(d4)

        d3 = self.upconv3(d4)
        d3 = torch.cat((skip3, d3), dim=1)
        d3 = self.decoder3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat((skip2, d2), dim=1)
        d2 = self.decoder2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat((skip1, d1), dim=1)
        d1 = self.decoder1(d1)

        # 输出
        return self.out_conv(d1)
    
def create_model(in_channels: int = 5, out_channels: int = 5) -> nn.Module:
    return UNetSASameRes(in_channels=in_channels, out_channels=out_channels)


# ----------------------------------------------------------------#
# 4. 示例用法
# ----------------------------------------------------------------#
if __name__ == '__main__':
    # 检查是否有可用的GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 模型实例化
    # 输入5通道, 输出5通道
    model = UNetSASameRes(in_channels=5, out_channels=5).to(device)

    # 创建一个假的输入张量来测试
    # 尺寸为 (batch_size, channels, height, width)
    # 这里 batch_size=1, channels=5, height=256, width=256
    dummy_input = torch.randn(1, 5, 256, 256).to(device)
    
    # 模型前向传播
    output = model(dummy_input)

    # 打印输入和输出的尺寸以验证
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}({total_params / 1e6:.2f} M)")