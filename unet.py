"""
U-Net Architecture for Aerial Image Segmentation
=================================================
Reference: Ronneberger et al. (2015) - "U-Net: Convolutional Networks for
Biomedical Image Segmentation" (https://arxiv.org/abs/1505.04597)

Adapted for multi-class semantic segmentation of aerial/satellite imagery.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ---------------------------------------------------------------------------
# Basic Building Blocks
# ---------------------------------------------------------------------------

class DoubleConv(nn.Module):
    """
    Two consecutive (Conv2d → BatchNorm → ReLU) blocks.
    This is the core repeating unit of U-Net.
    """

    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None, dropout: float = 0.0):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels

        layers = [
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(p=dropout))
        layers += [
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EncoderBlock(nn.Module):
    """Downsampling block: MaxPool2d → DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels, dropout=dropout)

    def forward(self, x: torch.Tensor):
        x = self.pool(x)
        x = self.conv(x)
        return x


class DecoderBlock(nn.Module):
    """
    Upsampling block: Upsample / ConvTranspose2d → concat skip → DoubleConv.
    Supports both bilinear upsampling and transposed convolution.
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int,
                 bilinear: bool = True, dropout: float = 0.0):
        super().__init__()
        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(in_channels, in_channels // 2, kernel_size=1),
            )
            conv_in = in_channels // 2 + skip_channels
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            conv_in = in_channels // 2 + skip_channels

        self.conv = DoubleConv(conv_in, out_channels, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)

        # Handle size mismatch (input might not be perfectly divisible by 2)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


# ---------------------------------------------------------------------------
# U-Net
# ---------------------------------------------------------------------------

class UNet(nn.Module):
    """
    Full U-Net for semantic segmentation.

    Args:
        in_channels  (int): Number of input image channels (3 for RGB).
        num_classes  (int): Number of output segmentation classes.
        features     (list[int]): Channel sizes for encoder stages.
        bilinear     (bool): Use bilinear upsampling (True) or ConvTranspose2d (False).
        dropout      (float): Dropout probability applied in DoubleConv blocks.

    Input shape:  (B, in_channels, H, W)
    Output shape: (B, num_classes, H, W)   — raw logits
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 6,
        features: list = [64, 128, 256, 512],
        bilinear: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        # --- Encoder ---
        self.encoder_input = DoubleConv(in_channels, features[0], dropout=dropout)
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(features[i], features[i + 1], dropout=dropout)
            for i in range(len(features) - 1)
        ])

        # --- Bottleneck ---
        bottleneck_channels = features[-1] * 2
        self.bottleneck = EncoderBlock(features[-1], bottleneck_channels, dropout=dropout)

        # --- Decoder ---
        decoder_in = [bottleneck_channels] + [features[i] * 2 for i in range(len(features) - 1, 0, -1)]
        decoder_skip = features[::-1]  # skip connections in reverse
        decoder_out = features[::-1]

        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(decoder_in[i], decoder_skip[i], decoder_out[i],
                         bilinear=bilinear, dropout=dropout)
            for i in range(len(features))
        ])

        # --- Output head ---
        self.output_conv = nn.Conv2d(features[0], num_classes, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        skip_connections = []
        x = self.encoder_input(x)
        skip_connections.append(x)

        for enc in self.encoder_blocks:
            x = enc(x)
            skip_connections.append(x)

        # Bottleneck (no skip)
        x = self.bottleneck(x)

        # Decoder
        for i, dec in enumerate(self.decoder_blocks):
            skip = skip_connections[-(i + 1)]
            x = dec(x, skip)

        return self.output_conv(x)

    def get_num_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


# ---------------------------------------------------------------------------
# Pretrained Encoder variant (ResNet backbone)
# ---------------------------------------------------------------------------

class ResNetEncoder(nn.Module):
    """ResNet-34 encoder with U-Net-compatible skip connections."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        backbone = models.resnet34(weights="DEFAULT" if pretrained else None)

        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)  # /2
        self.pool = backbone.maxpool                                                # /4
        self.layer1 = backbone.layer1   # 64  ch  /4
        self.layer2 = backbone.layer2   # 128 ch  /8
        self.layer3 = backbone.layer3   # 256 ch  /16
        self.layer4 = backbone.layer4   # 512 ch  /32

    def forward(self, x):
        s0 = self.layer0(x)          # 64,  H/2
        s1 = self.layer1(self.pool(s0))   # 64,  H/4
        s2 = self.layer2(s1)         # 128, H/8
        s3 = self.layer3(s2)         # 256, H/16
        s4 = self.layer4(s3)         # 512, H/32
        return [s0, s1, s2, s3, s4]


class UNetResNet(nn.Module):
    """
    U-Net with a pretrained ResNet-34 encoder backbone.
    Better feature extraction vs. vanilla U-Net, especially with limited data.
    """

    def __init__(self, num_classes: int = 6, pretrained: bool = True, dropout: float = 0.1):
        super().__init__()
        self.encoder = ResNetEncoder(pretrained=pretrained)

        # Decoder channels: [512→256, 256→128, 128→64, 64→64, 64→32]
        self.dec4 = DecoderBlock(512, 256, 256, dropout=dropout)
        self.dec3 = DecoderBlock(256, 128, 128, dropout=dropout)
        self.dec2 = DecoderBlock(128, 64,  64,  dropout=dropout)
        self.dec1 = DecoderBlock(64,  64,  64,  dropout=dropout)
        self.dec0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            DoubleConv(64, 32, dropout=dropout),
        )
        self.output_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        s0, s1, s2, s3, s4 = self.encoder(x)

        x = self.dec4(s4, s3)
        x = self.dec3(x,  s2)
        x = self.dec2(x,  s1)
        x = self.dec1(x,  s0)
        x = self.dec0(x)
        return self.output_conv(x)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(config: dict) -> nn.Module:
    """
    Build a model from a config dictionary.

    Example config:
        {
          "architecture": "unet",          # "unet" | "unet_resnet"
          "in_channels": 3,
          "num_classes": 6,
          "features": [64, 128, 256, 512],
          "bilinear": true,
          "dropout": 0.1,
          "pretrained": true               # only for unet_resnet
        }
    """
    arch = config.get("architecture", "unet").lower()

    if arch == "unet":
        return UNet(
            in_channels=config.get("in_channels", 3),
            num_classes=config.get("num_classes", 6),
            features=config.get("features", [64, 128, 256, 512]),
            bilinear=config.get("bilinear", True),
            dropout=config.get("dropout", 0.1),
        )
    elif arch == "unet_resnet":
        return UNetResNet(
            num_classes=config.get("num_classes", 6),
            pretrained=config.get("pretrained", True),
            dropout=config.get("dropout", 0.1),
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}. Choose 'unet' or 'unet_resnet'.")


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("Testing Vanilla U-Net")
    print("=" * 60)
    model = UNet(in_channels=3, num_classes=6).to(device)
    x = torch.randn(2, 3, 512, 512).to(device)
    out = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
    params = model.get_num_parameters()
    print(f"Total params:     {params['total']:,}")
    print(f"Trainable params: {params['trainable']:,}")

    print("\n" + "=" * 60)
    print("Testing U-Net + ResNet-34 backbone")
    print("=" * 60)
    model2 = UNetResNet(num_classes=6, pretrained=False).to(device)
    out2 = model2(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out2.shape}")
