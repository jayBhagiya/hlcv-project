"""U-Net generator with global self-attention at its bottleneck."""

import torch
from torch import nn

from src.unet import UNet


class TransformerBottleneck(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.position = nn.Conv2d(
            channels, channels, 3, padding=1, groups=channels
        )
        layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=4,
            dim_feedforward=channels * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=2,
            norm=nn.LayerNorm(channels),
            enable_nested_tensor=False,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = features + self.position(features)
        batch, channels, height, width = features.shape
        tokens = features.flatten(2).transpose(1, 2)
        tokens = self.encoder(tokens)
        return tokens.transpose(1, 2).reshape(batch, channels, height, width)


class TransformerUNet(UNet):
    """Baseline U-Net plus a two-layer transformer bottleneck."""

    def __init__(self, base_channels: int = 12) -> None:
        super().__init__(base_channels)
        convolutional_bridge = self.bridge
        self.bridge = nn.Sequential(
            convolutional_bridge,
            TransformerBottleneck(base_channels * 16),
        )
