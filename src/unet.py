"""Small U-Net baseline for L1 image translation."""

import torch
from torch import nn


def _block(input_channels: int, output_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(input_channels, output_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(output_channels, output_channels, 3, padding=1),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, base_channels: int = 16) -> None:
        super().__init__()
        self.encoder1 = _block(3, base_channels)
        self.encoder2 = _block(base_channels, base_channels * 2)
        self.encoder3 = _block(base_channels * 2, base_channels * 4)
        self.encoder4 = _block(base_channels * 4, base_channels * 8)
        self.bridge = _block(base_channels * 8, base_channels * 16)
        self.pool = nn.MaxPool2d(2)
        self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, 2, stride=2)
        self.decoder4 = _block(base_channels * 16, base_channels * 8)
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.decoder3 = _block(base_channels * 8, base_channels * 4)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.decoder2 = _block(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.decoder1 = _block(base_channels * 2, base_channels)
        self.output = nn.Conv2d(base_channels, 3, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.shape[-2] % 16 or image.shape[-1] % 16:
            raise ValueError("Image height and width must be divisible by 16")
        encoded1 = self.encoder1(image)
        encoded2 = self.encoder2(self.pool(encoded1))
        encoded3 = self.encoder3(self.pool(encoded2))
        encoded4 = self.encoder4(self.pool(encoded3))
        bridge = self.bridge(self.pool(encoded4))
        decoded4 = self.decoder4(torch.cat((self.up4(bridge), encoded4), dim=1))
        decoded3 = self.decoder3(torch.cat((self.up3(decoded4), encoded3), dim=1))
        decoded2 = self.decoder2(torch.cat((self.up2(decoded3), encoded2), dim=1))
        decoded1 = self.decoder1(torch.cat((self.up1(decoded2), encoded1), dim=1))
        return torch.sigmoid(self.output(decoded1))
