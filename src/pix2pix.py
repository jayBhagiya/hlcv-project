"""PatchGAN discriminator used by the Pix2Pix experiment."""

import torch
from torch import nn


def _block(input_channels: int, output_channels: int, stride: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(input_channels, output_channels, 4, stride, 1, bias=False),
        nn.BatchNorm2d(output_channels),
        nn.LeakyReLU(0.2, inplace=True),
    )


class PatchDiscriminator(nn.Module):
    """Classify overlapping 70x70 patches from an input/output image pair."""

    def __init__(self, base_channels: int = 64) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(6, base_channels, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            _block(base_channels, base_channels * 2, 2),
            _block(base_channels * 2, base_channels * 4, 2),
            _block(base_channels * 4, base_channels * 8, 1),
            nn.Conv2d(base_channels * 8, 1, 4, 1, 1),
        )

    def forward(self, synthetic: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        return self.layers(torch.cat((synthetic, candidate), dim=1))
