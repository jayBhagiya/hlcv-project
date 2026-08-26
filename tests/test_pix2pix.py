import math
import unittest

import torch

from src.pix2pix import PatchDiscriminator
from src.train_pix2pix import train_epoch
from src.unet import UNet


class Pix2PixTest(unittest.TestCase):
    def test_patchgan_and_training_step(self) -> None:
        torch.manual_seed(0)
        synthetic = torch.rand(2, 3, 32, 48)
        real = torch.rand(2, 3, 32, 48)
        generator = UNet(base_channels=2)
        discriminator = PatchDiscriminator(base_channels=4)
        self.assertEqual(
            discriminator(synthetic, real).shape,
            (2, 1, 2, 4),
        )

        generator_before = next(generator.parameters()).detach().clone()
        discriminator_before = next(discriminator.parameters()).detach().clone()
        metrics = train_epoch(
            generator,
            discriminator,
            [(synthetic, real, ("first", "second"))],
            torch.device("cpu"),
            torch.optim.Adam(generator.parameters(), lr=2e-4),
            torch.optim.Adam(discriminator.parameters(), lr=2e-4),
        )
        self.assertTrue(all(math.isfinite(value) for value in metrics.values()))
        self.assertFalse(torch.equal(generator_before, next(generator.parameters())))
        self.assertFalse(
            torch.equal(discriminator_before, next(discriminator.parameters()))
        )


if __name__ == "__main__":
    unittest.main()
