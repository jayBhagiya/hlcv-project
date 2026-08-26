import unittest

import torch

from src.transformer_unet import TransformerUNet
from src.unet import UNet


class TransformerUNetTest(unittest.TestCase):
    def test_forward_and_backward(self) -> None:
        torch.manual_seed(0)
        model = TransformerUNet(base_channels=2)
        image = torch.rand(1, 3, 32, 48)
        prediction = model(image)
        self.assertEqual(prediction.shape, image.shape)
        prediction.mean().backward()
        self.assertIsNotNone(next(model.parameters()).grad)
        self.assertGreater(
            sum(parameter.numel() for parameter in model.parameters()),
            sum(parameter.numel() for parameter in UNet(base_channels=2).parameters()),
        )


if __name__ == "__main__":
    unittest.main()
