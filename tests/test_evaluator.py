import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

from src.evaluator import main
from src.transformer_unet import TransformerUNet
from src.unet import UNet


class FakeLPIPS:
    def __init__(self, **_: object) -> None:
        self.scores = []

    def to(self, _: torch.device) -> "FakeLPIPS":
        return self

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        self.scores.append((prediction - target).abs().flatten(1).mean(1))

    def compute(self) -> torch.Tensor:
        return torch.cat(self.scores)


class FakeKID:
    def __init__(self, **_: object) -> None:
        self.real = []
        self.fake = []

    def to(self, _: torch.device) -> "FakeKID":
        return self

    def update(self, images: torch.Tensor, real: bool) -> None:
        (self.real if real else self.fake).append(images)

    def compute(self) -> tuple[torch.Tensor, torch.Tensor]:
        difference = torch.cat(self.real).mean() - torch.cat(self.fake).mean()
        return difference.square(), torch.tensor(0.0)


class EvaluatorTest(unittest.TestCase):
    def test_compares_checkpoints_on_validation_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "data/images"
            images.mkdir(parents=True)
            Image.new("RGB", (48, 32), "#304050").save(images / "synthetic.png")
            Image.new("RGB", (48, 32), "#607080").save(images / "real.png")
            manifest = root / "manifest.csv"
            with manifest.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=("pair_id", "split", "synthetic_path", "real_path"),
                )
                writer.writeheader()
                writer.writerows(
                    {
                        "pair_id": f"sample-{index}",
                        "split": "val",
                        "synthetic_path": "images/synthetic.png",
                        "real_path": "images/real.png",
                    }
                    for index in range(2)
                )

            runs = []
            for name, key, config, model in (
                ("unet", "model", {"model": "unet"}, UNet()),
                ("pix2pix", "generator", {"method": "pix2pix"}, UNet()),
                (
                    "transformer",
                    "model",
                    {"model": "transformer"},
                    TransformerUNet(),
                ),
            ):
                run = root / name
                run.mkdir()
                checkpoint = run / "best.pt"
                torch.save(
                    {"epoch": 2, key: model.state_dict(), "config": config},
                    checkpoint,
                )
                (run / "history.csv").write_text(
                    "epoch,val_l1,val_psnr\n1,0.3,9.0\n2,0.2,10.0\n",
                    encoding="utf-8",
                )
                runs.extend(("--run", f"{name}={checkpoint}"))

            turbo = root / "turbo"
            turbo.mkdir()
            turbo_checkpoint = turbo / "last.pt"
            torch.save(
                {
                    "epoch": 4,
                    "config": {
                        "method": "cyclegan-turbo",
                        "target_prompt": "a real driving scene",
                    },
                },
                turbo_checkpoint,
            )
            (turbo / "history.csv").write_text(
                "step,epoch,cycle_a\n1,1,0.5\n", encoding="utf-8"
            )
            runs.extend(("--run", f"cyclegan-turbo={turbo_checkpoint}"))

            output = root / "evaluation"
            arguments = [
                "evaluator",
                "--manifest",
                str(manifest),
                "--data-root",
                str(root / "data"),
                "--output",
                str(output),
                "--height",
                "32",
                "--width",
                "48",
                "--workers",
                "0",
                "--device",
                "cpu",
                *runs,
            ]
            with patch.object(sys, "argv", arguments), patch(
                "src.evaluator.LearnedPerceptualImagePatchSimilarity", FakeLPIPS
            ), patch("src.evaluator.KernelInceptionDistance", FakeKID), patch(
                "src.turbo_generator.load_turbo_generator",
                return_value=torch.nn.Identity(),
            ):
                main()

            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["split"], "val")
            self.assertEqual(summary["samples"], 2)
            self.assertEqual(
                [result["name"] for result in summary["results"]],
                ["identity", "unet", "pix2pix", "transformer", "cyclegan-turbo"],
            )
            self.assertTrue(
                all(
                    {"lpips", "kid_mean", "kid_std"} <= result.keys()
                    for result in summary["results"]
                )
            )
            with (output / "per-image.csv").open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 2)
            self.assertIn("cyclegan-turbo_lpips", rows[0])
            with Image.open(output / "comparison.png") as panel:
                self.assertEqual(panel.size, (288, 88))
            self.assertTrue((output / "validation-curves.png").is_file())


if __name__ == "__main__":
    unittest.main()
