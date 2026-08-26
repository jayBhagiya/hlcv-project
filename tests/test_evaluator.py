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
                writer.writerow(
                    {
                        "pair_id": "sample",
                        "split": "val",
                        "synthetic_path": "images/synthetic.png",
                        "real_path": "images/real.png",
                    }
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
            with patch.object(sys, "argv", arguments):
                main()

            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["split"], "val")
            self.assertEqual(summary["samples"], 1)
            self.assertEqual(
                [result["name"] for result in summary["results"]],
                ["identity", "unet", "pix2pix", "transformer"],
            )
            lines = (output / "per-image.csv").read_text().splitlines()
            self.assertEqual(len(lines), 2)
            with Image.open(output / "comparison.png") as panel:
                self.assertEqual(panel.size, (240, 56))
            self.assertTrue((output / "validation-curves.png").is_file())


if __name__ == "__main__":
    unittest.main()
