import base64
import csv
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from src.data_manifest import build_manifest
from src.paired_dataset import PairedImageDataset
from src.train_l1 import main as train_main
from src.unet import UNet


PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


class DataManifestTest(unittest.TestCase):
    def test_builds_scene_held_out_pairs_and_rejects_missing_pair(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for domain in ("real", "synthetic"):
                for scene, old_split in {
                    "e1": "train",
                    "e2": "val",
                    "e6": "train",
                    "e18": "val",
                    "e20": "train",
                }.items():
                    path = root / "data" / domain / old_split / "images" / f"{scene}_000000.png"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(PNG_1X1)
                for old_split in ("train", "val"):
                    (root / "data" / domain / old_split / "images").mkdir(
                        parents=True, exist_ok=True
                    )

            output = root / "manifests" / "pairs.csv"
            counts = build_manifest(root / "data", output)
            with output.open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))

            self.assertEqual(counts, {"train": 3, "val": 1, "test": 1})
            self.assertEqual(len(rows), 5)
            self.assertEqual({row["split"] for row in rows if row["scene"] == "e18"}, {"test"})
            self.assertTrue(all(row["width"] == row["height"] == "1" for row in rows))

            synthetic, real, pair_id = PairedImageDataset(output, "test", size=(32, 48))[0]
            self.assertEqual(
                (synthetic.shape, real.shape), ((3, 32, 48), (3, 32, 48))
            )
            self.assertEqual(pair_id, "e18_000000")
            model = UNet(base_channels=4)
            self.assertEqual(model(synthetic[None]).shape, (1, 3, 32, 48))

            training_output = root / "training"
            arguments = [
                "train_l1",
                "--manifest",
                str(output),
                "--output",
                str(training_output),
                "--epochs",
                "5",
                "--patience",
                "1",
                "--batch-size",
                "3",
                "--workers",
                "0",
                "--height",
                "32",
                "--width",
                "48",
                "--device",
                "cpu",
                "--wandb-mode",
                "offline",
            ]
            epochs = [
                (0.5, 0.0),
                (0.4, 8.0),
                (0.4, 0.0),
                (0.5, 7.0),
            ]
            with patch.object(sys, "argv", arguments), patch(
                "src.train_l1.run_epoch", side_effect=epochs
            ), patch("src.train_l1.wandb.init") as wandb_init:
                train_main()
            self.assertTrue((training_output / "best.pt").is_file())
            self.assertTrue((training_output / "last.pt").is_file())
            self.assertTrue((training_output / "validation-best.png").is_file())
            self.assertTrue((training_output / "validation-last.png").is_file())
            with Image.open(training_output / "validation-best.png") as panel:
                self.assertEqual(panel.size, (144, 32))
            with (training_output / "history.csv").open(encoding="utf-8") as file:
                self.assertEqual(len(file.readlines()), 3)
            self.assertEqual(wandb_init.call_args.kwargs["mode"], "offline")
            self.assertEqual(
                wandb_init.return_value.__enter__.return_value.log.call_count, 2
            )

            (root / "data/real/val/images/e18_000000.png").unlink()
            with self.assertRaisesRegex(ValueError, "Unpaired filenames"):
                build_manifest(root / "data", output)


if __name__ == "__main__":
    unittest.main()
