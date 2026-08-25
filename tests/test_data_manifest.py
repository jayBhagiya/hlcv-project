import base64
import csv
import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data_manifest import build_manifest
from src.paired_dataset import PairedImageDataset
from src.train_l1 import run_epoch
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
            optimizer = torch.optim.Adam(model.parameters())
            train_l1, train_psnr = run_epoch(
                model,
                DataLoader(PairedImageDataset(output, "train", size=(32, 48)), batch_size=3),
                torch.device("cpu"),
                optimizer,
            )
            self.assertGreater(train_l1, 0)
            self.assertGreater(train_psnr, 0)

            (root / "data/real/val/images/e18_000000.png").unlink()
            with self.assertRaisesRegex(ValueError, "Unpaired filenames"):
                build_manifest(root / "data", output)


if __name__ == "__main__":
    unittest.main()
