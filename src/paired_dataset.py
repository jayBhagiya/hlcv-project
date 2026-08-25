"""Paired images loaded from the frozen experiment manifest."""

from __future__ import annotations

import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


class PairedImageDataset(Dataset):
    def __init__(
        self,
        manifest: Path,
        split: str,
        size: tuple[int, int] = (256, 384),
        root: Path | None = None,
        horizontal_flip: bool = False,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split: {split}")
        self.root = root.resolve() if root else manifest.resolve().parent.parent
        self.size = size
        self.horizontal_flip = horizontal_flip
        with manifest.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            required = {"pair_id", "split", "synthetic_path", "real_path"}
            missing = required - set(reader.fieldnames or ())
            if missing:
                raise ValueError(f"Manifest columns missing: {sorted(missing)}")
            self.rows = [row for row in reader if row["split"] == split]
        if not self.rows:
            raise ValueError(f"No rows for split: {split}")

    def __len__(self) -> int:
        return len(self.rows)

    def _load(self, relative_path: str) -> torch.Tensor:
        path = self.root / relative_path
        with Image.open(path) as image:
            image = image.convert("RGB").resize(
                (self.size[1], self.size[0]), Image.Resampling.BICUBIC
            )
            tensor = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
        return tensor.view(self.size[0], self.size[1], 3).permute(2, 0, 1).float().div_(255)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        row = self.rows[index]
        synthetic = self._load(row["synthetic_path"])
        real = self._load(row["real_path"])
        if self.horizontal_flip and torch.rand(()) < 0.5:
            synthetic = synthetic.flip(-1)
            real = real.flip(-1)
        return synthetic, real, row["pair_id"]
