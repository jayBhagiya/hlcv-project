#!/usr/bin/env python3
"""Build deterministic, scene-held-out pairs for synthetic-to-real training."""

from __future__ import annotations

import argparse
import csv
import re
import struct
from collections import Counter
from pathlib import Path


SCENE_SPLITS = {
    "e1": "train",
    "e6": "train",
    "e20": "train",
    "e2": "val",
    "e18": "test",
}
NAME_PATTERN = re.compile(r"^(e\d+)_(\d+)\.png$")
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _find_images(data_dir: Path, domain: str) -> dict[str, Path]:
    images: dict[str, Path] = {}
    for old_split in ("train", "val"):
        folder = data_dir / domain / old_split / "images"
        if not folder.is_dir():
            raise ValueError(f"Missing image folder: {folder}")
        for path in folder.glob("*.png"):
            if path.name in images:
                raise ValueError(f"Duplicate {domain} filename: {path.name}")
            images[path.name] = path
    return images


def _png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as file:
        header = file.read(24)
    if len(header) != 24 or header[:8] != PNG_SIGNATURE or header[12:16] != b"IHDR":
        raise ValueError(f"Invalid PNG header: {path}")
    return struct.unpack(">II", header[16:24])


def build_manifest(data_dir: Path, output: Path) -> Counter[str]:
    data_dir = data_dir.resolve()
    real = _find_images(data_dir, "real")
    synthetic = _find_images(data_dir, "synthetic")

    if real.keys() != synthetic.keys():
        real_only = sorted(real.keys() - synthetic.keys())[:5]
        synthetic_only = sorted(synthetic.keys() - real.keys())[:5]
        raise ValueError(
            f"Unpaired filenames; real-only={real_only}, synthetic-only={synthetic_only}"
        )

    parsed = []
    for filename in real:
        match = NAME_PATTERN.fullmatch(filename)
        if not match:
            raise ValueError(f"Unexpected filename: {filename}")
        scene, frame = match.groups()
        if scene not in SCENE_SPLITS:
            raise ValueError(f"No split configured for scene: {scene}")
        parsed.append((scene, frame, filename))

    missing_scenes = SCENE_SPLITS.keys() - {scene for scene, _, _ in parsed}
    if missing_scenes:
        raise ValueError(f"Configured scenes missing from data: {sorted(missing_scenes)}")

    rows = []
    counts: Counter[str] = Counter()
    for scene, frame, filename in sorted(
        parsed, key=lambda item: (int(item[0][1:]), int(item[1]))
    ):
        real_path = real[filename]
        synthetic_path = synthetic[filename]
        size = _png_size(real_path)
        if _png_size(synthetic_path) != size:
            raise ValueError(f"Pair dimensions differ: {filename}")
        split = SCENE_SPLITS[scene]
        counts[split] += 1
        rows.append(
            {
                "pair_id": Path(filename).stem,
                "scene": scene,
                "frame": frame,
                "split": split,
                "synthetic_path": synthetic_path.relative_to(data_dir.parent).as_posix(),
                "real_path": real_path.relative_to(data_dir.parent).as_posix(),
                "width": size[0],
                "height": size[1],
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("manifests/pairs.csv"))
    args = parser.parse_args()

    try:
        counts = build_manifest(args.data_dir, args.output)
    except ValueError as error:
        parser.error(str(error))
    total = sum(counts.values())
    summary = ", ".join(f"{split}={counts[split]}" for split in ("train", "val", "test"))
    print(f"Wrote {total} pairs to {args.output} ({summary})")


if __name__ == "__main__":
    main()
