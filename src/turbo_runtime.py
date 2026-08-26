"""Fetch and import the pinned official img2img-turbo model runtime."""

import argparse
import hashlib
import importlib
import os
import sys
import zipfile
from pathlib import Path
from urllib.request import urlopen


UPSTREAM_COMMIT = "86f54146590ffb4543c8cf85b5a36657da670924"
UPSTREAM_ARCHIVE_SHA256 = (
    "22e02fbf1c29d95338ce2153cac9ff4a1de89f9e0c300084cd99253adb40d0cd"
)
UPSTREAM_FILES = {
    "src/cyclegan_turbo.py": "7d59485b1d3a07fb4980a3d29a62feee81ed331ba8d84f23f8f83f23e8d8261f",
    "src/model.py": "8b21c91f7f1992883c72339532745319a4ff30b2edbf9c6d889a9bcdb5b80c1d",
    "LICENSE": "095da3e9d59babae7e41fbab33aa183ed1f45cf1be87a102c985d04db91ea88b",
}


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_upstream(output: Path) -> None:
    missing = []
    for relative, expected in UPSTREAM_FILES.items():
        destination = output / relative
        if destination.is_file():
            if _digest(destination) != expected:
                raise ValueError(f"Checksum mismatch: {destination}")
        else:
            missing.append(relative)
    if not missing:
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    archive = output.parent / f".img2img-turbo-{UPSTREAM_COMMIT}.zip"
    url = f"https://github.com/GaParmar/img2img-turbo/archive/{UPSTREAM_COMMIT}.zip"
    with urlopen(url, timeout=120) as response:
        archive.write_bytes(response.read())
    if _digest(archive) != UPSTREAM_ARCHIVE_SHA256:
        archive.unlink()
        raise ValueError("Downloaded img2img-turbo archive checksum mismatch")
    prefix = f"img2img-turbo-{UPSTREAM_COMMIT}/"
    with zipfile.ZipFile(archive) as source:
        for relative in missing:
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            temporary.write_bytes(source.read(prefix + relative))
            if _digest(temporary) != UPSTREAM_FILES[relative]:
                temporary.unlink()
                raise ValueError(f"Extracted checksum mismatch: {relative}")
            temporary.replace(destination)
    archive.unlink()


def load_upstream(root: Path | None = None):
    root = root or Path(os.environ["CYCLEGAN_TURBO_ROOT"])
    for relative, expected in UPSTREAM_FILES.items():
        path = root / relative
        if not path.is_file() or _digest(path) != expected:
            raise RuntimeError(
                f"Pinned CycleGAN-Turbo source missing or changed: {path}; "
                "submit condor/setup_turbo.sub"
            )
    source = str(root / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    return importlib.import_module("cyclegan_turbo")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fetch_upstream(args.output)
    print(f"Fetched img2img-turbo {UPSTREAM_COMMIT} to {args.output}")


if __name__ == "__main__":
    main()
