# Synthetic-to-real image translation

Reimplementation of the HLCV course project translating Virtual KITTI images into KITTI-like real images. Current work uses scene-held-out data splits and starts with a U-Net trained with L1 loss.

## Repository layout

- `src/data_manifest.py`, `src/paired_dataset.py`, `src/unet.py`, `src/overfit_l1.py`, and `src/train_l1.py`: current implementation.
- `manifests/pairs.csv`: frozen train, validation, and test assignments.
- `condor/`: uv environment and HTCondor job files.
- `reports/`: original proposal and course reports.
- Remaining `src/` scripts and `notebooks/`: historical experiments kept for reference.

## Data

Data is intentionally not committed. Expected image layout:

```text
data/
├── real/{train,val}/images/
└── synthetic/{train,val}/images/
```

Generate or verify the manifest:

```bash
python -m src.data_manifest --data-dir data --output manifests/pairs.csv
```

This command is for local use. Remote manifest generation uses the CPU job in `condor/data_manifest.sub`.

Expected result: 2,126 pairs, split into 1,554 train, 233 validation, and 339 test images.

## Local setup and checks

```bash
uv python install 3.11.15
uv venv --python 3.11.15
uv pip install --python .venv/bin/python --torch-backend cpu -e .
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m src.overfit_l1 --device cpu
```

For remote GPU setup and data transfer, see [`condor/README.md`](condor/README.md).
