# Synthetic-to-real image translation

High-Level Computer Vision course project for translating Virtual KITTI images into KITTI-like street scenes. The experiments compare four approaches:

- convolutional U-Net trained with L1 reconstruction
- Pix2Pix with a U-Net generator and PatchGAN discriminator
- parameter-matched transformer-bottleneck U-Net trained with L1
- unpaired CycleGAN-Turbo with an SD-Turbo generator

## Repository layout

```text
src/         datasets, models, trainers, and evaluation
tests/       focused unit tests for the active pipeline
condor/      uv setup and HTCondor submit files
manifests/   frozen scene-held-out split
reports/     course proposal and reports
```

## Data

Image data are not committed. Use the following layout:

```text
data/
├── real/{train,val}/images/
└── synthetic/{train,val}/images/
```

The frozen manifest contains 2,126 filename-matched pairs: 1,554 training, 233 validation, and 339 test images. To generate or verify it locally:

```bash
python -m src.data_manifest --data-dir data --output manifests/pairs.csv
```

Use `condor/data_manifest.sub` when the data live on a cluster filesystem.

## Local environment and checks

```bash
uv python install 3.11.15
uv venv --python 3.11.15
uv pip install --python .venv/bin/python --torch-backend cpu -e .
.venv/bin/python -m unittest discover -s tests -v
```

## HTCondor

See [`condor/README.md`](condor/README.md) for portable path configuration, environment setup, training, and evaluation.

CycleGAN-Turbo uses the official [img2img-turbo](https://github.com/GaParmar/img2img-turbo) runtime pinned to commit `86f5414`. The setup job downloads and verifies only the required upstream files.
