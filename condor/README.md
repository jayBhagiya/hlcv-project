# Remote HTCondor setup

These files follow the LSV Docker and HTCondor layout. Jobs are not submitted automatically. Run all project scripts through HTCondor; use the submit machine only for repository and data setup, job submission, and monitoring.

Submit files currently assume:

```text
Project: /nethome/jabhagiya/projects/hlcv-project-gans
Data:    /data/users/jabhagiya/hlcv-project-gans
Host:    submit.lsv.uni-saarland.de
```

If your remote account uses different paths, update `project_dir` and `data_dir` at the top of all `.sub` files.

## 1. Copy the repository

Commit and push local changes, then run on the submit machine:

```bash
ssh jabhagiya@submit.lsv.uni-saarland.de
mkdir -p /nethome/jabhagiya/projects
git clone https://github.com/jayBhagiya/hlcv-project.git \
    /nethome/jabhagiya/projects/hlcv-project-gans
```

For an existing clone:

```bash
git -C /nethome/jabhagiya/projects/hlcv-project-gans pull --ff-only
```

## 2. Create shared storage

Run on the submit machine before submitting setup. HTCondor needs the log directory to exist when the job is submitted.

```bash
mkdir -p \
    /data/users/jabhagiya/hlcv-project-gans/data \
    /data/users/jabhagiya/hlcv-project-gans/logs \
    /data/users/jabhagiya/hlcv-project-gans/runs
```

## 3. Copy data

Run from the local repository:

```bash
rsync -ah --info=progress2 data/ \
    jabhagiya@submit.lsv.uni-saarland.de:/data/users/jabhagiya/hlcv-project-gans/data/
```

The remote `data/` directory must contain the same `real/` and `synthetic/` tree documented in the root README.

## 4. Create the uv environment

Run on the submit machine:

```bash
cd /nethome/jabhagiya/projects/hlcv-project-gans
condor_submit condor/setup.sub
condor_q
```

`setup.sub` installs uv 0.11.30, Python 3.11.15, and CUDA 11.8 PyTorch into shared storage. Inspect setup output under `/data/users/jabhagiya/hlcv-project-gans/logs/` before continuing.

## 5. Generate the manifest on a CPU worker

After setup finishes, submit the manifest job:

```bash
cd /nethome/jabhagiya/projects/hlcv-project-gans
condor_submit condor/data_manifest.sub
condor_q
```

The CPU job reads the remote images and writes `/data/users/jabhagiya/hlcv-project-gans/manifests/pairs.csv`. Its output should contain:

```text
Wrote 2126 pairs to /data/users/jabhagiya/hlcv-project-gans/manifests/pairs.csv (train=1554, val=233, test=339)
```

## 6. Submit training

Submit the L1 baseline:

```bash
condor_submit condor/train_l1.sub
condor_q
```

After validating the L1 baseline, submit Pix2Pix:

```bash
condor_submit condor/train_pix2pix.sub
condor_q
```

After validating Pix2Pix, submit the transformer-bottleneck L1 model:

```bash
condor_submit condor/train_transformer_l1.sub
condor_q
```

Jobs write `config.json`, `history.csv`, `best.pt`, `last.pt`, and fixed validation panels under their run directories. Panel columns are synthetic, prediction, and real:

```text
/data/users/jabhagiya/hlcv-project-gans/runs/unet-l1-seed-7/
/data/users/jabhagiya/hlcv-project-gans/runs/pix2pix-seed-7/
/data/users/jabhagiya/hlcv-project-gans/runs/transformer-l1-seed-7/
```

Trainers refuse to reuse a non-empty output directory. Change `run` in the relevant submit file before starting another run.

## 7. Compare validation results

After all three training jobs finish, submit:

```bash
condor_submit condor/evaluate.sub
condor_q
```

This job reads each `best.pt` and writes `summary.csv`, `summary.json`, `per-image.csv`, `comparison.png`, and `validation-curves.png` under:

```text
/data/users/jabhagiya/hlcv-project-gans/evaluations/evaluation-seed-7/
```

Evaluation is validation-only. Test data remains locked until final model selection. The comparison panel columns are synthetic, U-Net L1, Pix2Pix, transformer L1, and real.

## 8. W&B logging

Training jobs log scalar metrics directly to W&B. They expect W&B authentication to be available in the worker environment; no offline sync is required.

Never place a W&B API key in a submit file or commit it. Copy run directories back with `rsync` when local checkpoints or validation panels are needed.
