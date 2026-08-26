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

Jobs write `config.json`, `history.csv`, and `best.pt` under their run directories:

```text
/data/users/jabhagiya/hlcv-project-gans/runs/unet-l1-seed-7/
/data/users/jabhagiya/hlcv-project-gans/runs/pix2pix-seed-7/
```

Trainers refuse to reuse a non-empty output directory. Change `run` in the relevant submit file before starting another run.

## 7. Sync W&B from the local machine

`train_l1.sub` records W&B runs offline, so no W&B command or API key is needed on the submit machine. After training finishes, copy the run back from your local repository:

```bash
rsync -ah --info=progress2 \
    jabhagiya@submit.lsv.uni-saarland.de:/data/users/jabhagiya/hlcv-project-gans/runs/unet-l1-seed-7/ \
    runs/unet-l1-seed-7/
.venv/bin/wandb login
.venv/bin/wandb sync runs/unet-l1-seed-7/wandb/offline-run-*
```

Replace `unet-l1-seed-7` with `pix2pix-seed-7` to copy and sync the Pix2Pix run.

Never place a W&B API key in a submit file or commit it.
