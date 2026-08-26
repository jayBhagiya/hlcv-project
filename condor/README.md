# Remote HTCondor setup

These files follow the LSV Docker and HTCondor layout. Jobs are not submitted automatically.

Submit files currently assume:

```text
Project: /nethome/jabhagiya/projects/hlcv-project-gans
Data:    /data/users/jabhagiya/hlcv-project-gans
Host:    submit.lsv.uni-saarland.de
```

If your remote account uses different paths, update `project_dir` and `data_dir` at the top of both `.sub` files.

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

## 5. Verify remote data

After setup finishes:

```bash
cd /nethome/jabhagiya/projects/hlcv-project-gans
/data/users/jabhagiya/hlcv-project-gans/venvs/hlcv-project-gans/bin/python \
    -m src.data_manifest \
    --data-dir /data/users/jabhagiya/hlcv-project-gans/data \
    --output manifests/pairs.csv
```

Expected output:

```text
Wrote 2126 pairs to manifests/pairs.csv (train=1554, val=233, test=339)
```

## 6. Submit training

```bash
condor_submit condor/train_l1.sub
condor_q
```

Training writes `config.json`, `history.csv`, and `best.pt` under:

```text
/data/users/jabhagiya/hlcv-project-gans/runs/unet-l1-seed-7/
```

Trainer refuses to reuse a non-empty output directory. Change `run` in `train_l1.sub` before starting another run.
