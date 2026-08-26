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

## 7. Replicate the L1 finalists

The L1 trainer stops after 10 consecutive epochs without validation-L1 improvement. Submit two additional seeds for both U-Net and transformer:

```bash
condor_submit condor/replicate_l1.sub
condor_q
```

One submission queues four independent jobs: seeds 21 and 42 for each model. Do not rerun Pix2Pix unchanged. Each run keeps its best and last checkpoints under the existing `runs/` directory.

## 8. Evaluate all validation runs

Evaluation now includes L1, PSNR, LPIPS, and KID. After pulling these changes, rerun the setup job once to install the image-metric dependencies:

```bash
condor_submit condor/setup.sub
condor_q
```

After setup finishes, rerun seed 7 and evaluate seeds 21 and 42:

```bash
condor_submit condor/evaluate_seed_7.sub
condor_submit condor/evaluate_replicates.sub
condor_q
```

The second submission queues one job per replication seed. Jobs read each `best.pt` and write `summary.csv`, `summary.json`, `per-image.csv`, `comparison.png`, and `validation-curves.png` under:

```text
/data/users/jabhagiya/hlcv-project-gans/evaluations/evaluation-seed-7-lpips-kid/
/data/users/jabhagiya/hlcv-project-gans/evaluations/evaluation-seed-21-lpips-kid/
/data/users/jabhagiya/hlcv-project-gans/evaluations/evaluation-seed-42-lpips-kid/
```

Seed 7 compares U-Net, Pix2Pix, and transformer. Seeds 21 and 42 compare only the two L1 finalists. KID uses 100 fixed-seed subsets of 100 validation images. Evaluation is validation-only; test data remains locked until final model selection.

## 9. Train and evaluate CycleGAN-Turbo

CycleGAN-Turbo uses a separate uv environment so its pinned diffusion dependencies do not disturb the completed baselines. Submit its setup job first:

```bash
condor_submit condor/setup_turbo.sub
condor_q
```

The job installs the `turbo` dependency extra and downloads the checksum-verified model runtime from official img2img-turbo commit `86f5414`. Model and pretrained feature weights use shared caches under `/data/users/jabhagiya/hlcv-project-gans/cache/`.

After setup succeeds, submit the single seed-7 training run:

```bash
condor_submit condor/train_cyclegan_turbo.sub
condor_q
```

The trainer samples the synthetic and real training domains independently, runs 5,000 steps at `256x384`, logs scalar losses to W&B, and writes `config.json`, `history.csv`, `last.pt`, and `validation-last.png` under:

```text
/data/users/jabhagiya/hlcv-project-gans/runs/cyclegan-turbo-seed-7/
```

It never uploads dataset or generated images to W&B. When training finishes, evaluate `last.pt` together with the seed-7 baselines:

```bash
condor_submit condor/evaluate_cyclegan_turbo.sub
condor_q
```

The validation-only result is written to:

```text
/data/users/jabhagiya/hlcv-project-gans/evaluations/evaluation-seed-7-with-turbo/
```

The job keeps test scene `e18` locked and uses batch size 1 because the diffusion backbone is substantially larger than the earlier generators.

## 10. W&B logging

Training jobs log scalar metrics directly to W&B. They expect W&B authentication to be available in the worker environment; no offline sync is required. CycleGAN-Turbo deliberately logs no images.

Never place a W&B API key in a submit file or commit it. Copy run directories back with `rsync` when local checkpoints or validation panels are needed.
