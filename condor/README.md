# HTCondor workflow

Use the submit machine only for repository and data setup, job submission, and monitoring. All Python commands run inside HTCondor jobs.

## Configure paths

Every submit file accepts two command-line macros:

- `PROJECT_DIR`: repository clone visible to the workers
- `DATA_DIR`: shared data, environments, caches, logs, runs, and evaluations

Both default to the directory from which `condor_submit` is called. For separate storage, define the locations on the submit machine:

```bash
export HLCV_PROJECT_DIR="$PWD"
export HLCV_DATA_DIR="<shared-data-directory>"
mkdir -p "$HLCV_DATA_DIR"/{data,logs,runs,evaluations,manifests}
```

Pass the values before the submit file:

```bash
condor_submit \
  PROJECT_DIR="$HLCV_PROJECT_DIR" \
  DATA_DIR="$HLCV_DATA_DIR" \
  condor/setup.sub
```

HTCondor requires the log directory to exist before any job is submitted. The chosen project and data directories must also be mounted on worker machines because the jobs use `should_transfer_files = NO`.

## Repository and data

Clone or update the repository on the submit machine:

```bash
git clone https://github.com/jayBhagiya/hlcv-project.git hlcv-project-gans
cd hlcv-project-gans
git pull --ff-only
```

Copy the local `data/` contents to `<shared-data-directory>/data/`. The remote tree must match the layout in the root README. A typical transfer from the local repository is:

```bash
rsync -ah --info=progress2 data/ <submit-host>:<shared-data-directory>/data/
```

## Base environment and manifest

Create the base uv environment:

```bash
condor_submit PROJECT_DIR="$HLCV_PROJECT_DIR" DATA_DIR="$HLCV_DATA_DIR" condor/setup.sub
```

The job installs uv 0.11.30, Python 3.11.15, and CUDA 11.8 PyTorch under `DATA_DIR`. After it finishes, generate the manifest on a CPU worker:

```bash
condor_submit PROJECT_DIR="$HLCV_PROJECT_DIR" DATA_DIR="$HLCV_DATA_DIR" condor/data_manifest.sub
```

Expected manifest counts are 1,554 training, 233 validation, and 339 test pairs.

## Baseline training

Submit the three seed-7 baselines:

```bash
condor_submit PROJECT_DIR="$HLCV_PROJECT_DIR" DATA_DIR="$HLCV_DATA_DIR" condor/train_l1.sub
condor_submit PROJECT_DIR="$HLCV_PROJECT_DIR" DATA_DIR="$HLCV_DATA_DIR" condor/train_pix2pix.sub
condor_submit PROJECT_DIR="$HLCV_PROJECT_DIR" DATA_DIR="$HLCV_DATA_DIR" condor/train_transformer_l1.sub
```

Run the additional U-Net and transformer seeds with:

```bash
condor_submit PROJECT_DIR="$HLCV_PROJECT_DIR" DATA_DIR="$HLCV_DATA_DIR" condor/replicate_l1.sub
```

Trainers refuse to reuse non-empty output directories. Change the `run` macro in the relevant submit file before starting a replacement run.

## Baseline evaluation

Evaluation reports L1, PSNR, LPIPS, and KID. Submit seed 7 and the two replicate seeds after training completes:

```bash
condor_submit PROJECT_DIR="$HLCV_PROJECT_DIR" DATA_DIR="$HLCV_DATA_DIR" condor/evaluate_seed_7.sub
condor_submit PROJECT_DIR="$HLCV_PROJECT_DIR" DATA_DIR="$HLCV_DATA_DIR" condor/evaluate_replicates.sub
```

Seed 7 compares U-Net, Pix2Pix, and the transformer. Seeds 21 and 42 compare the two L1 models. KID uses 100 fixed-seed subsets of 100 validation images.

## CycleGAN-Turbo

CycleGAN-Turbo uses a separate uv environment. Create it first:

```bash
condor_submit PROJECT_DIR="$HLCV_PROJECT_DIR" DATA_DIR="$HLCV_DATA_DIR" condor/setup_turbo.sub
```

The setup job installs the `turbo` dependency extra and downloads the checksum-verified runtime from img2img-turbo commit `86f5414`. Then submit training and final evaluation:

```bash
condor_submit PROJECT_DIR="$HLCV_PROJECT_DIR" DATA_DIR="$HLCV_DATA_DIR" condor/train_cyclegan_turbo.sub
condor_submit PROJECT_DIR="$HLCV_PROJECT_DIR" DATA_DIR="$HLCV_DATA_DIR" condor/evaluate_cyclegan_turbo.sub
```

Training runs for 5,000 optimizer steps at `256x384`. The final evaluation compares CycleGAN-Turbo with the three seed-7 baselines. All evaluations use the validation split; test scene `e18` remains locked.

## Outputs and monitoring

Paths below are relative to `DATA_DIR`:

```text
logs/          HTCondor stdout, stderr, and event logs
runs/          configs, histories, checkpoints, and validation panels
evaluations/   summaries, per-image metrics, curves, and comparison panels
manifests/     generated pair manifest
```

Use `condor_q` to monitor jobs and `condor_q -better-analyze <job-id>` to inspect a waiting job. Training logs scalar metrics directly to W&B. Keep API keys out of submit files and version control.
