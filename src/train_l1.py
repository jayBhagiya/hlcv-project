"""Train a scene-held-out image translator with pixel L1 loss."""

import argparse
import csv
import json
import math
from pathlib import Path

import torch
import wandb
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

from src.paired_dataset import PairedImageDataset
from src.transformer_unet import TransformerUNet
from src.unet import UNet


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_absolute = 0.0
    total_squared = 0.0
    total_pixels = 0

    with torch.set_grad_enabled(training):
        for synthetic, real, _ in loader:
            synthetic = synthetic.to(device, non_blocking=True)
            real = real.to(device, non_blocking=True)
            prediction = model(synthetic)
            difference = prediction - real
            loss = difference.abs().mean()
            if optimizer:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_absolute += difference.detach().abs().sum().item()
            total_squared += difference.detach().square().sum().item()
            total_pixels += real.numel()

    l1 = total_absolute / total_pixels
    mse = total_squared / total_pixels
    return l1, -10 * math.log10(mse) if mse else math.inf


def save_checkpoint(path: Path, state: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def save_validation_panel(
    path: Path,
    model: nn.Module,
    dataset: PairedImageDataset,
    device: torch.device,
) -> None:
    was_training = model.training
    model.eval()
    rows = []
    indices = sorted(
        {0, len(dataset) // 3, 2 * len(dataset) // 3, len(dataset) - 1}
    )
    with torch.no_grad():
        for index in indices:
            synthetic, real, _ = dataset[index]
            prediction = model(synthetic[None].to(device))[0].cpu()
            rows.append(torch.cat((synthetic, prediction, real), dim=2))
    panel = torch.cat(rows, dim=1)
    array = panel.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
    temporary = path.with_suffix(".tmp.png")
    Image.fromarray(array).save(temporary)
    temporary.replace(path)
    model.train(was_training)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("manifests/pairs.csv"))
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("runs/unet-l1/seed-7"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model", choices=("unet", "transformer"), default="unet")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--wandb-project", default="hlcv-sim2real")
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="disabled",
    )
    args = parser.parse_args()

    if args.height % 16 or args.width % 16:
        parser.error("Height and width must be divisible by 16")
    if args.output.exists() and (
        not args.output.is_dir() or any(args.output.iterdir())
    ):
        parser.error(f"Output path is not an empty directory: {args.output}")
    device_name = (
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    if device_name == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")

    torch.manual_seed(args.seed)
    device = torch.device(device_name)
    size = (args.height, args.width)
    train_data = PairedImageDataset(
        args.manifest,
        "train",
        size=size,
        root=args.data_root,
        horizontal_flip=True,
    )
    val_data = PairedImageDataset(args.manifest, "val", size=size, root=args.data_root)
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(
        train_data,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        **loader_options,
    )
    val_loader = DataLoader(val_data, shuffle=False, **loader_options)

    model = (TransformerUNet() if args.model == "transformer" else UNet()).to(
        device
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, betas=(0.5, 0.999))
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )

    with wandb.init(
        project=args.wandb_project,
        name=args.output.name,
        config=config,
        mode=args.wandb_mode,
        dir=args.output,
        job_type="train",
    ) as run:
        best_l1 = math.inf
        with (args.output / "history.csv").open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(
                file, fieldnames=("epoch", "train_l1", "val_l1", "val_psnr")
            )
            writer.writeheader()
            for epoch in range(1, args.epochs + 1):
                train_l1, _ = run_epoch(model, train_loader, device, optimizer)
                val_l1, val_psnr = run_epoch(model, val_loader, device)
                metrics = {
                    "epoch": epoch,
                    "train_l1": train_l1,
                    "val_l1": val_l1,
                    "val_psnr": val_psnr,
                }
                writer.writerow(metrics)
                file.flush()
                state = {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "metrics": metrics,
                    "config": config,
                }
                improved = val_l1 < best_l1
                if improved:
                    best_l1 = val_l1
                    save_checkpoint(args.output / "best.pt", state)
                    save_validation_panel(
                        args.output / "validation-best.png", model, val_data, device
                    )
                save_checkpoint(args.output / "last.pt", state)
                save_validation_panel(
                    args.output / "validation-last.png", model, val_data, device
                )
                run.log(metrics, step=epoch)
                print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
