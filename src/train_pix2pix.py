"""Train Pix2Pix with the baseline U-Net and a 70x70 PatchGAN."""

import argparse
import csv
import json
import math
from pathlib import Path

import torch
import wandb
from torch import nn
from torch.utils.data import DataLoader

from src.paired_dataset import PairedImageDataset
from src.pix2pix import PatchDiscriminator
from src.train_l1 import run_epoch, save_checkpoint, save_validation_panel
from src.unet import UNet


def train_epoch(
    generator: UNet,
    discriminator: PatchDiscriminator,
    loader: DataLoader,
    device: torch.device,
    generator_optimizer: torch.optim.Optimizer,
    discriminator_optimizer: torch.optim.Optimizer,
    l1_weight: float = 100.0,
) -> dict[str, float]:
    generator.train()
    discriminator.train()
    adversarial_loss = nn.BCEWithLogitsLoss()
    totals = {"g_total": 0.0, "g_gan": 0.0, "g_l1": 0.0, "d": 0.0}
    total_samples = 0

    for synthetic, real, _ in loader:
        synthetic = synthetic.to(device, non_blocking=True)
        real = real.to(device, non_blocking=True)
        samples = synthetic.shape[0]

        generator_optimizer.zero_grad(set_to_none=True)
        prediction = generator(synthetic)

        discriminator.requires_grad_(True)
        discriminator_optimizer.zero_grad(set_to_none=True)
        real_logits = discriminator(synthetic, real)
        fake_logits = discriminator(synthetic, prediction.detach())
        discriminator_loss = 0.5 * (
            adversarial_loss(real_logits, torch.ones_like(real_logits))
            + adversarial_loss(fake_logits, torch.zeros_like(fake_logits))
        )
        discriminator_loss.backward()
        discriminator_optimizer.step()

        discriminator.requires_grad_(False)
        fake_logits = discriminator(synthetic, prediction)
        generator_gan = adversarial_loss(fake_logits, torch.ones_like(fake_logits))
        generator_l1 = nn.functional.l1_loss(prediction, real)
        generator_total = generator_gan + l1_weight * generator_l1
        generator_total.backward()
        generator_optimizer.step()
        discriminator.requires_grad_(True)

        for key, value in {
            "g_total": generator_total,
            "g_gan": generator_gan,
            "g_l1": generator_l1,
            "d": discriminator_loss,
        }.items():
            totals[key] += value.detach().item() * samples
        total_samples += samples

    return {key: value / total_samples for key, value in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("manifests/pairs.csv"))
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("runs/pix2pix/seed-7"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--l1-weight", type=float, default=100.0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--wandb-project", default="hlcv-sim2real")
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="disabled",
    )
    args = parser.parse_args()

    if args.height < 32 or args.width < 32 or args.height % 16 or args.width % 16:
        parser.error("Height and width must be at least 32 and divisible by 16")
    if args.l1_weight < 0:
        parser.error("L1 weight must be non-negative")
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

    generator = UNet().to(device)
    discriminator = PatchDiscriminator().to(device)
    generator_optimizer = torch.optim.Adam(
        generator.parameters(), lr=2e-4, betas=(0.5, 0.999)
    )
    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999)
    )
    config = {
        "method": "pix2pix",
        **{
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )

    fields = (
        "epoch",
        "train_g_total",
        "train_g_gan",
        "train_g_l1",
        "train_d",
        "val_l1",
        "val_psnr",
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
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            for epoch in range(1, args.epochs + 1):
                train = train_epoch(
                    generator,
                    discriminator,
                    train_loader,
                    device,
                    generator_optimizer,
                    discriminator_optimizer,
                    args.l1_weight,
                )
                val_l1, val_psnr = run_epoch(generator, val_loader, device)
                metrics = {
                    "epoch": epoch,
                    "train_g_total": train["g_total"],
                    "train_g_gan": train["g_gan"],
                    "train_g_l1": train["g_l1"],
                    "train_d": train["d"],
                    "val_l1": val_l1,
                    "val_psnr": val_psnr,
                }
                writer.writerow(metrics)
                file.flush()
                state = {
                    "epoch": epoch,
                    "generator": generator.state_dict(),
                    "discriminator": discriminator.state_dict(),
                    "metrics": metrics,
                    "config": config,
                }
                improved = val_l1 < best_l1
                if improved:
                    best_l1 = val_l1
                    save_checkpoint(args.output / "best.pt", state)
                    save_validation_panel(
                        args.output / "validation-best.png",
                        generator,
                        val_data,
                        device,
                    )
                save_checkpoint(args.output / "last.pt", state)
                save_validation_panel(
                    args.output / "validation-last.png", generator, val_data, device
                )
                run.log(metrics, step=epoch)
                print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
