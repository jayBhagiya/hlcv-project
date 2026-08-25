"""Prove the L1 pipeline can memorize eight real pairs before full training."""

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from src.paired_dataset import PairedImageDataset
from src.unet import UNet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("manifests/pairs.csv"))
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-loss-ratio", type=float, default=0.60)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = (
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")

    dataset = PairedImageDataset(args.manifest, "train", (args.height, args.width))
    if len(dataset) < 8:
        parser.error("Training split needs at least 8 pairs")
    indices = [round(index * (len(dataset) - 1) / 7) for index in range(8)]
    synthetic, real, pair_ids = next(
        iter(DataLoader(Subset(dataset, indices), batch_size=8, shuffle=False))
    )
    synthetic, real = synthetic.to(device), real.to(device)

    model = UNet().to(device)
    loss_function = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    with torch.no_grad():
        initial_loss = loss_function(model(synthetic), real).item()
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_function(model(synthetic), real)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        final_loss = loss_function(model(synthetic), real).item()

    loss_ratio = final_loss / initial_loss
    print(
        json.dumps(
            {
                "device": device,
                "pairs": list(pair_ids),
                "steps": args.steps,
                "initial_l1": initial_loss,
                "final_l1": final_loss,
                "loss_ratio": loss_ratio,
            }
        )
    )
    if loss_ratio > args.max_loss_ratio:
        raise SystemExit(
            f"Overfit check failed: loss ratio {loss_ratio:.3f} > {args.max_loss_ratio:.3f}"
        )


if __name__ == "__main__":
    main()
