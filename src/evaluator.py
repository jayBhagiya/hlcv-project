"""Compare trained generators on the locked validation split."""

import argparse
import csv
import json
import math
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from src.paired_dataset import PairedImageDataset
from src.transformer_unet import TransformerUNet
from src.unet import UNet


COLORS = ("#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c")


def parse_run(value: str) -> tuple[str, Path]:
    name, separator, checkpoint = value.partition("=")
    if not separator or not name or not checkpoint:
        raise argparse.ArgumentTypeError("Run must be NAME=CHECKPOINT")
    return name, Path(checkpoint)


def load_generator(
    checkpoint_path: Path, device: torch.device
) -> tuple[nn.Module, str, int]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Invalid checkpoint: {checkpoint_path}")
    config = checkpoint.get("config", {})
    if "generator" in checkpoint:
        model, method, weights = UNet(), "pix2pix", checkpoint["generator"]
    elif "model" in checkpoint:
        method = config.get("model", "unet")
        if method not in {"unet", "transformer"}:
            raise ValueError(f"Unknown model in checkpoint: {method}")
        model = TransformerUNet() if method == "transformer" else UNet()
        weights = checkpoint["model"]
    else:
        raise ValueError(f"Checkpoint has no generator weights: {checkpoint_path}")
    model.load_state_dict(weights)
    return model.to(device).eval(), method, int(checkpoint.get("epoch", 0))


def _psnr(mse: float) -> float:
    return math.inf if mse == 0 else -10 * math.log10(mse)


@torch.inference_mode()
def evaluate(
    models: dict[str, nn.Module], loader: DataLoader, device: torch.device
) -> tuple[list[dict[str, str | float]], dict[str, dict[str, float]]]:
    names = ("identity", *models)
    totals = {name: [0.0, 0.0, 0] for name in names}
    lpips_metrics = {
        name: LearnedPerceptualImagePatchSimilarity(
            net_type="alex", reduction="none", normalize=True
        ).to(device)
        for name in names
    }
    kid_metrics = {
        name: KernelInceptionDistance(
            feature=2048,
            subsets=100,
            subset_size=min(100, len(loader.dataset)),
            normalize=True,
        ).to(device)
        for name in names
    }
    rows: list[dict[str, str | float]] = []

    for synthetic, real, pair_ids in loader:
        synthetic = synthetic.to(device, non_blocking=True)
        real = real.to(device, non_blocking=True)
        predictions = {"identity": synthetic}
        predictions.update({name: model(synthetic) for name, model in models.items()})
        batch_rows = [{"pair_id": pair_id} for pair_id in pair_ids]
        for name, prediction in predictions.items():
            difference = prediction - real
            absolute = difference.abs().flatten(1)
            squared = difference.square().flatten(1)
            l1 = absolute.mean(1)
            mse = squared.mean(1)
            for index, row in enumerate(batch_rows):
                row[f"{name}_l1"] = l1[index].item()
                row[f"{name}_psnr"] = _psnr(mse[index].item())
            totals[name][0] += absolute.sum().item()
            totals[name][1] += squared.sum().item()
            totals[name][2] += difference.numel()
            lpips_metrics[name].update(prediction, real)
            kid_metrics[name].update(real, real=True)
            kid_metrics[name].update(prediction, real=False)
        rows.extend(batch_rows)

    summary = {}
    for name, (absolute, squared, pixels) in totals.items():
        lpips = lpips_metrics[name].compute().flatten().cpu()
        if len(lpips) != len(rows):
            raise RuntimeError(f"LPIPS returned {len(lpips)} scores for {len(rows)} images")
        for row, score in zip(rows, lpips, strict=True):
            row[f"{name}_lpips"] = score.item()
        torch.manual_seed(7)
        kid_mean, kid_std = kid_metrics[name].compute()
        summary[name] = {
            "l1": absolute / pixels,
            "psnr": _psnr(squared / pixels),
            "lpips": lpips.mean().item(),
            "kid_mean": kid_mean.item(),
            "kid_std": kid_std.item(),
        }
    return rows, summary


@torch.inference_mode()
def save_comparison_panel(
    path: Path,
    models: dict[str, nn.Module],
    dataset: PairedImageDataset,
    device: torch.device,
) -> None:
    indices = sorted({0, len(dataset) // 3, 2 * len(dataset) // 3, len(dataset) - 1})
    rows = []
    for index in indices:
        synthetic, real, _ = dataset[index]
        batch = synthetic[None].to(device)
        predictions = [model(batch)[0].cpu() for model in models.values()]
        rows.append(torch.cat((synthetic, *predictions, real), dim=2))
    panel = torch.cat(rows, dim=1)
    array = panel.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
    grid = Image.fromarray(array)
    header_height = 24
    output = Image.new("RGB", (grid.width, grid.height + header_height), "white")
    output.paste(grid, (0, header_height))
    draw = ImageDraw.Draw(output)
    column_width = dataset.size[1]
    for column, label in enumerate(("synthetic", *models, "real")):
        draw.text((column * column_width + 4, 6), label, fill="black")
    output.save(path)


def _read_history(checkpoint: Path) -> list[dict[str, str]]:
    path = checkpoint.parent / "history.csv"
    if not path.is_file():
        raise ValueError(f"Training history missing: {path}")
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows or not {"epoch", "val_l1", "val_psnr"} <= rows[0].keys():
        raise ValueError(f"Invalid training history: {path}")
    return rows


def save_validation_curves(path: Path, runs: list[tuple[str, Path]]) -> None:
    histories = {name: _read_history(checkpoint) for name, checkpoint in runs}
    image = Image.new("RGB", (1000, 700), "white")
    draw = ImageDraw.Draw(image)
    for index, (name, _) in enumerate(runs):
        x = 20 + index * 190
        draw.line((x, 25, x + 24, 25), fill=COLORS[index % len(COLORS)], width=3)
        draw.text((x + 30, 18), name, fill="black")

    for metric, title, box in (
        ("val_l1", "Validation L1", (70, 75, 970, 330)),
        ("val_psnr", "Validation PSNR (dB)", (70, 410, 970, 665)),
    ):
        x0, y0, x1, y1 = box
        series = {
            name: [(float(row["epoch"]), float(row[metric])) for row in rows]
            for name, rows in histories.items()
        }
        values = [value for points in series.values() for _, value in points]
        epochs = [epoch for points in series.values() for epoch, _ in points]
        low, high = min(values), max(values)
        if low == high:
            low, high = low - 0.5, high + 0.5
        first_epoch, last_epoch = min(epochs), max(epochs)
        draw.text((x0, y0 - 22), title, fill="black")
        draw.rectangle(box, outline="#6b7280")
        for tick in range(5):
            y = y1 - tick * (y1 - y0) / 4
            value = low + tick * (high - low) / 4
            draw.line((x0, y, x1, y), fill="#e5e7eb")
            draw.text((8, y - 7), f"{value:.3f}", fill="#374151")
        draw.text((x0, y1 + 8), f"epoch {first_epoch:g}", fill="#374151")
        draw.text((x1 - 65, y1 + 8), f"{last_epoch:g}", fill="#374151")
        for index, points in enumerate(series.values()):
            coordinates = []
            for epoch, value in points:
                x = (
                    x0
                    if first_epoch == last_epoch
                    else x0
                    + (epoch - first_epoch)
                    * (x1 - x0)
                    / (last_epoch - first_epoch)
                )
                y = y1 - (value - low) * (y1 - y0) / (high - low)
                coordinates.append((x, y))
            color = COLORS[index % len(COLORS)]
            if len(coordinates) == 1:
                x, y = coordinates[0]
                draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)
            else:
                draw.line(coordinates, fill=color, width=3)
    image.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("manifests/pairs.csv"))
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("evaluations/seed-7"))
    parser.add_argument("--run", type=parse_run, action="append", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    names = [name for name, _ in args.run]
    if len(names) != len(set(names)):
        parser.error("Run names must be unique")
    if "identity" in names:
        parser.error("Run name 'identity' is reserved")
    if args.height < 16 or args.width < 16 or args.height % 16 or args.width % 16:
        parser.error("Height and width must be at least 16 and divisible by 16")
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
    for _, checkpoint in args.run:
        if not checkpoint.is_file():
            parser.error(f"Checkpoint does not exist: {checkpoint}")
        try:
            _read_history(checkpoint)
        except ValueError as error:
            parser.error(str(error))

    device = torch.device(device_name)
    loaded = [load_generator(checkpoint, device) for _, checkpoint in args.run]
    models = {name: loaded[index][0] for index, (name, _) in enumerate(args.run)}
    methods = {name: loaded[index][1] for index, (name, _) in enumerate(args.run)}
    epochs = {name: loaded[index][2] for index, (name, _) in enumerate(args.run)}
    dataset = PairedImageDataset(
        args.manifest, "val", size=(args.height, args.width), root=args.data_root
    )
    if len(dataset) < 2:
        parser.error("KID requires at least two validation images")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )

    rows, metrics = evaluate(models, loader, device)
    args.output.mkdir(parents=True, exist_ok=True)
    fields = ["pair_id"] + [
        field
        for name in ("identity", *models)
        for field in (f"{name}_l1", f"{name}_psnr", f"{name}_lpips")
    ]
    with (args.output / "per-image.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary_rows = [
        {
            "name": name,
            "method": "identity" if name == "identity" else methods[name],
            "checkpoint_epoch": "" if name == "identity" else epochs[name],
            **values,
        }
        for name, values in metrics.items()
    ]
    with (args.output / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=summary_rows[0])
        writer.writeheader()
        writer.writerows(summary_rows)
    (args.output / "summary.json").write_text(
        json.dumps(
            {
                "split": "val",
                "samples": len(dataset),
                "size": [args.height, args.width],
                "lpips": {"network": "alex", "normalize": True},
                "kid": {
                    "feature": 2048,
                    "subsets": 100,
                    "subset_size": min(100, len(dataset)),
                    "normalize": True,
                    "seed": 7,
                },
                "panel_columns": ["synthetic", *models, "real"],
                "results": summary_rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    save_comparison_panel(args.output / "comparison.png", models, dataset, device)
    save_validation_curves(args.output / "validation-curves.png", args.run)
    print(json.dumps(summary_rows), flush=True)


if __name__ == "__main__":
    main()
