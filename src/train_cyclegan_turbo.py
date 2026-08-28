"""Fine-tune CycleGAN-Turbo on unpaired synthetic and real driving images.

The training objective follows the MIT-licensed official img2img-turbo implementation.
"""

import argparse
import copy
import csv
import json
from pathlib import Path

import torch
import wandb
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

from src.paired_dataset import PairedImageDataset, UnpairedImageDataset
from src.train_l1 import save_checkpoint
from src.turbo_runtime import load_upstream


BASE_MODEL = "stabilityai/sd-turbo"
SOURCE_PROMPT = "a synthetic driving scene"
TARGET_PROMPT = "a real photograph of a driving scene"
LOSS_WEIGHTS = {
    "gan": 0.5,
    "cycle": 1.0,
    "cycle_lpips": 10.0,
    "identity": 1.0,
    "identity_lpips": 1.0,
}


def _forward(
    runtime,
    image: torch.Tensor,
    direction: str,
    vae_encoder: nn.Module,
    unet: nn.Module,
    vae_decoder: nn.Module,
    scheduler,
    embedding: torch.Tensor,
) -> torch.Tensor:
    timesteps = torch.full(
        (image.shape[0],),
        scheduler.config.num_train_timesteps - 1,
        device=image.device,
        dtype=torch.long,
    )
    return runtime.CycleGAN_Turbo.forward_with_networks(
        image,
        direction,
        vae_encoder,
        unet,
        vae_decoder,
        scheduler,
        timesteps,
        embedding.expand(image.shape[0], -1, -1),
    )


@torch.inference_mode()
def save_validation_panel(
    path: Path,
    runtime,
    dataset: PairedImageDataset,
    device: torch.device,
    vae_encoder: nn.Module,
    unet: nn.Module,
    vae_decoder: nn.Module,
    scheduler,
    target_embedding: torch.Tensor,
) -> None:
    modules = (vae_encoder, unet, vae_decoder)
    states = [module.training for module in modules]
    for module in modules:
        module.eval()
    rows = []
    indices = sorted({0, len(dataset) // 3, 2 * len(dataset) // 3, len(dataset) - 1})
    for index in indices:
        synthetic, real, _ = dataset[index]
        prediction = _forward(
            runtime,
            synthetic[None].to(device).mul(2).sub(1),
            "a2b",
            vae_encoder,
            unet,
            vae_decoder,
            scheduler,
            target_embedding,
        )[0].cpu().add(1).mul(0.5).clamp(0, 1)
        rows.append(torch.cat((synthetic, prediction, real), dim=2))
    for module, state in zip(modules, states, strict=True):
        module.train(state)
    panel = torch.cat(rows, dim=1)
    array = panel.mul(255).byte().permute(1, 2, 0).numpy()
    temporary = path.with_suffix(".tmp.png")
    Image.fromarray(array).save(temporary)
    temporary.replace(path)


def _checkpoint(
    step: int,
    epoch: int,
    config: dict,
    unet: nn.Module,
    vae_encoder: nn.Module,
    vae_decoder: nn.Module,
    unet_modules: tuple[list[str], list[str], list[str]],
    vae_modules: list[str],
    get_peft_model_state_dict,
) -> dict:
    encoder_modules, decoder_modules, other_modules = unet_modules
    return {
        "step": step,
        "epoch": epoch,
        "config": config,
        "l_target_modules_encoder": encoder_modules,
        "l_target_modules_decoder": decoder_modules,
        "l_modules_others": other_modules,
        "rank_unet": config["lora_rank_unet"],
        "sd_encoder": get_peft_model_state_dict(
            unet, adapter_name="default_encoder"
        ),
        "sd_decoder": get_peft_model_state_dict(
            unet, adapter_name="default_decoder"
        ),
        "sd_other": get_peft_model_state_dict(
            unet, adapter_name="default_others"
        ),
        "rank_vae": config["lora_rank_vae"],
        "vae_lora_target_modules": vae_modules,
        "sd_vae_enc": vae_encoder.state_dict(),
        "sd_vae_dec": vae_decoder.state_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("manifests/pairs.csv"))
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("runs/cyclegan-turbo-seed-7"))
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--wandb-project", default="hlcv-sim2real")
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="disabled",
    )
    args = parser.parse_args()

    if args.max_steps < 1 or args.checkpoint_every < 1:
        parser.error("Steps and checkpoint interval must be positive")
    if args.batch_size < 1:
        parser.error("Batch size must be positive")
    if args.height < 64 or args.width < 64 or args.height % 8 or args.width % 8:
        parser.error("Height and width must be at least 64 and divisible by 8")
    if args.output.exists() and (
        not args.output.is_dir() or any(args.output.iterdir())
    ):
        parser.error(f"Output path is not an empty directory: {args.output}")
    if not torch.cuda.is_available():
        parser.error("CycleGAN-Turbo requires CUDA")

    import lpips
    import vision_aided_loss
    from peft.utils import get_peft_model_state_dict
    from transformers import AutoTokenizer, CLIPTextModel

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    runtime = load_upstream()

    size = (args.height, args.width)
    train_data = UnpairedImageDataset(
        args.manifest,
        "train",
        size=size,
        root=args.data_root,
        horizontal_flip=True,
    )
    val_data = PairedImageDataset(args.manifest, "val", size=size, root=args.data_root)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        generator=torch.Generator().manual_seed(args.seed),
    )

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, subfolder="tokenizer", use_fast=False
    )
    text_encoder = CLIPTextModel.from_pretrained(
        BASE_MODEL, subfolder="text_encoder"
    ).to(device)
    text_encoder.requires_grad_(False)
    scheduler = runtime.make_1step_sched()
    unet, encoder_modules, decoder_modules, other_modules = runtime.initialize_unet(
        128, return_lora_module_names=True
    )
    vae_a2b, vae_modules = runtime.initialize_vae(
        4, return_lora_module_names=True
    )
    unet.enable_xformers_memory_efficient_attention()
    unet.enable_gradient_checkpointing()
    unet.to(device)
    vae_a2b.to(device)
    vae_b2a = copy.deepcopy(vae_a2b)
    vae_encoder = runtime.VAE_encode(vae_a2b, vae_b2a=vae_b2a)
    vae_decoder = runtime.VAE_decode(vae_a2b, vae_b2a=vae_b2a)

    generator_parameters = runtime.CycleGAN_Turbo.get_traininable_params(
        unet, vae_a2b, vae_b2a
    )
    discriminator_a = vision_aided_loss.Discriminator(
        cv_type="clip", loss_type="multilevel_sigmoid", device="cuda"
    ).to(device)
    discriminator_b = vision_aided_loss.Discriminator(
        cv_type="clip", loss_type="multilevel_sigmoid", device="cuda"
    ).to(device)
    discriminator_a.cv_ensemble.requires_grad_(False)
    discriminator_b.cv_ensemble.requires_grad_(False)
    perceptual_loss = lpips.LPIPS(net="vgg").to(device).eval()
    perceptual_loss.requires_grad_(False)
    cycle_loss = nn.L1Loss()

    optimizer_generator = torch.optim.AdamW(
        generator_parameters, lr=1e-5, betas=(0.9, 0.999), weight_decay=1e-2
    )
    discriminator_parameters = list(discriminator_a.parameters()) + list(
        discriminator_b.parameters()
    )
    optimizer_discriminator = torch.optim.AdamW(
        discriminator_parameters, lr=1e-5, betas=(0.9, 0.999), weight_decay=1e-2
    )

    def encode_prompt(prompt: str) -> torch.Tensor:
        tokens = tokenizer(
            prompt,
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(device)
        return text_encoder(tokens)[0].detach()

    source_embedding = encode_prompt(SOURCE_PROMPT)
    target_embedding = encode_prompt(TARGET_PROMPT)
    del text_encoder, tokenizer
    torch.cuda.empty_cache()

    config = {
        "method": "cyclegan-turbo",
        "base_model": BASE_MODEL,
        "source_prompt": SOURCE_PROMPT,
        "target_prompt": TARGET_PROMPT,
        "learning_rate": 1e-5,
        "lora_rank_unet": 128,
        "lora_rank_vae": 4,
        **LOSS_WEIGHTS,
        **{
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    history_fields = (
        "step",
        "epoch",
        "cycle_a",
        "cycle_b",
        "gan_a",
        "gan_b",
        "identity_a",
        "identity_b",
        "discriminator_a",
        "discriminator_b",
    )
    unet_modules = (encoder_modules, decoder_modules, other_modules)

    with wandb.init(
        project=args.wandb_project,
        name=args.output.name,
        config=config,
        mode=args.wandb_mode,
        dir=args.output,
        job_type="train",
    ) as run, (args.output / "history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as history_file:
        writer = csv.DictWriter(history_file, fieldnames=history_fields)
        writer.writeheader()
        step = 0
        epoch = 0
        while step < args.max_steps:
            epoch += 1
            for source, target in train_loader:
                source = source.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)

                optimizer_generator.zero_grad(set_to_none=True)
                fake_target = _forward(
                    runtime,
                    source,
                    "a2b",
                    vae_encoder,
                    unet,
                    vae_decoder,
                    scheduler,
                    target_embedding,
                )
                recovered_source = _forward(
                    runtime,
                    fake_target,
                    "b2a",
                    vae_encoder,
                    unet,
                    vae_decoder,
                    scheduler,
                    source_embedding,
                )
                fake_source = _forward(
                    runtime,
                    target,
                    "b2a",
                    vae_encoder,
                    unet,
                    vae_decoder,
                    scheduler,
                    source_embedding,
                )
                recovered_target = _forward(
                    runtime,
                    fake_source,
                    "a2b",
                    vae_encoder,
                    unet,
                    vae_decoder,
                    scheduler,
                    target_embedding,
                )
                loss_cycle_a = (
                    cycle_loss(recovered_source, source) * LOSS_WEIGHTS["cycle"]
                    + perceptual_loss(recovered_source, source).mean()
                    * LOSS_WEIGHTS["cycle_lpips"]
                )
                loss_cycle_b = (
                    cycle_loss(recovered_target, target) * LOSS_WEIGHTS["cycle"]
                    + perceptual_loss(recovered_target, target).mean()
                    * LOSS_WEIGHTS["cycle_lpips"]
                )
                (loss_cycle_a + loss_cycle_b).backward()
                torch.nn.utils.clip_grad_norm_(generator_parameters, 10.0)
                optimizer_generator.step()

                discriminator_a.requires_grad_(False)
                discriminator_b.requires_grad_(False)
                optimizer_generator.zero_grad(set_to_none=True)
                fake_target = _forward(
                    runtime,
                    source,
                    "a2b",
                    vae_encoder,
                    unet,
                    vae_decoder,
                    scheduler,
                    target_embedding,
                )
                fake_source = _forward(
                    runtime,
                    target,
                    "b2a",
                    vae_encoder,
                    unet,
                    vae_decoder,
                    scheduler,
                    source_embedding,
                )
                loss_gan_a = (
                    discriminator_a(fake_target, for_G=True).mean()
                    * LOSS_WEIGHTS["gan"]
                )
                loss_gan_b = (
                    discriminator_b(fake_source, for_G=True).mean()
                    * LOSS_WEIGHTS["gan"]
                )
                (loss_gan_a + loss_gan_b).backward()
                torch.nn.utils.clip_grad_norm_(generator_parameters, 10.0)
                optimizer_generator.step()
                discriminator_a.requires_grad_(True)
                discriminator_b.requires_grad_(True)
                discriminator_a.cv_ensemble.requires_grad_(False)
                discriminator_b.cv_ensemble.requires_grad_(False)

                optimizer_generator.zero_grad(set_to_none=True)
                identity_target = _forward(
                    runtime,
                    target,
                    "a2b",
                    vae_encoder,
                    unet,
                    vae_decoder,
                    scheduler,
                    target_embedding,
                )
                identity_source = _forward(
                    runtime,
                    source,
                    "b2a",
                    vae_encoder,
                    unet,
                    vae_decoder,
                    scheduler,
                    source_embedding,
                )
                loss_identity_a = (
                    cycle_loss(identity_target, target) * LOSS_WEIGHTS["identity"]
                    + perceptual_loss(identity_target, target).mean()
                    * LOSS_WEIGHTS["identity_lpips"]
                )
                loss_identity_b = (
                    cycle_loss(identity_source, source) * LOSS_WEIGHTS["identity"]
                    + perceptual_loss(identity_source, source).mean()
                    * LOSS_WEIGHTS["identity_lpips"]
                )
                (loss_identity_a + loss_identity_b).backward()
                torch.nn.utils.clip_grad_norm_(generator_parameters, 10.0)
                optimizer_generator.step()

                optimizer_discriminator.zero_grad(set_to_none=True)
                loss_fake_a = (
                    discriminator_a(fake_target.detach(), for_real=False).mean()
                    * LOSS_WEIGHTS["gan"]
                )
                loss_fake_b = (
                    discriminator_b(fake_source.detach(), for_real=False).mean()
                    * LOSS_WEIGHTS["gan"]
                )
                ((loss_fake_a + loss_fake_b) * 0.5).backward()
                torch.nn.utils.clip_grad_norm_(discriminator_parameters, 10.0)
                optimizer_discriminator.step()

                optimizer_discriminator.zero_grad(set_to_none=True)
                loss_real_a = (
                    discriminator_a(target, for_real=True).mean()
                    * LOSS_WEIGHTS["gan"]
                )
                loss_real_b = (
                    discriminator_b(source, for_real=True).mean()
                    * LOSS_WEIGHTS["gan"]
                )
                ((loss_real_a + loss_real_b) * 0.5).backward()
                torch.nn.utils.clip_grad_norm_(discriminator_parameters, 10.0)
                optimizer_discriminator.step()

                step += 1
                metrics = {
                    "step": step,
                    "epoch": epoch,
                    "cycle_a": loss_cycle_a.detach().item(),
                    "cycle_b": loss_cycle_b.detach().item(),
                    "gan_a": loss_gan_a.detach().item(),
                    "gan_b": loss_gan_b.detach().item(),
                    "identity_a": loss_identity_a.detach().item(),
                    "identity_b": loss_identity_b.detach().item(),
                    "discriminator_a": (loss_fake_a + loss_real_a).detach().item(),
                    "discriminator_b": (loss_fake_b + loss_real_b).detach().item(),
                }
                writer.writerow(metrics)
                history_file.flush()
                run.log({key: value for key, value in metrics.items() if key != "step"}, step=step)
                if step == 1 or step % 10 == 0:
                    print(json.dumps(metrics), flush=True)
                if step % args.checkpoint_every == 0 or step == args.max_steps:
                    state = _checkpoint(
                        step,
                        epoch,
                        config,
                        unet,
                        vae_encoder,
                        vae_decoder,
                        unet_modules,
                        vae_modules,
                        get_peft_model_state_dict,
                    )
                    save_checkpoint(args.output / "last.pt", state)
                    save_validation_panel(
                        args.output / "validation-last.png",
                        runtime,
                        val_data,
                        device,
                        vae_encoder,
                        unet,
                        vae_decoder,
                        scheduler,
                        target_embedding,
                    )
                if step >= args.max_steps:
                    break
        run.summary.update({"final_step": step, "final_epoch": epoch})


if __name__ == "__main__":
    main()
