"""CycleGAN-Turbo adapter for the common validation evaluator."""

from pathlib import Path

import torch
from torch import nn

from src.turbo_runtime import load_upstream


class TurboGenerator(nn.Module):
    def __init__(self, checkpoint: Path, prompt: str, device: torch.device) -> None:
        super().__init__()
        if device.type != "cuda":
            raise ValueError("CycleGAN-Turbo requires CUDA")
        runtime = load_upstream()
        self.model = runtime.CycleGAN_Turbo(pretrained_path=str(checkpoint)).eval()
        self.model.unet.enable_xformers_memory_efficient_attention()
        tokens = self.model.tokenizer(
            prompt,
            max_length=self.model.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(device)
        embedding = self.model.text_encoder(tokens)[0].detach()
        self.register_buffer("prompt_embedding", embedding)
        del self.model.text_encoder
        del self.model.tokenizer

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        embedding = self.prompt_embedding.expand(image.shape[0], -1, -1)
        prediction = self.model(
            image.mul(2).sub(1),
            direction="a2b",
            caption_emb=embedding,
        )
        return prediction.add(1).mul(0.5).clamp(0, 1)


def load_turbo_generator(
    checkpoint: Path, config: dict, device: torch.device
) -> nn.Module:
    prompt = config.get("target_prompt")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError(f"CycleGAN-Turbo target prompt missing: {checkpoint}")
    return TurboGenerator(checkpoint, prompt, device)
