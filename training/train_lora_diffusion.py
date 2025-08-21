import os
import math
import time
import argparse
import random
import numpy as np
from dataclasses import dataclass
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import load_from_disk
from torchvision import transforms

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed

from diffusers import StableDiffusionPipeline, DDPMScheduler
from diffusers.optimization import get_scheduler
from diffusers.models.attention_processor import LoRAAttnProcessor

from safetensors.torch import save_file

@dataclass
class TrainConfig:
    model_name: str
    work_dir: str
    dataset_path: str
    resolution: int = 512
    center_crop: bool = True

    seed: int = 42
    max_train_steps: int = 100
    learning_rate: float = 5e-6
    train_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    mixed_precision: str = "fp16"    # "no" / "fp16" / "bf16"
    enable_xformers: bool = True
    gradient_checkpointing: bool = False
    num_workers: int = 4

    lr_scheduler: str = "cosine"
    lr_warmup_steps: int = 0
    noise_offset: float = 0.0

    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0

    log_steps: int = 10
    save_steps: int = 50
    output_dir: str = "./outputs/lora_sd15_no_adverbs"

def parse_args():
    import yaml
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config/train_config.yaml")
    args = ap.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 安全获取可选项
    train_cfg = cfg.get("train", {})
    data_cfg = cfg.get("dataset", {})
    log_cfg = cfg.get("logging", {})

    tc = TrainConfig(
        model_name=cfg["model_name"],
        work_dir=cfg["work_dir"],
        dataset_path=os.path.join(data_cfg["processed_root"], data_cfg["split"]),
        resolution=data_cfg.get("resolution", 512),
        center_crop=data_cfg.get("center_crop", True),

        seed=train_cfg.get("seed", 42),
        max_train_steps=train_cfg.get("max_train_steps", 100),
        learning_rate=train_cfg.get("learning_rate", 5e-6),
        train_batch_size=train_cfg.get("train_batch_size", 8),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 1),
        mixed_precision=train_cfg.get("mixed_precision", "fp16"),
        enable_xformers=train_cfg.get("enable_xformers", True),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", False),
        num_workers=train_cfg.get("num_workers", 4),

        lr_scheduler=train_cfg.get("lr_scheduler", "cosine"),
        lr_warmup_steps=train_cfg.get("lr_warmup_steps", 0),
        noise_offset=train_cfg.get("noise_offset", 0.0),

        lora_rank=train_cfg.get("lora_rank", 8),
        lora_alpha=train_cfg.get("lora_alpha", 16),
        lora_dropout=train_cfg.get("lora_dropout", 0.0),

        log_steps=log_cfg.get("log_steps", 10),
        save_steps=log_cfg.get("save_steps", 50),
        output_dir=log_cfg.get("output_dir", "./outputs/lora_sd15_no_adverbs"),
    )
    return tc

def make_transforms(resolution: int, center_crop: bool):
    crop = transforms.CenterCrop(resolution) if center_crop else transforms.RandomCrop(resolution)
    return transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
        crop,
        transforms.ToTensor(),  # [0,1]
        transforms.Normalize([0.5], [0.5])  # [-1,1]
    ])

class CocoNoAdvDataset(torch.utils.data.Dataset):
    def __init__(self, hf_path: str, resolution: int, center_crop: bool):
        self.ds = load_from_disk(hf_path)
        self.preproc = make_transforms(resolution, center_crop)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        img = item["image"]
        if not isinstance(img, Image.Image):
            img = Image.open(item["image"]).convert("RGB")
        pixel_values = self.preproc(img)  # (3,H,W) in [-1,1]
        text = item["text"]               # 去副词文本
        return {"pixel_values": pixel_values, "text": text}

def add_lora_to_unet(unet, rank: int, alpha: int, dropout: float = 0.0):
    for name, module in unet.named_modules():
        if hasattr(module, "set_attn_processor"):
            module.set_attn_processor(LoRAAttnProcessor(hidden_size=module.to_q.in_features, rank=rank))

def main():
    cfg = parse_args()
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.work_dir, exist_ok=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision=cfg.mixed_precision,
        project_config=ProjectConfiguration(project_dir=cfg.work_dir, automatic_checkpoint_naming=True)
    )
    set_seed(cfg.seed)

    # 数据
    train_dataset = CocoNoAdvDataset(cfg.dataset_path, cfg.resolution, cfg.center_crop)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
    )

    # 模型
    pipe = StableDiffusionPipeline.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.bfloat16 if cfg.mixed_precision == "bf16" else (torch.float16 if cfg.mixed_precision == "fp16" else torch.float32),
        safety_checker=None,
    )
    pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

    if cfg.enable_xformers:
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception as e:
            if accelerator.is_main_process:
                print(f"Xformers not enabled: {e}")

    # 冻结 VAE 与 Text Encoder（不动 CLIP）
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)

    # 可选：UNet gradient checkpointing
    if cfg.gradient_checkpointing and hasattr(pipe.unet, "enable_gradient_checkpointing"):
        pipe.unet.enable_gradient_checkpointing()

    # 仅在 UNet 上添加 LoRA
    add_lora_to_unet(pipe.unet, rank=cfg.lora_rank, alpha=cfg.lora_alpha)
    trainable_params = [p for p in pipe.unet.parameters() if p.requires_grad]
    if accelerator.is_main_process:
        total_trainable = sum(p.numel() for p in trainable_params)
        print(f"Trainable UNet (LoRA) params: {total_trainable/1e6:.2f}M")

    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.learning_rate)

    # 学习率调度
    lr_scheduler = get_scheduler(
        name=cfg.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.lr_warmup_steps,
        num_training_steps=cfg.max_train_steps,
    )

    pipe.unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        pipe.unet, optimizer, train_dataloader, lr_scheduler
    )

    # 训练循环（仅 diff loss：MSE(noise_pred, noise)）
    vae_dtype = torch.bfloat16 if cfg.mixed_precision == "bf16" else (torch.float16 if cfg.mixed_precision == "fp16" else torch.float32)
    pipe.vae.to(accelerator.device, dtype=vae_dtype)
    pipe.text_encoder.to(accelerator.device, dtype=vae_dtype)

    step_times = []
    global_step = 0
    pipe.unet.train()

    tokenizer = pipe.tokenizer
    vae = pipe.vae
    text_encoder = pipe.text_encoder
    noise_scheduler = pipe.scheduler
    scale_factor = 0.18215  # StableDiffusion VAE latent scaling

    for epoch in range(10**6):
        for batch in train_dataloader:
            t0 = time.time()
            with accelerator.accumulate(pipe.unet):
                # 准备图像 latents
                pixel_values = batch["pixel_values"].to(accelerator.device, dtype=vae_dtype)
                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample() * scale_factor

                # 采样 timestep 与噪声
                noise = torch.randn_like(latents)
                if cfg.noise_offset > 0:
                    noise += cfg.noise_offset * torch.randn(latents.shape[0], latents.shape[1], 1, 1, device=latents.device, dtype=latents.dtype)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # 文本条件（不训练 text encoder，仅前向）
                text_inputs = tokenizer(
                    batch["text"],
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt"
                )
                input_ids = text_inputs.input_ids.to(accelerator.device)
                attention_mask = text_inputs.attention_mask.to(accelerator.device)
                with torch.no_grad():
                    encoder_hidden_states = text_encoder(input_ids, attention_mask=attention_mask)[0]

                # 预测噪声
                noise_pred = pipe.unet(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states).sample

                # 仅用 diff loss
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            global_step += 1
            step_time = time.time() - t0
            step_times.append(step_time)

            if accelerator.is_main_process and (global_step % cfg.log_steps == 0 or global_step == 1):
                avg_time = sum(step_times[-cfg.log_steps:]) / min(len(step_times), cfg.log_steps)
                print(f"step {global_step}/{cfg.max_train_steps} loss={loss.item():.4f} lr={lr_scheduler.get_last_lr()[0]:.2e} step_time={avg_time:.2f}s")

            if accelerator.is_main_process and (global_step % cfg.save_steps == 0 or global_step == cfg.max_train_steps):
                save_dir = os.path.join(cfg.output_dir, f"checkpoint-{global_step}")
                os.makedirs(save_dir, exist_ok=True)
                unet = accelerator.unwrap_model(pipe.unet)
                unet.save_attn_procs(save_dir)
                print(f"Saved LoRA UNet attn processors to: {save_dir}")

            if global_step >= cfg.max_train_steps:
                break
        if global_step >= cfg.max_train_steps:
            break

    if accelerator.is_main_process:
        print("Training finished.")
        if len(step_times) > 0:
            avg_step_time = sum(step_times) / len(step_times)
            print(f"Average step time over {len(step_times)} steps: {avg_step_time:.2f}s")

if __name__ == "__main__":
    main()