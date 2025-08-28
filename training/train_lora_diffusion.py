import os
import math
import time
import argparse
import random
import numpy as np
from dataclasses import dataclass
from PIL import Image
import gc

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import load_from_disk
from torchvision import transforms

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed

from diffusers import StableDiffusionPipeline, DDPMScheduler
from diffusers.optimization import get_scheduler

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
    num_workers: int = 1  # 降低为1，避免多进程问题

    lr_scheduler: str = "cosine"
    lr_warmup_steps: int = 0
    noise_offset: float = 0.0

    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0

    log_steps: int = 10
    save_steps: int = 50
    output_dir: str = "./outputs/lora_sd15_no_adverbs"

def check_versions():
    """显示关键库版本信息"""
    import diffusers
    import torch
    
    print("\n===== 环境版本信息 =====")
    print(f"diffusers版本: {diffusers.__version__}")
    print(f"PyTorch版本: {torch.__version__}")
    try:
        import peft
        print(f"PEFT版本: {peft.__version__}")
    except ImportError:
        print("PEFT未安装")
    print("=========================\n")

def parse_args():
    import yaml
    import os
    import re
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config/train_config.yaml")
    args = ap.parse_args()
    
    # ===== 调试信息 =====
    print("\n===== 环境变量调试信息 =====")
    print(f"WORK_BASE = {os.environ.get('WORK_BASE', '未设置')}")
    print(f"当前工作目录 = {os.getcwd()}")
    # ====================
    
    # 定义环境变量替换函数
    def replace_env_vars(value):
        if isinstance(value, str):
            # 替换${VAR}格式的环境变量
            pattern = r'\${([^}]+)}'
            def replace_var(match):
                var_name = match.group(1)
                env_value = os.environ.get(var_name)
                if env_value is None:
                    print(f"警告: 环境变量 '{var_name}' 未找到")
                    return match.group(0)  # 保持原样
                return env_value
            return re.sub(pattern, replace_var, value)
        elif isinstance(value, dict):
            return {k: replace_env_vars(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [replace_env_vars(item) for item in value]
        return value
    
    # 加载原始配置
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    # ===== 调试信息 =====
    print("\n===== 原始配置信息 =====")
    print(f"model_name: {cfg.get('model_name')}")
    print(f"work_dir: {cfg.get('work_dir')}")
    data_cfg = cfg.get("dataset", {})
    print(f"dataset.raw_root: {data_cfg.get('raw_root')}")
    print(f"dataset.processed_root: {data_cfg.get('processed_root')}")
    # ====================
    
    # 替换所有环境变量引用
    cfg = replace_env_vars(cfg)
    
    # ===== 调试信息 =====
    print("\n===== 环境变量替换后的配置 =====")
    print(f"model_name: {cfg.get('model_name')}")
    print(f"work_dir: {cfg.get('work_dir')}")
    data_cfg = cfg.get("dataset", {})
    print(f"dataset.raw_root: {data_cfg.get('raw_root')}")
    print(f"dataset.processed_root: {data_cfg.get('processed_root')}")
    # ====================
    
    # 安全获取可选项
    train_cfg = cfg.get("train", {})
    data_cfg = cfg.get("dataset", {})
    log_cfg = cfg.get("logging", {})
    
    # 构建完整路径
    dataset_path = os.path.join(data_cfg["processed_root"], data_cfg["split"])
    
    # ===== 调试信息 =====
    print(f"\n最终数据集路径: {dataset_path}")
    print(f"该路径是否存在: {os.path.exists(dataset_path)}")
    if not os.path.exists(dataset_path):
        parent_dir = os.path.dirname(dataset_path)
        print(f"父目录是否存在: {os.path.exists(parent_dir)}")
        if os.path.exists(parent_dir):
            print(f"父目录内容: {os.listdir(parent_dir)}")
    print("===== 调试信息结束 =====\n")
    # ====================

    tc = TrainConfig(
        model_name=cfg["model_name"],
        work_dir=cfg["work_dir"],
        dataset_path=dataset_path,
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
        num_workers=1,  # 强制设为1，避免多进程问题

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
        print(f"正在加载数据集: {hf_path}")
        try:
            self.ds = load_from_disk(hf_path)
            print(f"数据集加载成功，共 {len(self.ds)} 条记录")
        except Exception as e:
            print(f"数据集加载失败: {str(e)}")
            # 尝试检查父目录
            parent_dir = os.path.dirname(hf_path)
            if os.path.exists(parent_dir):
                print(f"父目录 {parent_dir} 内容: {os.listdir(parent_dir)}")
            raise
            
        self.preproc = make_transforms(resolution, center_crop)
        # 保存一个可用样本索引列表，在__getitem__中如果遇到错误可跳过问题样本
        self.valid_indices = list(range(len(self.ds)))
        self.resolution = resolution

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        try:
            item = self.ds[real_idx]
            img = item["image"]
            if not isinstance(img, Image.Image):
                img = Image.open(item["image"]).convert("RGB")
                
            # 确保图像有效
            if img.width < 10 or img.height < 10:
                raise ValueError(f"图像太小: {img.width}x{img.height}")
            
            # 应用预处理
            pixel_values = self.preproc(img)
            
            # 确保tensor形状正确
            if pixel_values.shape != (3, self.resolution, self.resolution):
                raise ValueError(f"预处理后图像形状错误: {pixel_values.shape}")
                
            text = item["text"]
            if not isinstance(text, str) or len(text) < 2:
                raise ValueError(f"文本无效: {text}")
                
            return {"pixel_values": pixel_values, "text": text}
            
        except Exception as e:
            # 如果有错误，尝试返回另一个有效样本
            fallback_idx = random.randint(0, len(self.valid_indices)-1)
            print(f"样本 {real_idx} 处理出错: {e}，使用替代样本 {self.valid_indices[fallback_idx]}")
            if fallback_idx != idx:
                return self.__getitem__(fallback_idx)
            else:
                # 创建一个安全的默认返回值
                blank_img = torch.zeros((3, self.resolution, self.resolution))
                return {"pixel_values": blank_img, "text": "a blank image"}

def make_lora_linear(layer, rank=4, alpha=1.0):
    """
    手动将标准线性层转换为LoRA增强层 - 内存优化版本
    """
    import math
    import torch.nn as nn

    # 保存原始前向传播
    original_forward = layer.forward
    # 创建LoRA层
    if hasattr(layer, 'in_features') and hasattr(layer, 'out_features'):
        in_dim, out_dim = layer.in_features, layer.out_features
        lora_down = nn.Linear(in_dim, rank, bias=False)
        lora_up = nn.Linear(rank, out_dim, bias=False)
        
        # 使用正确的缩放初始化
        nn.init.normal_(lora_down.weight, std=1 / rank)
        nn.init.zeros_(lora_up.weight)
        
        # 设置为需要梯度
        lora_down.weight.requires_grad_(True)
        lora_up.weight.requires_grad_(True)
        # 冻结原始权重
        layer.weight.requires_grad_(False)
        if hasattr(layer, 'bias') and layer.bias is not None:
            layer.bias.requires_grad_(False)
        
        # 新的前向传播，添加LoRA贡献
        def lora_forward(x):
            # 内存优化：不保留中间计算的梯度
            with torch.set_grad_enabled(True):
                lora_output = lora_up(lora_down(x)) * alpha
            return original_forward(x) + lora_output
        
        # 替换前向传播
        layer.forward = lora_forward
        # 保存LoRA组件以便以后访问
        layer.lora_down = lora_down
        layer.lora_up = lora_up
        layer.lora_alpha = alpha
        
        return layer
    return layer

def add_lora_to_unet(unet, rank: int, alpha: int, dropout: float = 0.0):
    """
    使用手动方法为UNet添加LoRA - 针对关键层，减少参数量
    """
    print(f"正在为UNet添加轻量级LoRA (rank={rank}, alpha={alpha})...")
    
    # 提取所有注意力模块中的线性层
    lora_layers = []
    
    # 我们只关注部分关键层，减少训练参数量
    key_patterns = [
        "to_q",      # 查询投影
        # "to_k",      # 键投影 - 可以跳过
        # "to_v",      # 值投影 - 可以跳过
        "to_out.0"   # 输出投影
    ]
    
    # 访问注意力模块中的关键层
    for name, module in unet.named_modules():
        if any(pattern in name for pattern in key_patterns):
            if hasattr(module, "weight"):
                print(f"添加LoRA到: {name}")
                make_lora_linear(module, rank=rank, alpha=alpha)
                lora_layers.append(name)
    
    # 验证是否找到了层
    if not lora_layers:
        raise ValueError("没有找到适合添加LoRA的层")
    
    # 获取可训练参数数量
    trainable_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    print(f"LoRA添加成功! 找到{len(lora_layers)}个层，可训练参数: {trainable_params:,}")
    
    # 为了方便后续保存，添加辅助方法
    def gather_lora_weights():
        """收集所有LoRA权重用于保存"""
        state_dict = {}
        for name, module in unet.named_modules():
            if hasattr(module, "lora_up") and hasattr(module, "lora_down"):
                state_dict[f"{name}.lora_up.weight"] = module.lora_up.weight.data
                state_dict[f"{name}.lora_down.weight"] = module.lora_down.weight.data
                state_dict[f"{name}.alpha"] = torch.tensor([module.lora_alpha])
        return state_dict
    
    # 添加辅助方法到UNet
    unet.gather_lora_weights = gather_lora_weights
    
    return unet

def main():
    # 检查库版本
    check_versions()
    
    cfg = parse_args()
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.work_dir, exist_ok=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision=cfg.mixed_precision,
        project_config=ProjectConfiguration(project_dir=cfg.work_dir, automatic_checkpoint_naming=True)
    )
    set_seed(cfg.seed)

    # 数据 - 使用单worker，避免多进程问题
    train_dataset = CocoNoAdvDataset(cfg.dataset_path, cfg.resolution, cfg.center_crop)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        num_workers=cfg.num_workers, # 使用1个worker更安全
        pin_memory=True,
        persistent_workers=False,  # 关闭持久工作进程
        drop_last=True,  # 丢弃不完整的最后一个batch
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
                print(f"无法启用 Xformers: {e}")

    # 冻结 VAE 与 Text Encoder（不动 CLIP）
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)

    # 可选：UNet gradient checkpointing
    if cfg.gradient_checkpointing and hasattr(pipe.unet, "enable_gradient_checkpointing"):
        pipe.unet.enable_gradient_checkpointing()

    # 为 UNet 添加 LoRA 适配器 - 使用轻量级版本
    add_lora_to_unet(pipe.unet, rank=cfg.lora_rank, alpha=cfg.lora_alpha, dropout=cfg.lora_dropout)
    
    # 仅获取可训练参数
    trainable_params = [p for p in pipe.unet.parameters() if p.requires_grad]
    if accelerator.is_main_process:
        total_trainable = sum(p.numel() for p in trainable_params)
        print(f"可训练 UNet (LoRA) 参数: {total_trainable/1e6:.2f}M")

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

    # 主训练循环
    for epoch in range(10**6):
        for batch_idx, batch in enumerate(train_dataloader):
            # 定期清理CUDA缓存
            if batch_idx % 50 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                
            t0 = time.time()
            with accelerator.accumulate(pipe.unet):
                try:
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
                    
                except Exception as e:
                    print(f"处理批次 {batch_idx} 时出错: {e}")
                    continue

            global_step += 1
            step_time = time.time() - t0
            step_times.append(step_time)

            if accelerator.is_main_process and (global_step % cfg.log_steps == 0 or global_step == 1):
                avg_time = sum(step_times[-cfg.log_steps:]) / min(len(step_times), cfg.log_steps)
                print(f"步骤 {global_step}/{cfg.max_train_steps} 损失={loss.item():.4f} 学习率={lr_scheduler.get_last_lr()[0]:.2e} 步骤耗时={avg_time:.2f}秒")

            if accelerator.is_main_process and (global_step % cfg.save_steps == 0 or global_step == cfg.max_train_steps):
                save_dir = os.path.join(cfg.output_dir, f"checkpoint-{global_step}")
                os.makedirs(save_dir, exist_ok=True)
                
                # 获取未包装的 UNet
                unet = accelerator.unwrap_model(pipe.unet)
                
                # 使用我们自定义的方法收集LoRA权重
                lora_state_dict = unet.gather_lora_weights()
                
                # 保存权重
                torch.save(lora_state_dict, os.path.join(save_dir, "lora_weights.bin"))
                
                # 同时保存为safetensors格式（如果可能）
                try:
                    save_file(lora_state_dict, os.path.join(save_dir, "lora_weights.safetensors"))
                except Exception as e:
                    print(f"无法保存为safetensors格式: {e}")
                
                print(f"保存 LoRA 权重到: {save_dir}")

            if global_step >= cfg.max_train_steps:
                break
        if global_step >= cfg.max_train_steps:
            break

    if accelerator.is_main_process:
        print("训练完成。")
        if len(step_times) > 0:
            avg_step_time = sum(step_times) / len(step_times)
            print(f"在 {len(step_times)} 步骤中的平均步骤耗时: {avg_step_time:.2f}秒")

if __name__ == "__main__":
    main()