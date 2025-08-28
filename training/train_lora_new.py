#!/usr/bin/env python
# coding=utf-8

import os
import yaml
import torch
import logging
import argparse
import numpy as np
from pathlib import Path
from string import Template

import datasets
import diffusers
import transformers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available

from transformers import CLIPTextModel, CLIPTokenizer

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from tqdm.auto import tqdm
from PIL import Image
from torchvision import transforms

# 检查 diffusers 版本
check_min_version("0.34.0")

logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="LoRA训练脚本")
    parser.add_argument(
        "--config_path",
        type=str,
        default="config/train_config.yaml",
        help="配置文件路径",
    )
    args = parser.parse_args()
    return args

def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # 处理环境变量替换
    work_base = os.environ.get("WORK_BASE", ".")
    template_dict = {"WORK_BASE": work_base}
    
    # 递归替换模板变量
    def replace_templates(obj):
        if isinstance(obj, str) and "${" in obj:
            return Template(obj).safe_substitute(template_dict)
        elif isinstance(obj, dict):
            return {k: replace_templates(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [replace_templates(item) for item in obj]
        else:
            return obj
    
    config = replace_templates(config)
    return config

def main():
    args = parse_args()
    config = load_config(args.config_path)
    
    # 提取配置
    model_name = config["model_name"]
    dataset_config = config["dataset"]
    train_config = config["train"]
    logging_config = config["logging"]
    inference_config = config["inference"]
    work_dir = config["work_dir"]
    
    # 设置加速器
    accelerator_project_config = ProjectConfiguration(
        project_dir=logging_config["output_dir"],
        logging_dir=os.path.join(logging_config["output_dir"], "logs"),
    )
    
    accelerator = Accelerator(
        gradient_accumulation_steps=train_config["gradient_accumulation_steps"],
        mixed_precision=train_config["mixed_precision"],
        project_config=accelerator_project_config,
    )
    
    # 创建输出目录
    os.makedirs(logging_config["output_dir"], exist_ok=True)
    
    # 设置日志
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    
    # 设置随机种子
    if train_config["seed"] is not None:
        set_seed(train_config["seed"])
    
    # 加载模型组件
    tokenizer = CLIPTokenizer.from_pretrained(
        model_name,
        subfolder="tokenizer",
    )
    
    text_encoder = CLIPTextModel.from_pretrained(
        model_name,
        subfolder="text_encoder",
    )
    
    vae = AutoencoderKL.from_pretrained(
        model_name,
        subfolder="vae",
    )
    
    unet = UNet2DConditionModel.from_pretrained(
        model_name,
        subfolder="unet",
    )
    
    # 使用 PEFT 库添加 LoRA
    from peft import LoraConfig, get_peft_model
    
    # 配置 LoRA
    lora_config = LoraConfig(
        r=train_config["lora_rank"],
        lora_alpha=train_config["lora_alpha"],
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=train_config["lora_dropout"],
    )
    
    # 冻结所有 UNet 参数
    for param in unet.parameters():
        param.requires_grad = False
    
    # 应用 LoRA 到 UNet
    unet = get_peft_model(unet, lora_config)
    
    # 收集 LoRA 参数进行优化
    unet_lora_parameters = []
    for name, param in unet.named_parameters():
        if "lora" in name:
            param.requires_grad = True
            unet_lora_parameters.append(param)
    
    # 冻结 VAE 和 text encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    
    # 启用 xformers
    if train_config["enable_xformers"]:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            logger.warning("xformers is not available, skipping.")
    
    # 启用梯度检查点
    if train_config["gradient_checkpointing"]:
        unet.enable_gradient_checkpointing()
    
    # 创建噪声调度器
    noise_scheduler = DDPMScheduler.from_pretrained(model_name, subfolder="scheduler")
    
    # 加载训练数据集
    train_dataset_path = os.path.join(dataset_config["processed_root"], "train")
    logger.info(f"Loading dataset from {train_dataset_path}")
    
    # 检查目录是否存在
    if not os.path.exists(train_dataset_path):
        raise FileNotFoundError(f"训练数据集目录不存在: {train_dataset_path}")
    
    # 加载训练数据集
    train_dataset = datasets.load_from_disk(train_dataset_path)
    
    # 打印数据集列名和示例数据，方便调试
    logger.info(f"Dataset columns: {train_dataset.column_names}")
    
    # 显示一个样本，便于理解数据格式
    if len(train_dataset) > 0:
        sample = train_dataset[0]
        logger.info(f"Sample data: {sample}")
    
    def tokenize_captions(examples):
        # 使用 'text' 列作为caption
        captions = examples["text"]
        max_length = tokenizer.model_max_length
        return tokenizer(
            captions,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        ).input_ids
    
    # 图像转换
    image_transforms = transforms.Compose([
        transforms.Resize(
            dataset_config["resolution"],
            interpolation=transforms.InterpolationMode.BILINEAR,
        ),
        transforms.CenterCrop(dataset_config["resolution"]) if dataset_config["center_crop"] else transforms.RandomCrop(dataset_config["resolution"]),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    
    # 设置数据集预处理 - 正确处理HuggingFace的Image对象
    def transform_images(examples):
        # HuggingFace datasets的Image对象会自动解码为PIL图像
        # 这里直接使用解码后的PIL图像
        images = []
        for img in examples["image"]:
            # 确保图像已经转为RGB格式
            if not img.mode == "RGB":
                img = img.convert("RGB")
            images.append(img)
        
        examples["pixel_values"] = [image_transforms(image) for image in images]
        examples["input_ids"] = tokenize_captions(examples)
        
        return examples
    
    # 设置数据集预处理
    logger.info("开始处理数据集...")
    train_dataset = train_dataset.map(
        transform_images,
        batched=True,
        batch_size=4,  # 较小的批次大小，减少内存使用
        num_proc=1,    # 单进程处理，避免多进程问题
        remove_columns=["image", "text", "text_raw", "image_id"],
    )
    logger.info("数据集处理完成")
    
    # 设置数据加载器
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_config["train_batch_size"],
        shuffle=True,
        num_workers=train_config["num_workers"],
    )
    
    # 创建优化器
    optimizer = torch.optim.AdamW(
        unet_lora_parameters,
        lr=train_config["learning_rate"],
        betas=(0.9, 0.999),
        weight_decay=0.01,
        eps=1e-8,
    )
    
    # 获取学习率调度器
    lr_scheduler = get_scheduler(
        train_config["lr_scheduler"],
        optimizer=optimizer,
        num_warmup_steps=train_config["lr_warmup_steps"] * train_config["gradient_accumulation_steps"],
        num_training_steps=train_config["max_train_steps"] * train_config["gradient_accumulation_steps"],
    )
    
    # 使用 accelerator 准备模型和数据加载器
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )
    
    # 将其他模型传递到设备
    text_encoder.to(accelerator.device)
    vae.to(accelerator.device)
    
    # 准备加速器
    num_update_steps_per_epoch = len(train_dataloader)
    num_train_epochs = train_config["max_train_steps"] // num_update_steps_per_epoch + 1
    total_batch_size = train_config["train_batch_size"] * accelerator.num_processes * train_config["gradient_accumulation_steps"]
    
    logger.info("***** 开始训练 *****")
    logger.info(f"  训练步数 = {train_config['max_train_steps']}")
    logger.info(f"  梯度累积步数 = {train_config['gradient_accumulation_steps']}")
    logger.info(f"  总批次大小 = {total_batch_size}")
    
    global_step = 0
    first_epoch = 0
    
    progress_bar = tqdm(
        range(0, train_config["max_train_steps"]),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")
    
    # 训练循环
    for epoch in range(first_epoch, num_train_epochs):
        unet.train()
        train_loss = 0.0
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # 将输入转换为 latents
                latents = vae.encode(batch["pixel_values"].to(accelerator.device)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                
                # 对文本进行编码
                encoder_hidden_states = text_encoder(batch["input_ids"].to(accelerator.device))[0]
                
                # 采样噪声
                noise = torch.randn_like(latents)
                if train_config["noise_offset"] > 0:
                    noise = noise + train_config["noise_offset"] * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1),
                        device=latents.device,
                    )
                
                # 获取时间步
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
                ).long()
                
                # 添加噪声
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # 预测噪声
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                
                # 获取目标噪声
                target = noise
                
                # 计算 MSE 损失
                loss = torch.nn.functional.mse_loss(model_pred.float(), target.float(), reduction="mean")
                
                # 反向传播
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet_lora_parameters, 1.0)
                
                # 更新参数
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # 记录损失
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
                train_loss += loss.detach().item()
                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                
                if global_step % logging_config["log_steps"] == 0:
                    accelerator.log({"train_loss": train_loss / logging_config["log_steps"]}, step=global_step)
                    train_loss = 0.0
                
                if global_step % logging_config["save_steps"] == 0:
                    if accelerator.is_main_process:
                        # 取消包装模型
                        unwrapped_unet = accelerator.unwrap_model(unet)
                        
                        # 保存 LoRA 权重
                        ckpt_dir = os.path.join(logging_config["output_dir"], f"checkpoint-{global_step}")
                        os.makedirs(ckpt_dir, exist_ok=True)
                        
                        # 使用 PEFT 的方式保存 LoRA 权重
                        unwrapped_unet.save_pretrained(ckpt_dir)
                        
                        # 保存 Pipeline
                        pipeline = StableDiffusionPipeline.from_pretrained(
                            model_name,
                            unet=unwrapped_unet,
                            text_encoder=text_encoder,
                            vae=vae,
                            tokenizer=tokenizer,
                        )
                        
                        # 生成样本图像进行测试
                        pipeline.to(accelerator.device)
                        pipeline.set_progress_bar_config(disable=True)
                        
                        prompt = "a photo of a cat"
                        generator = torch.Generator(device=accelerator.device).manual_seed(inference_config["seed"])
                        images = pipeline(
                            prompt=prompt,
                            num_inference_steps=inference_config["num_inference_steps"],
                            guidance_scale=inference_config["guidance_scale"],
                            generator=generator,
                        ).images
                        
                        images[0].save(os.path.join(ckpt_dir, "sample.png"))
                        
                        # 创建并保存示例
                        prompt = "a photo of a dog"
                        generator = torch.Generator(device=accelerator.device).manual_seed(inference_config["seed"])
                        images = pipeline(
                            prompt=prompt,
                            num_inference_steps=inference_config["num_inference_steps"],
                            guidance_scale=inference_config["guidance_scale"],
                            generator=generator,
                        ).images
                        
                        images[0].save(os.path.join(ckpt_dir, "sample_dog.png"))
                        
                        logger.info(f"已保存模型到 {ckpt_dir}")
                
                if global_step >= train_config["max_train_steps"]:
                    break
    
    # 完成训练后保存最终模型
    if accelerator.is_main_process:
        # 取消包装模型
        unwrapped_unet = accelerator.unwrap_model(unet)
        
        # 保存 LoRA 权重
        final_dir = os.path.join(logging_config["output_dir"], "final_model")
        os.makedirs(final_dir, exist_ok=True)
        
        # 使用 PEFT 的方式保存 LoRA 权重
        unwrapped_unet.save_pretrained(final_dir)
        
        # 保存 Pipeline
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_name,
            unet=unwrapped_unet,
            text_encoder=text_encoder,
            vae=vae,
            tokenizer=tokenizer,
        )
        
        logger.info(f"训练完成！最终模型已保存到 {final_dir}")

if __name__ == "__main__":
    main()