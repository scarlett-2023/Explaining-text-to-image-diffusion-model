import os
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_from_disk
from diffusers import (
    StableDiffusionPipeline,
    UNet2DConditionModel,
    DDPMScheduler,
    AutoencoderKL,
)
from transformers import CLIPTextModel, CLIPTokenizer
from accelerate import Accelerator
from tqdm.auto import tqdm
import logging
from torchvision import transforms

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 加载配置文件
with open("config/train_config.yaml", "r") as f:
    config = yaml.safe_load(f)

# 设置工作目录
work_base = os.environ.get("WORK_BASE", ".")
output_dir = os.path.join(work_base, config["logging"]["output_dir"].split("/")[-1])
os.makedirs(output_dir, exist_ok=True)

# 初始化accelerator
accelerator = Accelerator(
    mixed_precision=config["train"]["mixed_precision"],
    gradient_accumulation_steps=config["train"]["gradient_accumulation_steps"],
)

# 加载模型和调度器
model_id = config["model_name"]
noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")

# 冻结模型参数
vae.requires_grad_(False)
text_encoder.requires_grad_(False)

# 启用梯度检查点
if config["train"]["gradient_checkpointing"]:
    unet.enable_gradient_checkpointing()

# 使用PEFT库添加LoRA层
try:
    from peft import LoraConfig, get_peft_model
    
    # 配置LoRA
    lora_config = LoraConfig(
        r=config["train"]["lora_rank"],
        lora_alpha=config["train"]["lora_alpha"],
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        lora_dropout=config["train"]["lora_dropout"],
        bias="none",
    )
    
    # 应用LoRA到UNet
    unet = get_peft_model(unet, lora_config)
    logger.info("成功使用PEFT库添加LoRA层")
    
except ImportError as e:
    logger.error(f"PEFT库不可用: {e}")
    raise

# 获取所有需要训练的参数（LoRA层）
lora_params = []
for name, param in unet.named_parameters():
    if "lora" in name.lower():
        lora_params.append(param)
        param.requires_grad = True
    else:
        param.requires_grad = False

logger.info(f"可训练的LoRA参数数量: {len(lora_params)}")

# 优化器
optimizer = torch.optim.AdamW(
    lora_params,
    lr=config["train"]["learning_rate"],
    betas=(0.9, 0.999),
    weight_decay=1e-2,
    eps=1e-08,
)

# 加载数据集
dataset_path = os.path.join(work_base, "data/processed/coco_no_adverbs/train")
try:
    dataset = load_from_disk(dataset_path)
    logger.info(f"成功加载数据集从: {dataset_path}")
    
    # 检查数据集结构
    logger.info(f"数据集特征: {dataset.features}")
    if len(dataset) > 0:
        sample = dataset[0]
        logger.info(f"样本键: {list(sample.keys())}")
        
except Exception as e:
    logger.error(f"加载数据集失败: {e}")
    raise

# 定义图像预处理转换
image_transforms = transforms.Compose([
    transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop((512, 512)) if config["dataset"]["center_crop"] else transforms.Lambda(lambda x: x),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

def collate_fn(examples):
    """
    根据您的数据集结构定制的collate函数
    """
    batch = {}
    
    # 处理图像数据
    images = [example["image"] for example in examples]
    
    # 将PIL图像转换为张量并应用预处理
    pixel_values = []
    for img in images:
        # 如果图像已经是张量，直接使用
        if isinstance(img, torch.Tensor):
            pixel_values.append(img)
        else:
            # 如果是PIL图像，应用转换
            pixel_values.append(image_transforms(img.convert("RGB")))
    
    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    batch["pixel_values"] = pixel_values
    
    # 处理文本数据 - 使用处理后的文本（去除了副词）
    texts = [example["text"] for example in examples]
    
    # 分词处理
    inputs = tokenizer(
        texts, 
        max_length=tokenizer.model_max_length, 
        padding="max_length", 
        truncation=True, 
        return_tensors="pt"
    )
    batch["input_ids"] = inputs.input_ids
    
    return batch

# 创建数据加载器
train_dataloader = DataLoader(
    dataset, 
    batch_size=config["train"]["train_batch_size"], 
    shuffle=True,
    num_workers=min(config["train"]["num_workers"], 4),  # 限制worker数量以避免系统问题
    collate_fn=collate_fn
)

# 学习率调度器
from diffusers.optimization import get_cosine_schedule_with_warmup
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config["train"]["lr_warmup_steps"],
    num_training_steps=config["train"]["max_train_steps"],
)

# 准备训练
unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
    unet, optimizer, train_dataloader, lr_scheduler
)
text_encoder.to(accelerator.device)
vae.to(accelerator.device)

# 启用xformers（如果可用）
if config["train"]["enable_xformers"]:
    try:
        import xformers.ops
        unet.enable_xformers_memory_efficient_attention()
        logger.info("已启用xformers内存高效注意力")
    except ImportError:
        logger.warning("xformers不可用，跳过")

# 训练循环
global_step = 0
progress_bar = tqdm(
    range(0, config["train"]["max_train_steps"]),
    disable=not accelerator.is_local_main_process,
)

unet.train()
for step, batch in enumerate(train_dataloader):
    if global_step >= config["train"]["max_train_steps"]:
        break

    with accelerator.accumulate(unet):
        # 将图像转换为潜变量
        latents = vae.encode(batch["pixel_values"].to(accelerator.device)).latent_dist.sample()
        latents = latents * vae.config.scaling_factor
        
        # 采样噪声
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        
        # 采样时间步
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
        ).long()
        
        # 添加噪声到潜变量
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        
        # 获取文本嵌入
        encoder_hidden_states = text_encoder(batch["input_ids"].to(accelerator.device))[0]
        
        # 预测噪声
        model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
        
        # 计算损失
        loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
        
        # 反向传播
        accelerator.backward(loss)
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(lora_params, 1.0)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
    
    # 更新进度条
    if accelerator.sync_gradients:
        progress_bar.update(1)
        global_step += 1
        
        # 记录日志
        if global_step % config["logging"]["log_steps"] == 0:
            accelerator.log({"loss": loss.detach().item()}, step=global_step)
            progress_bar.set_postfix(loss=loss.detach().item())
            logger.info(f"步骤 {global_step}, 损失: {loss.detach().item():.4f}")
        
        # 保存检查点
        if global_step % config["logging"]["save_steps"] == 0:
            if accelerator.is_main_process:
                save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                os.makedirs(save_path, exist_ok=True)
                
                # 保存LoRA权重
                unwrapped_unet = accelerator.unwrap_model(unet)
                lora_state_dict = {
                    k: v for k, v in unwrapped_unet.state_dict().items() 
                    if "lora" in k.lower()
                }
                torch.save(lora_state_dict, os.path.join(save_path, "pytorch_lora_weights.bin"))
                
                logger.info(f"检查点已保存到步骤 {global_step}")

# 保存最终模型
if accelerator.is_main_process:
    final_save_path = os.path.join(output_dir, "final_model")
    os.makedirs(final_save_path, exist_ok=True)
    
    # 保存LoRA权重
    unwrapped_unet = accelerator.unwrap_model(unet)
    lora_state_dict = {
        k: v for k, v in unwrapped_unet.state_dict().items() 
        if "lora" in k.lower()
    }
    torch.save(lora_state_dict, os.path.join(final_save_path, "pytorch_lora_weights.bin"))
    
    logger.info(f"训练完成! LoRA权重已保存到 {final_save_path}")