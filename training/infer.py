import os
import argparse
import torch
from diffusers import StableDiffusionPipeline
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    ap = argparse.ArgumentParser()
    work_base = os.environ.get("WORK_BASE", ".")
    default_lora = os.path.join(work_base, "lora_sd15_no_adverbs_a100/final_model")
    default_output = os.path.join(work_base, "outputs/sample.png")
    
    ap.add_argument("--model_name", type=str, default="runwayml/stable-diffusion-v1-5")
    ap.add_argument("--lora_path", type=str, default=default_lora, 
                    help="Path to LoRA weights file or directory")
    ap.add_argument("--prompt", type=str, default="a photo of a small red car on the street")
    ap.add_argument("--negative_prompt", type=str, default="")
    ap.add_argument("--num_inference_steps", type=int, default=30)
    ap.add_argument("--guidance_scale", type=float, default=7.5)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--output", type=str, default=default_output)
    
    return ap.parse_args()

def main():
    args = parse_args()
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # 创建基础管道
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
        safety_checker=None,
        requires_safety_checker=False
    )
    
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            logger.warning("xformers不可用，跳过")

    # 检查LoRA权重路径
    lora_path = args.lora_path
    
    # 检查路径是文件还是目录
    if os.path.isdir(lora_path):
        # 如果是目录，查找可能的权重文件
        possible_files = [
            os.path.join(lora_path, "pytorch_lora_weights.bin"),
            os.path.join(lora_path, "lora_weights.safetensors"),
            os.path.join(lora_path, "lora_weights.bin"),
        ]
        
        for file_path in possible_files:
            if os.path.exists(file_path):
                lora_path = file_path
                break
        else:
            logger.error(f"在目录 {lora_path} 中找不到LoRA权重文件")
            lora_path = None
    
    # 尝试加载LoRA权重
    if lora_path and os.path.exists(lora_path):
        try:
            logger.info(f"尝试加载LoRA权重: {lora_path}")
            
            # 使用Diffusers 0.34.0的load_lora_weights方法
            if hasattr(pipe, 'load_lora_weights'):
                try:
                    # 如果是目录，直接传递目录路径
                    if os.path.isdir(lora_path):
                        pipe.load_lora_weights(lora_path)
                    else:
                        # 如果是文件，传递目录和文件名
                        pipe.load_lora_weights(os.path.dirname(lora_path), weight_name=os.path.basename(lora_path))
                    logger.info("使用load_lora_weights方法成功加载LoRA权重")
                except Exception as e:
                    logger.warning(f"load_lora_weights方法失败: {e}")
                    
                    # 回退到手动加载
                    logger.info("尝试手动加载LoRA权重...")
                    lora_weights = torch.load(lora_path, map_location="cpu")
                    
                    # 检查权重格式并加载到UNet
                    if any("lora" in key.lower() for key in lora_weights.keys()):
                        # 这是LoRA权重
                        pipe.unet.load_state_dict(lora_weights, strict=False)
                    else:
                        # 这可能是完整的UNet权重，但只包含LoRA层
                        pipe.unet.load_state_dict(lora_weights, strict=False)
                    
                    logger.info("手动加载LoRA权重成功")
            else:
                # 对于较旧版本的Diffusers
                logger.info("使用load_attn_procs加载LoRA权重")
                pipe.unet.load_attn_procs(lora_path)
                
        except Exception as e:
            logger.error(f"加载LoRA权重失败: {e}")
            logger.info("将继续使用基础模型（无LoRA）进行推理")
    else:
        logger.info("未提供LoRA权重路径或路径不存在，将使用基础模型")
    
    # 设置随机种子
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    
    # 生成图像
    logger.info(f"生成提示: {args.prompt}")
    
    # 创建生成器
    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=pipe.device).manual_seed(args.seed)
    
    # 生成图像
    image = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt if args.negative_prompt else None,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generator=generator
    ).images[0]

    # 保存图像
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    image.save(args.output)
    logger.info(f"图像已保存到 {args.output}")

if __name__ == "__main__":
    main()