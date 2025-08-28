import torch
from diffusers import StableDiffusionPipeline
import numpy as np
from PIL import Image
import os

def compare_lora_effect():
    # 加载基础模型
    base_pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False
    ).to("cuda")
    
    # 加载LoRA模型
    lora_pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False
    ).to("cuda")
    
    # 加载LoRA权重
    lora_path = "/work/nvme/bepi/sxie7/training/lora_sd15_no_adverbs_a100/final_model/pytorch_lora_weights.bin"
    lora_weights = torch.load(lora_path, map_location="cpu")
    lora_pipe.unet.load_state_dict(lora_weights, strict=False)
    
    # 设置相同的随机种子
    seed = 1234
    generator = torch.Generator(device="cuda").manual_seed(seed)
    
    # 测试提示词
    prompts = [
        "a photo of a small red car on the street",
        "a cat sitting on a couch",
        "a dog playing in the park",
        "a person riding a bicycle",
        "a bowl of fruit on a table"
    ]
    
    # 创建输出目录
    os.makedirs("outputs/comparison", exist_ok=True)
    
    for i, prompt in enumerate(prompts):
        print(f"生成图像 {i+1}/{len(prompts)}: {prompt}")
        
        # 使用基础模型生成
        base_image = base_pipe(
            prompt=prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
            generator=generator
        ).images[0]
        
        # 使用LoRA模型生成
        lora_image = lora_pipe(
            prompt=prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
            generator=generator
        ).images[0]
        
        # 保存图像
        base_image.save(f"outputs/comparison/base_{i}.png")
        lora_image.save(f"outputs/comparison/lora_{i}.png")
        
        # 计算图像差异
        base_array = np.array(base_image.convert("RGB"))
        lora_array = np.array(lora_image.convert("RGB"))
        
        diff = np.abs(base_array.astype(np.float32) - lora_array.astype(np.float32))
        mean_diff = np.mean(diff)
        
        print(f"  平均差异: {mean_diff:.4f}")
        
        # 如果差异很小，可能LoRA没有生效
        if mean_diff < 5.0:  # 阈值可以根据需要调整
            print(f"  警告: 基础模型和LoRA模型的输出差异很小")
        
        print()

if __name__ == "__main__":
    compare_lora_effect()