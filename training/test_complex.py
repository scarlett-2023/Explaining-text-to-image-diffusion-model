import torch
from diffusers import StableDiffusionPipeline
import numpy as np
from PIL import Image, ImageChops
import os
import matplotlib.pyplot as plt

def visualize_lora_differences():
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
    prompt = "a photo of a small red car on the street"
    
    # 创建输出目录
    os.makedirs("outputs/detailed_comparison", exist_ok=True)
    
    print(f"生成图像: {prompt}")
    
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
    
    # 保存原始图像
    base_image.save("outputs/detailed_comparison/base.png")
    lora_image.save("outputs/detailed_comparison/lora.png")
    
    # 计算差异图像
    base_array = np.array(base_image.convert("RGB"))
    lora_array = np.array(lora_image.convert("RGB"))
    
    diff = np.abs(base_array.astype(np.float32) - lora_array.astype(np.float32))
    mean_diff = np.mean(diff)
    
    # 将差异图像标准化到0-255范围
    diff_normalized = (diff / np.max(diff) * 255).astype(np.uint8)
    diff_image = Image.fromarray(diff_normalized)
    diff_image.save("outputs/detailed_comparison/difference.png")
    
    print(f"平均差异: {mean_diff:.4f}")
    
    # 创建并排比较图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(base_image)
    axes[0].set_title("Base Model")
    axes[0].axis('off')
    
    axes[1].imshow(lora_image)
    axes[1].set_title("LoRA Model")
    axes[1].axis('off')
    
    axes[2].imshow(diff_image)
    axes[2].set_title(f"Difference (Mean: {mean_diff:.2f})")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig("outputs/detailed_comparison/comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print("详细比较图已保存到 outputs/detailed_comparison/comparison.png")
    
    # 分析差异分布
    print("\n差异分布分析:")
    print(f"最大差异: {np.max(diff):.2f}")
    print(f"最小差异: {np.min(diff):.2f}")
    print(f"差异标准差: {np.std(diff):.2f}")
    
    # 计算差异大于阈值的像素比例
    threshold = 30  # 设置差异阈值
    high_diff_pixels = np.sum(diff > threshold) / diff.size * 100
    print(f"差异大于 {threshold} 的像素比例: {high_diff_pixels:.2f}%")
    
    return mean_diff, high_diff_pixels

if __name__ == "__main__":
    visualize_lora_differences()