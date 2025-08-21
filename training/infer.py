import os
import argparse
import torch
from diffusers import StableDiffusionPipeline

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, default="runwayml/stable-diffusion-v1-5")
    ap.add_argument("--lora_dir", type=str, required=True, help="训练保存的 LoRA 目录（包含 attn processor 权重）")
    ap.add_argument("--prompt", type=str, default="a photo of a small red car on the street")
    ap.add_argument("--negative_prompt", type=str, default="")
    ap.add_argument("--num_inference_steps", type=int, default=30)
    ap.add_argument("--guidance_scale", type=float, default=7.5)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--output", type=str, default="./outputs/sample.png")
    return ap.parse_args()

def main():
    args = parse_args()
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
        safety_checker=None,
    )
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

    # 加载 LoRA 到 UNet
    pipe.unet.load_attn_procs(args.lora_dir)

    g = torch.Generator(device=pipe.device).manual_seed(args.seed)
    image = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt if args.negative_prompt else None,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generator=g
    ).images[0]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    image.save(args.output)
    print(f"Saved image to {args.output}")

if __name__ == "__main__":
    main()