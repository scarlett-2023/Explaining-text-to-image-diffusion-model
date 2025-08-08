import os
import torch
import json
import torch.nn as nn
from config import BASE_MODELS, OUTPUT_ROOT, EPOCHS, BATCH_SIZE, LR, DEVICE
from diffusers import StableDiffusionPipeline
from datasets import load_dataset
from PIL import Image
import requests
from io import BytesIO
from itertools import islice
import torch.nn.functional as F
from transformers import CLIPModel
import matplotlib.pyplot as plt
from collections import defaultdict
import time
import random
from urllib.parse import urlparse

def is_valid_image_response(response):
    """Check if the response contains a valid image"""
    content_type = response.headers.get('Content-Type', '')
    if not content_type.startswith('image/'):
        return False
    
    # Check minimum size (to avoid placeholder images)
    if len(response.content) < 1000:  # Less than 1KB is suspicious
        return False
        
    return True

def collate_fn(batch):
    images, prompts, groups, valid_indices = [], [], [], []
    
    for idx, item in enumerate(batch):
        prompt = item["caption"]
        url = item["image_url"]
        
        # Skip Getty Images URLs which are likely to be protected
        domain = urlparse(url).netloc
        if "gettyimages" in domain or "shutterstock" in domain or "stock" in domain:
            continue
            
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, timeout=5, headers=headers)
            
            if not response.ok or not is_valid_image_response(response):
                continue
                
            image = Image.open(BytesIO(response.content)).convert("RGB")
            
            # Validate image dimensions
            if image.width < 64 or image.height < 64:
                continue
                
            images.append(image)
            prompts.append(prompt)
            groups.append(item.get("group", "default"))
            valid_indices.append(idx)
            
        except Exception as e:
            print(f"Failed to load image: {url}, error: {e}")
            continue
            
    return images, prompts, groups, valid_indices

def get_batch(dataset_iter, batch_size):
    return list(islice(dataset_iter, batch_size))

def mask_token_loss(token_loss, func_indices):
    weights = torch.ones_like(token_loss)
    for idx in func_indices:
        if 0 <= idx < weights.shape[0]:
            weights[idx] = 0.0
    denom = weights.sum()
    if denom == 0:
        return torch.tensor(0.0, device=token_loss.device, dtype=token_loss.dtype)
    return (token_loss * weights).sum() / denom

def check_nan_inf(tensor, name):
    if torch.isnan(tensor).any():
        print(f"{name} contains nan!")
        return True
    if torch.isinf(tensor).any():
        print(f"{name} contains inf!")
        return True
    return False

def get_grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

def get_param_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        param_norm = p.data.norm(2)
        total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

def plot_and_save_loss(loss_dict, output_dir, model_tag):
    plt.figure(figsize=(10,6))
    for k, v in loss_dict.items():
        if k in ["step", "epoch", "sample_idx"]: continue
        if not v:  # Skip empty lists
            continue
        plt.plot(v, label=k)
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.title(f"Training Metrics [{model_tag}]")
    plt.legend()
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"loss_curve_{model_tag}.png")
    plt.savefig(save_path)
    print(f"Loss curve saved to {save_path}")
    plt.close()

def save_loss_log(loss_dict, output_dir, model_tag):
    log_path = os.path.join(output_dir, f"loss_log_{model_tag}.json")
    with open(log_path, "w") as f:
        json.dump(loss_dict, f)
    print(f"Loss log saved to {log_path}")

def compute_validation_loss(pipe, clip_model, proj_vision, val_dataset, func_indices_by_text, device, dtype, token_loss_weight, num_batches=5):
    # Only a few batches for speed
    pipe.unet.eval()
    proj_vision.eval()
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder

    val_iter = iter(val_dataset)
    total_loss = 0.0
    total_diff = 0.0
    total_token = 0.0
    n = 0
    with torch.no_grad():
        for _ in range(num_batches):
            batch_raw = get_batch(val_iter, BATCH_SIZE)
            if not batch_raw:
                break
            images, prompts, groups, valid_indices = collate_fn(batch_raw)
            if not images:
                continue
            batch_func_ids = [func_indices_by_text.get(str(sample_idx + i), []) for i, sample_idx in enumerate(valid_indices)]
            inputs = tokenizer(prompts, padding="max_length", truncation=True, return_tensors="pt").to(device)
            text_embeds = text_encoder(**inputs)[0]
            seq_len = text_embeds.shape[1]

            processed = [pipe.feature_extractor(image, return_tensors="pt")["pixel_values"].squeeze(0) for image in images]
            images_tensor = torch.stack(processed)
            images_tensor = images_tensor.to(device=device, dtype=torch.float32)
            images_tensor = images_tensor * 2.0 - 1.0

            vision_out = clip_model.vision_model(images_tensor)
            img_hidden = vision_out.last_hidden_state
            image_features = clip_model.visual_projection(img_hidden)
            if image_features.shape[1] < seq_len:
                pad_len = seq_len - image_features.shape[1]
                image_features = F.pad(image_features, (0,0,0,pad_len))[:, :seq_len, :]
            elif image_features.shape[1] > seq_len:
                image_features = image_features[:, :seq_len, :]
            image_features = image_features.to(dtype=dtype)
            image_features_proj = proj_vision(image_features)
            text_embeds = text_embeds.to(dtype=dtype)

            images_tensor_sd = images_tensor.to(dtype=dtype)
            latents = pipe.vae.encode(images_tensor_sd).latent_dist.sample()
            scaling_factor = getattr(pipe.vae.config, "scaling_factor", 0.18215)
            latents = latents * scaling_factor
            noise = torch.randn_like(latents, device=device, dtype=dtype)
            timesteps = torch.randint(
                0, pipe.scheduler.config.num_train_timesteps, 
                (latents.shape[0],), device=device
            ).long()
            noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)
            noisy_latents = noisy_latents.to(device=device, dtype=dtype)
            text_embeds_sd = text_embeds
            pred = pipe.unet(noisy_latents, timesteps, encoder_hidden_states=text_embeds_sd).sample
            diffusion_loss = F.mse_loss(pred, noise, reduction='mean')

            token_losses = []
            for b in range(len(prompts)):
                token_loss = F.mse_loss(text_embeds[b], image_features_proj[b], reduction='none').mean(dim=1)
                token_loss = torch.clamp(token_loss, min=-10, max=10)
                masked_token_loss = mask_token_loss(token_loss, batch_func_ids[b])
                token_losses.append(masked_token_loss)
            if token_losses:
                token_loss = torch.stack(token_losses).mean()
            else:
                token_loss = torch.tensor(0.0, device=device, dtype=dtype)
            total = diffusion_loss + token_loss_weight * token_loss
            total_loss += float(total.item())
            total_diff += float(diffusion_loss.item())
            total_token += float(token_loss.item())
            n += 1
    pipe.unet.train()
    proj_vision.train()
    if n == 0:
        return {"val_total_loss": None, "val_diffusion_loss": None, "val_token_loss": None}
    return {
        "val_total_loss": total_loss / n,
        "val_diffusion_loss": total_diff / n,
        "val_token_loss": total_token / n,
    }

def finetune_one(model_tag, base_model_path, dataset, func_indices_by_text):
    print(f"==== Start finetune: {model_tag} ({base_model_path}) ====")
    output_dir = os.path.join(OUTPUT_ROOT, f"finetuned_{model_tag}")
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(DEVICE if DEVICE != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            base_model_path,
            torch_dtype=torch.float32,
            use_safetensors=True
        ).to(device)
    except Exception as e:
        print(f"Skip model [{model_tag}] as it cannot be loaded: {e}")
        return
        
    pipe.enable_attention_slicing()
    pipe.unet.train()
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder

    dtype = pipe.unet.dtype

    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
    clip_model.eval()
    proj_vision = nn.Linear(512, 768).to(device)
    nn.init.zeros_(proj_vision.weight)
    nn.init.zeros_(proj_vision.bias)

    optimizer = torch.optim.Adam(list(pipe.unet.parameters()) + list(proj_vision.parameters()), lr=LR)
    token_loss_weight = 0.01

    # Split dataset for validation
    dataset_iter = iter(dataset)
    
    # Try to use a custom validation dataset if conceptual_captions fails
    try:
        val_dataset = load_dataset("conceptual_captions", split="validation", streaming=True)
        print("Loaded conceptual_captions validation dataset")
    except Exception as e:
        print(f"Could not load validation dataset: {e}")
        # Fallback to using LAION-Aesthetics or another dataset
        try:
            val_dataset = load_dataset("laion/laion-aesthetics", "laion-aesthetics-6plus", split="train", streaming=True)
            print("Using LAION-Aesthetics as fallback validation dataset")
        except Exception as e:
            print(f"Fallback validation dataset failed too: {e}")
            # Last resort: use training dataset as validation
            val_dataset = dataset
            print("Using training dataset for validation")

    # Convert func_indices_by_text keys to strings if they aren't already
    if func_indices_by_text and not all(isinstance(k, str) for k in func_indices_by_text.keys()):
        func_indices_by_text = {str(k): v for k, v in func_indices_by_text.items()}

    # ========== Loss Monitor ==========
    loss_dict = defaultdict(list)
    step_count = 0
    val_every = 100  # validate every N steps
    checkpoint_every = 500  # save checkpoint every N steps
    empty_batch_limit = 10  # Maximum number of consecutive empty batches before raising warning

    # Calculate approximate dataset size for progress tracking
    total_samples = 5000  # Just a reasonable assumption if we don't know exact size
    
    empty_batch_counter = 0
    skipped_sample_count = 0
    for epoch in range(EPOCHS):
        print(f"==== Epoch {epoch+1} of {EPOCHS} ====")
        sample_idx = 0
        dataset_iter = iter(dataset)
        
        while sample_idx < total_samples:
            # Get a new batch of samples
            batch_raw = get_batch(dataset_iter, BATCH_SIZE)
            if not batch_raw:
                print("End of dataset reached")
                break
                
            images, prompts, groups, valid_indices = collate_fn(batch_raw)
            
            # Handle empty batches
            if not images:
                empty_batch_counter += 1
                skipped_sample_count += len(batch_raw)
                sample_idx += len(batch_raw)
                
                if empty_batch_counter >= empty_batch_limit:
                    print(f"WARNING: {empty_batch_counter} consecutive empty batches. Check your dataset or image loading.")
                    print(f"Skipped {skipped_sample_count} samples so far.")
                    empty_batch_counter = 0  # Reset counter but continue
                
                # Small delay to avoid hammering servers if that's the issue
                time.sleep(0.5)
                continue
            
            # Reset counter when we get valid images
            empty_batch_counter = 0
            
            # Convert indices for proper lookup
            valid_str_indices = [str(sample_idx + i) for i in valid_indices]
            batch_func_ids = [func_indices_by_text.get(idx, []) for idx in valid_str_indices]
            
            # Tokenize prompts
            inputs = tokenizer(prompts, padding="max_length", truncation=True, return_tensors="pt").to(device)
            text_embeds = text_encoder(**inputs)[0]
            seq_len = text_embeds.shape[1]

            # Process images
            processed = [pipe.feature_extractor(image, return_tensors="pt")["pixel_values"].squeeze(0) for image in images]
            images_tensor = torch.stack(processed)
            images_tensor = images_tensor.to(device=device, dtype=torch.float32)
            images_tensor = images_tensor * 2.0 - 1.0

            # Extract CLIP features
            with torch.no_grad():
                vision_out = clip_model.vision_model(images_tensor)
                img_hidden = vision_out.last_hidden_state
                image_features = clip_model.visual_projection(img_hidden)
                
            # Ensure compatible shapes
            if image_features.shape[1] < seq_len:
                pad_len = seq_len - image_features.shape[1]
                image_features = F.pad(image_features, (0,0,0,pad_len))[:, :seq_len, :]
            elif image_features.shape[1] > seq_len:
                image_features = image_features[:, :seq_len, :]
                
            image_features = image_features.to(dtype=dtype)
            image_features_proj = proj_vision(image_features)
            text_embeds = text_embeds.to(dtype=dtype)

            # Check for NaN/Inf
            nan_flag = False
            nan_flag |= check_nan_inf(text_embeds, "text_embeds")
            nan_flag |= check_nan_inf(image_features_proj, "image_features_proj")
            if nan_flag:
                print("Nan/inf detected in input features, skipping batch.")
                sample_idx += len(batch_raw)
                continue

            # Generate diffusion training samples
            images_tensor_sd = images_tensor.to(dtype=dtype)
            with torch.no_grad():
                latents = pipe.vae.encode(images_tensor_sd).latent_dist.sample()
                scaling_factor = getattr(pipe.vae.config, "scaling_factor", 0.18215)
                latents = latents * scaling_factor
                
            noise = torch.randn_like(latents, device=device, dtype=dtype)
            timesteps = torch.randint(
                0, pipe.scheduler.config.num_train_timesteps, 
                (latents.shape[0],), device=device
            ).long()
            noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)
            noisy_latents = noisy_latents.to(device=device, dtype=dtype)
            
            # Forward pass through UNet
            text_embeds_sd = text_embeds
            pred = pipe.unet(noisy_latents, timesteps, encoder_hidden_states=text_embeds_sd).sample
            diffusion_loss = F.mse_loss(pred, noise, reduction='mean')

            # Calculate token-level losses
            token_losses = []
            for b in range(len(prompts)):
                token_loss = F.mse_loss(text_embeds[b], image_features_proj[b], reduction='none').mean(dim=1)
                token_loss = torch.clamp(token_loss, min=-10, max=10)
                masked_token_loss = mask_token_loss(token_loss, batch_func_ids[b])
                token_losses.append(masked_token_loss)
                
            if token_losses:
                token_loss = torch.stack(token_losses).mean()
            else:
                token_loss = torch.tensor(0.0, device=device, dtype=dtype)
                
            if torch.isnan(token_loss) or torch.isinf(token_loss):
                print("token_loss nan/inf! Zeroing out.")
                token_loss = torch.tensor(0.0, device=device, dtype=dtype)

            # Total loss
            total_loss = diffusion_loss + token_loss_weight * token_loss
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print("total_loss nan/inf! Step skipped.")
                sample_idx += len(batch_raw)
                continue

            # Optimization step
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(pipe.unet.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(proj_vision.parameters(), 1.0)
            optimizer.step()

            # ========== Record Loss & Monitoring ==========
            loss_dict["total_loss"].append(float(total_loss.item()))
            loss_dict["diffusion_loss"].append(float(diffusion_loss.item()))
            loss_dict["token_loss"].append(float(token_loss.item()))
            loss_dict["step"].append(step_count)
            loss_dict["epoch"].append(epoch+1)
            loss_dict["sample_idx"].append(sample_idx)
            # Gradient norms, parameter norms
            loss_dict["unet_grad_norm"].append(get_grad_norm(pipe.unet))
            loss_dict["proj_grad_norm"].append(get_grad_norm(proj_vision))
            loss_dict["unet_param_norm"].append(get_param_norm(pipe.unet))
            loss_dict["proj_param_norm"].append(get_param_norm(proj_vision))
            step_count += 1
            # ========== End Record ==========

            # Print progress
            print(f"[{model_tag}] Epoch {epoch+1}, step {step_count}, sample_idx {sample_idx}, "
                  f"loss={total_loss.item():.4f} (diff={diffusion_loss.item():.4f}, token={token_loss.item():.4f}), "
                  f"unet_gn={loss_dict['unet_grad_norm'][-1]:.4f}, proj_gn={loss_dict['proj_grad_norm'][-1]:.4f}")

            # Validation
            if step_count % val_every == 0:
                val_metrics = compute_validation_loss(pipe, clip_model, proj_vision, val_dataset, func_indices_by_text, device, dtype, token_loss_weight)
                for k, v in val_metrics.items():
                    if v is not None:
                        loss_dict[k].append(float(v))
                print(f"[VALIDATION] step {step_count} -- " +
                      ", ".join([f"{k}={v:.4f}" for k, v in val_metrics.items() if v is not None]))
                save_loss_log(loss_dict, output_dir, model_tag)
                plot_and_save_loss(loss_dict, output_dir, model_tag)
                
            # Checkpoint saving
            if step_count % checkpoint_every == 0:
                checkpoint_dir = os.path.join(output_dir, f"checkpoint-{step_count}")
                os.makedirs(checkpoint_dir, exist_ok=True)
                pipe.save_pretrained(checkpoint_dir)
                print(f"Saved checkpoint at step {step_count} to {checkpoint_dir}")

            sample_idx += len(batch_raw)

        # Save after each epoch
        plot_and_save_loss(loss_dict, output_dir, model_tag)
        save_loss_log(loss_dict, output_dir, model_tag)
        epoch_dir = os.path.join(output_dir, f"epoch-{epoch+1}")
        os.makedirs(epoch_dir, exist_ok=True)
        pipe.save_pretrained(epoch_dir)
        print(f"Saved model after epoch {epoch+1} to {epoch_dir}")

    # Save final model
    pipe.save_pretrained(output_dir)
    print(f"Finetuned model [{model_tag}] saved to {output_dir}")
    plot_and_save_loss(loss_dict, output_dir, model_tag)
    save_loss_log(loss_dict, output_dir, model_tag)

def main():
    hi_func_path = os.path.join(OUTPUT_ROOT, "hi_func_indices.json")
    if not os.path.exists(hi_func_path):
        print(f"ERROR: {hi_func_path} not found.")
        return
        
    with open(hi_func_path) as f:
        func_indices_by_text = json.load(f)
        
    # Try different datasets if conceptual_captions fails
    datasets_to_try = [
        ("conceptual_captions", "train"),
        ("laion/laion-aesthetics", "laion-aesthetics-6plus"),
        ("dalle-mini/yfcc15m", "train")
    ]
    
    dataset = None
    for dataset_name, split in datasets_to_try:
        try:
            if "/" in dataset_name:
                name, subset = dataset_name.split("/")
                dataset = load_dataset(name, subset, split=split, streaming=True)
            else:
                dataset = load_dataset(dataset_name, split=split, streaming=True)
            print(f"Successfully loaded dataset: {dataset_name} ({split})")
            break
        except Exception as e:
            print(f"Failed to load dataset {dataset_name}: {e}")
    
    if dataset is None:
        print("ERROR: Could not load any dataset. Exiting.")
        return
        
    for model_tag, base_model_path in BASE_MODELS.items():
        finetune_one(model_tag, base_model_path, dataset, func_indices_by_text)
    
    print("All model fine-tuning completed")

if __name__ == "__main__":
    main()