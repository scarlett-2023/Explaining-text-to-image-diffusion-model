import os, importlib.util, spacy, torch, clip, numpy as np, json, csv
from PIL import Image
import matplotlib.pyplot as plt
from config import TXT_PATH, BASE_MODELS, OUTPUT_ROOT, K_IMPORTANCE, FID_BATCH
from inception_score import get_inception_score
import torch

def get_importance_and_metrics(text, main_batch, model_path, nlp, FUNCTION_WORD_POS, text_idx, model_tag):
    segmenter = main_batch.OrderedPhraseSegmenter(max_phrase_length=main_batch.MAX_PHRASE_LENGTH)
    units = segmenter.segment(text)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = main_batch.StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16 if device.type=="cuda" else torch.float32, use_safetensors=True).to(device)
    pipe.enable_attention_slicing()
    model_clip, preprocess_clip = clip.load("ViT-B/32", device=device)
    model_clip.eval()
    generator = torch.Generator(device=device).manual_seed(42)
    base_latents = torch.randn((1, 4, 64, 64), generator=generator, device=device, dtype=pipe.unet.dtype)
    # 原始图片和特征
    with torch.no_grad():
        original_image = pipe(prompt=text, latents=base_latents.clone(), num_inference_steps=25, guidance_scale=7.5, height=512, width=512).images[0]
        original_feature = main_batch.get_clip_feature(original_image, preprocess_clip, model_clip, device)
        img_dir = os.path.join(OUTPUT_ROOT, "clip_images", model_tag)
        os.makedirs(img_dir, exist_ok=True)
        orig_img_save = os.path.join(img_dir, f"original_img_text{text_idx+1}.png")
        original_image.save(orig_img_save)
        original_image.close()
    # importance
    perturbator = main_batch.MExGenPerturbator(K=main_batch.K, N_MULTIPLIER=main_batch.N_MULTIPLIER)
    samples = perturbator.generate(units)
    explainer = main_batch.TextImageSimilarityExplainer(original_feature, units)
    explanation_result = explainer.explain(samples, base_latents, pipe, preprocess_clip, model_clip, device)
    unit_weights = explanation_result["unit_weights"]
    doc = nlp(text)
    unit_types = []
    for unit in units:
        found_type = "other"
        for token in doc:
            if token.text == unit:
                found_type = ("function" if token.pos_ in FUNCTION_WORD_POS else "content")
                break
        unit_types.append(found_type)
    importance_sorted = sorted(
        [(i, units[i], unit_weights.get(units[i], 0), unit_types[i]) for i in range(len(units))],
        key=lambda x: abs(x[2]), reverse=True
    )
    n_units = len(units)
    n_top = max(1, int(n_units * K_IMPORTANCE))
    top_units = importance_sorted[:n_top]
    top_func_units = [item for item in top_units if item[3] == "function"]
    # remove分数
    rm_scores = []
    for i in range(len(units)):
        new_text = ' '.join([u for j, u in enumerate(units) if j != i])
        with torch.no_grad():
            img = pipe(prompt=new_text, latents=base_latents.clone(), num_inference_steps=25, guidance_scale=7.5, height=512, width=512).images[0]
            score = run_clip_similarity(img, preprocess_clip, model_clip, original_feature, device)
            img_path = os.path.join(img_dir, f"rm_unit{i}_text{text_idx+1}_{units[i]}.png")
            img.save(img_path)
            img.close()
        rm_scores.append({
            "unit_idx": i, "unit_text": units[i],
            "unit_type": unit_types[i],
            "clip_score": score,
            "img_path": img_path
        })
    orig_sim = run_clip_similarity(Image.open(orig_img_save), preprocess_clip, model_clip, original_feature, device)
    hi_func_drops = []
    for item in top_func_units:
        idx = item[0]
        drop = orig_sim - rm_scores[idx]["clip_score"]
        hi_func_drops.append(drop)
    # 图片质量评估用
    img_quality_dir = os.path.join(OUTPUT_ROOT, "fid_imgs", model_tag)
    os.makedirs(img_quality_dir, exist_ok=True)
    Image.open(orig_img_save).save(os.path.join(img_quality_dir, f"{text_idx+1}.png"))
    return {
        "importance_sorted": importance_sorted,
        "top_func_units": top_func_units,
        "rm_scores": rm_scores,
        "hi_func_drops": hi_func_drops,
        "original_sim": orig_sim,
        "orig_img_save": orig_img_save,
        "rm_img_paths": [rm["img_path"] for rm in rm_scores]
    }

def run_clip_similarity(img, preprocess_clip, model_clip, original_feature, device):
    with torch.no_grad():
        img_tensor = preprocess_clip(img).unsqueeze(0).to(device)
        image_feature = model_clip.encode_image(img_tensor)
        score = torch.nn.functional.cosine_similarity(
            original_feature, image_feature, dim=1
        ).item() * 100
    return score

def calc_fid_score(img_dir1, img_dir2):
    try:
        from pytorch_fid import fid_score
        score = fid_score.calculate_fid_given_paths([img_dir1, img_dir2], batch_size=FID_BATCH, device="cuda" if torch.cuda.is_available() else "cpu", dims=2048)
        return score
    except Exception as e:
        print("FID计算失败:", e)
        return None

def calc_is_score(img_dir):
    try:
        from inception_score import get_inception_score
        imgs = []
        files = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".png")])
        for fpath in files:
            img = Image.open(fpath).convert("RGB")
            img = img.resize((299, 299))
            img = np.asarray(img).astype(np.float32) / 255.0
            img = img * 2 - 1
            img = torch.from_numpy(img).permute(2, 0, 1)
            print(f"{os.path.basename(fpath)} shape: {img.shape}")  # 应为(3,299,299)
            imgs.append(img)
        if not imgs:
            print("IS计算失败: 没有图片")
            return None, None
        imgs = torch.stack(imgs, dim=0)
        print("最终 imgs shape:", imgs.shape)
        assert imgs.shape[1] == 3, f"图片不是3通道: {imgs.shape}"
        is_mean, is_std = get_inception_score(imgs, cuda=torch.cuda.is_available(), batch_size=FID_BATCH, resize=False, splits=1)
        return is_mean, is_std
    except Exception as e:
        print("IS计算失败:", e)
        return None, None

def main():
    spec = importlib.util.spec_from_file_location("main_batch", "main_batch.py")
    main_batch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(main_batch)
    nlp = spacy.load("en_core_web_sm")
    FUNCTION_WORD_POS = {'ADP','CCONJ','DET','PART','PRON','SCONJ','AUX','INTJ','NUM','SYM','X'}
    with open(TXT_PATH, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip()]
    all_stats = []
    for text_idx, text in enumerate(texts):
        per_model_stats = {}
        for model_tag, model_path in BASE_MODELS.items():
            stats = get_importance_and_metrics(
                text, main_batch, model_path, nlp, FUNCTION_WORD_POS, text_idx, model_tag
            )
            per_model_stats[model_tag] = stats
        all_stats.append({"text": text, "stats": per_model_stats})
        print(f"\n=== Text {text_idx+1} ===")
        print(f"{text}")
        for model_tag in BASE_MODELS.keys():
            print(f"Model: {model_tag}")
            print("High importance function words (top 30%):")
            for i, unit, weight, typ in per_model_stats[model_tag]["top_func_units"]:
                print(f"unit={unit} type={typ} importance={weight:.4f}")
            print("平均CLIP分数:", per_model_stats[model_tag]["original_sim"])
            print("高importance虚词 sim drop:", per_model_stats[model_tag]["hi_func_drops"])
    # 汇总
    metrics_table = []
    for idx, s in enumerate(all_stats):
        row = {"text_idx": idx+1, "text": s["text"]}
        for model_tag in BASE_MODELS.keys():
            row[f"{model_tag}_clip"] = s["stats"][model_tag]["original_sim"]
            row[f"{model_tag}_hi_func_drop"] = np.mean(s["stats"][model_tag]["hi_func_drops"]) if s["stats"][model_tag]["hi_func_drops"] else 0
        metrics_table.append(row)
    # FID/IS
    fid_results = {}
    is_results = {}
    model_tags = list(BASE_MODELS.keys())
    for i in range(len(model_tags)):
        for j in range(i+1, len(model_tags)):
            tag1 = model_tags[i]
            tag2 = model_tags[j]
            fid_score_val = calc_fid_score(
                os.path.join(OUTPUT_ROOT, "fid_imgs", tag1),
                os.path.join(OUTPUT_ROOT, "fid_imgs", tag2)
            )
            fid_results[f"{tag1}_vs_{tag2}"] = fid_score_val
    for tag in model_tags:
        is_mean, is_std = calc_is_score(os.path.join(OUTPUT_ROOT, "fid_imgs", tag))
        is_results[tag] = {"mean": is_mean, "std": is_std}
    # 可视化（改进：子图分别画CLIP和hi_func_drop分布，区间不再互相影响）
    plt.figure(figsize=(12, 2 * len(model_tags)))
    for idx, tag in enumerate(model_tags):
        vals_clip = [row[f"{tag}_clip"] for row in metrics_table]
        vals_drop = [row[f"{tag}_hi_func_drop"] for row in metrics_table]
        plt.subplot(len(model_tags), 2, idx * 2 + 1)
        plt.boxplot(vals_clip)
        plt.ylabel("CLIP score")
        plt.title(f"{tag} CLIP")
        plt.subplot(len(model_tags), 2, idx * 2 + 2)
        plt.boxplot(vals_drop)
        plt.ylabel("Hi importance function drop")
        plt.title(f"{tag} hi func drop")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_ROOT, "multi_model_summary.png"))
    print("可视化已保存:", os.path.join(OUTPUT_ROOT, "multi_model_summary.png"))
    # 保存CSV/JSON
    with open(os.path.join(OUTPUT_ROOT, "summary.csv"), "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metrics_table[0].keys())
        writer.writeheader()
        for row in metrics_table:
            writer.writerow(row)
    with open(os.path.join(OUTPUT_ROOT, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({
            "metrics": metrics_table,
            "fid": fid_results,
            "is": is_results
        }, f, indent=2)
    print("汇总csv/json已保存:", os.path.join(OUTPUT_ROOT, "summary.csv"), os.path.join(OUTPUT_ROOT, "summary.json"))
    print("FID结果:", fid_results)
    print("IS结果:", is_results)

if __name__ == "__main__":
    main()