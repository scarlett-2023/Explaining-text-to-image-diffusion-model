import os
import random
import spacy
import numpy as np
import nltk
from typing import List, Tuple
from PIL import Image
import torch
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import gc
import psutil
import sys
from nltk.corpus import wordnet
nltk.download('wordnet')
nltk.download('omw-1.4')

from diffusers import StableDiffusionPipeline
from transformers import CLIPTokenizer, CLIPTextModel
import clip

K = 1
N_MULTIPLIER = 5
MAX_PHRASE_LENGTH = 1
OUTPUT_DIR = "importance_eval_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def print_mem(prefix=""):
    process = psutil.Process(os.getpid())
    print(f"{prefix} 内存(MB): {process.memory_info().rss / 1024 / 1024:.2f}")

nlp = spacy.load("en_core_web_sm")

def get_best_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

class OrderedPhraseSegmenter:
    def __init__(self, max_phrase_length=5):
        self.max_phrase_length = max_phrase_length

    def _get_span_positions(self, tokens: List[spacy.tokens.Token]) -> Tuple[int, int]:
        starts = [t.idx for t in tokens]
        ends = [t.idx + len(t.text) for t in tokens]
        return min(starts), max(ends)

    def _recursive_segment(self, token, spans):
        subtree = list(token.subtree)
        if len(subtree) <= self.max_phrase_length:
            start, end = self._get_span_positions(subtree)
            text = ' '.join([t.text for t in subtree if not t.is_punct])
            spans.append((start, end, text))
            return

        for child in sorted(token.children, key=lambda x: x.i):
            self._recursive_segment(child, spans)

        current_tokens = [t for t in [token] if t not in {t for span in spans for t in span}]
        if current_tokens:
            start, end = self._get_span_positions(current_tokens)
            spans.append((start, end, token.text))

    def _merge_and_sort(self, spans):
        sorted_spans = sorted(spans, key=lambda x: x[0])
        merged = []
        for span in sorted_spans:
            if not merged:
                merged.append(span)
            else:
                last = merged[-1]
                if span[0] <= last[1]:
                    new_span = (min(last[0], span[0]), max(last[1], span[1]), f"{last[2]} {span[2]}")
                    merged[-1] = new_span
                else:
                    merged.append(span)
        return [text for _, _, text in merged]

    def segment(self, text: str) -> List[str]:
        doc = nlp(text)
        spans = []
        for sent in doc.sents:
            sent_spans = []
            self._recursive_segment(sent.root, sent_spans)
            for token in sent:
                if token.is_punct or any(token.idx >= s[0] and token.idx < s[1] for s in sent_spans):
                    continue
                sent_spans.append((token.idx, token.idx + len(token.text), token.text))
            spans.extend(sent_spans)
        return self._merge_and_sort(spans)

class MExGenPerturbator:
    def __init__(self, K=1, N_MULTIPLIER=5):
        self.K = K
        self.N_MULTIPLIER = N_MULTIPLIER
    def generate(self, units):
        d = len(units)
        n = d * self.N_MULTIPLIER
        samples = []
        for _ in range(n):
            k = random.randint(0, self.K)
            perturb_indices = random.sample(range(d), k) if k > 0 else []
            z = [0 if i in perturb_indices else 1 for i in range(d)]
            perturbed = ' '.join(units[i] if z[i] else '' for i in range(d))
            samples.append((perturbed, z))
        return samples

class TextImageSimilarityExplainer:
    def __init__(self, original_feature, units: List[str]):
        self.original_feature = original_feature
        self.units = units

    def calculate_similarity(self, image: Image.Image, preprocess_clip, model_clip, device) -> float:
        with torch.no_grad():
            img_tensor = preprocess_clip(image).unsqueeze(0).to(device)
            image_feature = model_clip.encode_image(img_tensor)
        similarity = torch.nn.functional.cosine_similarity(
            self.original_feature, image_feature, dim=1
        ).item() * 100
        return similarity

    def explain(self, perturbations: list, base_latents, pipe, preprocess_clip, model_clip, device):
        X_train = []
        y_train = []
        for (perturbed_text, z) in perturbations:
            with torch.no_grad():
                pert_img = pipe(
                    prompt=perturbed_text,
                    latents=base_latents.clone(),
                    num_inference_steps=25,
                    guidance_scale=7.5,
                    height=512,
                    width=512
                ).images[0]
                score = self.calculate_similarity(pert_img, preprocess_clip, model_clip, device)
                pert_img.close()
                del pert_img
            gc.collect()
            X_train.append(z)
            y_train.append(score)
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        model = LinearRegression()
        model.fit(X_train, y_train)
        coefficients = model.coef_
        unit_weights = {unit: coeff for unit, coeff in zip(self.units, coefficients)}
        return {'unit_weights': unit_weights, 'model': model}

def get_clip_feature(image, preprocess_clip, model_clip, device):
    with torch.no_grad():
        img_tensor = preprocess_clip(image).unsqueeze(0).to(device)
        feature = model_clip.encode_image(img_tensor)
    return feature

def mask_replace_phrase(phrase):
    return "[MASK]"

def main(txt_path, batch_limit=None, batch_idx=None):
    # Load models freshly with from_pretrained
    device = get_best_device()
    print(f"使用设备: {device}")
    # StableDiffusion
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", 
        torch_dtype=torch.float16 if device == "cuda" else torch.float32, 
        use_safetensors=True
    ).to(device)
    pipe.enable_attention_slicing()
    # CLIP
    import clip
    model_clip, preprocess_clip = clip.load("ViT-B/32", device=device)
    model_clip.eval()

    with open(txt_path, "r", encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]
    print(f"总输入文本数: {len(texts)}")
    generator = torch.Generator(device=device).manual_seed(42)
    base_latents = torch.randn((1, 4, 64, 64), generator=generator, device=device, dtype=pipe.unet.dtype)

    all_top_scores = []
    replaced_texts = []
    for idx, text in enumerate(texts):
        print_mem(f"\n=====[{idx+1}/{len(texts)}] 前")
        segmenter = OrderedPhraseSegmenter(max_phrase_length=MAX_PHRASE_LENGTH)
        units = segmenter.segment(text)
        print(f"Units: {units}")

        # 原始图片和特征
        with torch.no_grad():
            original_image = pipe(
                prompt=text,
                latents=base_latents.clone(),
                num_inference_steps=25,
                guidance_scale=7.5,
                height=512,
                width=512
            ).images[0]
            original_feature = get_clip_feature(original_image, preprocess_clip, model_clip, device)
            orig_img_save = os.path.join(OUTPUT_DIR, f"original_img_text{idx+1}_batch{batch_idx}.png")
            original_image.save(orig_img_save)
            original_image.close()
        gc.collect()

        perturbator = MExGenPerturbator(K=K, N_MULTIPLIER=N_MULTIPLIER)
        samples = perturbator.generate(units)

        explainer = TextImageSimilarityExplainer(original_feature, units)
        explanation_result = explainer.explain(samples, base_latents, pipe, preprocess_clip, model_clip, device)
        weights = sorted(
            explanation_result['unit_weights'].items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        print("重要性排名：")
        for unit, weight in weights[:5]:
            print(f"  {unit}: {weight:.2f}")

        replaced_unit_images = []
        replaced_unit_scores = []
        replaced_unit_texts = []
        replaced_unit_names = []
        for topk in [1, 2, 3]:
            z = [1]*len(units)
            if topk <= len(units):
                replace_idx = units.index(weights[topk-1][0])
                replaced_phrase = mask_replace_phrase(units[replace_idx])
                replaced_units = units.copy()
                replaced_units[replace_idx] = replaced_phrase
                replaced_unit_names.append((units[replace_idx], replaced_phrase))
            else:
                replaced_units = units.copy()
                replaced_unit_names.append(("None", "object"))
            replaced_text = ' '.join(replaced_units)
            replaced_unit_texts.append(replaced_text)
            with torch.no_grad():
                img = pipe(
                    prompt=replaced_text,
                    latents=base_latents.clone(),
                    num_inference_steps=25,
                    guidance_scale=7.5,
                    height=512,
                    width=512
                ).images[0]
                score = float(torch.nn.functional.cosine_similarity(
                    original_feature,
                    get_clip_feature(img, preprocess_clip, model_clip, device),
                    dim=1
                ).item() * 100)
                img_save_path = os.path.join(OUTPUT_DIR, f"replaced_text{idx+1}_top{topk}_batch{batch_idx}.png")
                img.save(img_save_path)
                img.close()
                del img
            gc.collect()
            replaced_unit_images.append(img_save_path)
            replaced_unit_scores.append(score)
            print(f"Top{topk} 替换 {replaced_unit_names[-1][0]} -> {replaced_unit_names[-1][1]} | score: {score:.2f}")

        with open(os.path.join(OUTPUT_DIR, f"text{idx+1}_original_batch{batch_idx}.txt"), "w", encoding="utf-8") as f:
            f.write(text)
        for topk, replaced_txt in enumerate(replaced_unit_texts, start=1):
            with open(os.path.join(OUTPUT_DIR, f"text{idx+1}_top{topk}_replaced_batch{batch_idx}.txt"), "w", encoding="utf-8") as f:
                f.write(replaced_txt)
        replaced_texts.append(replaced_unit_texts)

        all_top_scores.append(replaced_unit_scores)
        del segmenter, units, original_feature, explainer, explanation_result, weights
        gc.collect()
        print_mem(f"=====[{idx+1}/{len(texts)}] 后")

        if batch_limit is not None and (idx + 1) % batch_limit == 0:
            print(f"已处理{idx+1}条，建议重启kernel释放全部内存，再继续运行！")
            break

    all_top_scores = np.array(all_top_scores)
    mean_scores = all_top_scores.mean(axis=0)
    plt.figure(figsize=(12, 7))
    n_texts = len(all_top_scores)
    x = np.arange(1, n_texts+1)
    for i, (scores, name) in enumerate(zip(zip(*all_top_scores), ['Top1', 'Top2', 'Top3'])):
        plt.plot(x, scores, marker='o', label=f'{name} Replaced')
        plt.axhline(mean_scores[i], color=f"C{i}", linestyle="--", label=f'{name} Mean: {mean_scores[i]:.2f}')
    plt.xlabel('Text Index')
    plt.ylabel('CLIP Similarity (with original)')
    plt.title('CLIP Score after Replacing Top1/2/3 Important Units (Batch {})'.format(batch_idx))
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, f"score_lines_batch{batch_idx}.png")
    plt.savefig(plot_path)
    plt.close('all')
    print(f"分数曲线保存：{plot_path}")

    if batch_idx is not None:
        np.save(os.path.join(OUTPUT_DIR, f"batch_{batch_idx}_replaced_scores.npy"), all_top_scores)
        print(f"本批分数已保存: batch_{batch_idx}_replaced_scores.npy")
    else:
        np.save(os.path.join(OUTPUT_DIR, f"batch_replaced_scores.npy"), all_top_scores)
        print("本批分数已保存: batch_replaced_scores.npy")

if __name__ == "__main__":
    if len(sys.argv) > 2:
        txt_path = sys.argv[1]
        batch_idx = int(sys.argv[2])
        main(txt_path, batch_limit=None, batch_idx=batch_idx)
    else:
        txt_path = "your_texts.txt"
        main(txt_path, batch_limit=8, batch_idx=None)