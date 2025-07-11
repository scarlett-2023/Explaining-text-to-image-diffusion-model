import random
import spacy
import numpy as np
from typing import List, Tuple
from PIL import Image
import torch
import clip
import matplotlib.pyplot as plt
from diffusers import StableDiffusionPipeline
from sklearn.linear_model import LinearRegression
from itertools import combinations

# ----------------- 1. 初始化与参数 -----------------
nlp = spacy.load("en_core_web_sm")
device = "cuda" if torch.cuda.is_available() else "cpu"
model_clip, preprocess_clip = clip.load("ViT-B/32", device=device)
model_clip.eval()

# 输入文本，100词长句，无标点
text_length=100
text = "A serene mountain landscape at sunrise with golden sunlight illuminating snowcapped peaks and lush green pine forests A crystal clear river winds through the valley eflecting the vibrant colors of the sky Wildflowers in shades of purple yellow and pink blanket the meadows In the distance a small wooden cabin with smoke curling from its chimney sits near the riverbank Birds soar gracefully above the trees while a family of deer grazes peacefully by the water edge The atmosphere is tranquil magical and filled with the soft glow of dawn" 
MAX_PHRASE_LENGTH = 4  # unit最大长度
K = 1                 # 每次扰动的pair数
N_MULTIPLIER = 40   # 扰动样本数限制
N_PERTURBATIONS = 200  # 扰动样本数限制
PLOTPATH = f"pair/unit_pair_{text_length}_{MAX_PHRASE_LENGTH}_{K}_{N_PERTURBATIONS}.png"

# ----------------- 2. Unit分割 -----------------
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

segmenter = OrderedPhraseSegmenter(max_phrase_length=MAX_PHRASE_LENGTH)
units = segmenter.segment(text)
print(f"unit总数: {len(units)}")
for i, u in enumerate(units[:10]):
    print(f"[unit {i+1}] {u}")
if len(units) > 10:
    print("...")

# ----------------- 3. 生成所有unit pair -----------------
unit_indices = list(range(len(units)))
unit_pairs = list(combinations(unit_indices, 2))
print(f"unit pair总数: {len(unit_pairs)}")

# ----------------- 4. 扰动样本生成器（以pair为扰动单元） -----------------
class PairPerturbator:
    def __init__(self, K, N_MULTIPLIER, unit_pairs):
        self.K = K
        self.N_MULTIPLIER = N_MULTIPLIER
        self.unit_pairs = unit_pairs

    def generate(self, units):
        d = len(units)
        n = d * self.N_MULTIPLIER
        samples = []
        for _ in range(n):
            chosen_pairs = random.sample(self.unit_pairs, self.K)
            zero_indices = set()
            for i, j in chosen_pairs:
                zero_indices.add(i)
                zero_indices.add(j)
            z = [0 if idx in zero_indices else 1 for idx in range(d)]
            perturbed = ' '.join(units[i] if z[i] else '' for i in range(d))
            samples.append((perturbed, z, chosen_pairs))
        return samples

perturbator = PairPerturbator(K, N_MULTIPLIER, unit_pairs)
samples = perturbator.generate(units)[:N_PERTURBATIONS]
perturb_texts = [ptext for ptext, _, _ in samples]
z_vectors = [z for _, z, _ in samples]
pair_lists = [pairs for _, _, pairs in samples]

print("\n扰动样本示例：")
for i, (ptext, z, pairs) in enumerate(samples[:3]):
    removed_units = [units[idx] for idx, val in enumerate(z) if val == 0]
    print(f"扰动样本 {i+1}:")
    print(f"生成文本: '{ptext}'")
    print(f"删除的unit: {removed_units[:6]} ...")
    print(f"扰动的pair: {pairs[:2]} ...")
    print("-" * 80)

# ----------------- 5. 图像生成与特征提取（GPU） -----------------
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    use_safetensors=True
).to(device)
pipe.enable_attention_slicing()
generator = torch.Generator(device=device).manual_seed(42)
base_latents = torch.randn(
    (1, 4, 64, 64),
    generator=generator,
    device=device,
    dtype=pipe.unet.dtype
)

def generate_images(prompts: list, base_latents: torch.Tensor, pipe) -> list:
    images = []
    for prompt in prompts:
        image = pipe(
            prompt=prompt,
            latents=base_latents.clone(),
            num_inference_steps=25,
            guidance_scale=7.5,
            height=512,
            width=512
        ).images[0]
        images.append(image)
    return images

print("\n生成原始图片 ...")
original_image = pipe(
    prompt=text,
    latents=base_latents.clone(),
    num_inference_steps=25,
    guidance_scale=7.5,
    height=512,
    width=512
).images[0]
with torch.no_grad():
    img_tensor = preprocess_clip(original_image).unsqueeze(0).to(device)
    original_feature = model_clip.encode_image(img_tensor)

print("生成扰动图片 ...")
perturbed_images = generate_images(perturb_texts, base_latents, pipe)

# ----------------- 6. 相似度解释与pair重要性回归 -----------------
def calc_similarity(image: Image.Image, original_feature, model_clip, preprocess_clip, device):
    with torch.no_grad():
        img_tensor = preprocess_clip(image).unsqueeze(0).to(device)
        image_feature = model_clip.encode_image(img_tensor)
        similarity = torch.nn.functional.cosine_similarity(
            original_feature, image_feature, dim=1
        ).item() * 100
    return similarity

print("计算所有样本与原图像的相似度 ...")
similarities = [
    calc_similarity(img, original_feature, model_clip, preprocess_clip, device)
    for img in perturbed_images
]

# 构造pair mask矩阵
n_samples = len(samples)
n_pairs = len(unit_pairs)
pair_mask = np.zeros((n_samples, n_pairs), dtype=int)
for i, pairs in enumerate(pair_lists):
    for pair in pairs:
        pair_idx = unit_pairs.index(pair)
        pair_mask[i, pair_idx] = 1
X = pair_mask
y = np.array(similarities)

model = LinearRegression()
model.fit(X, y)
pair_importances = model.coef_

# 还原pair名字
pair_names = [f"{units[i]} + {units[j]}" for (i, j) in unit_pairs]
pair_weights = {name: coef for name, coef in zip(pair_names, pair_importances)}

# ----------------- 7. 结果输出与可视化 -----------------
sorted_pairs = sorted(pair_weights.items(), key=lambda x: abs(x[1]), reverse=True)

print("\n最重要的前10个unit pair：")
for pair, w in sorted_pairs[:10]:
    print(f"{pair}: {w:.2f}%")

plt.figure(figsize=(10, 7))
top_pairs = sorted_pairs[:20]
labels = [x[0] for x in top_pairs]
values = [x[1] for x in top_pairs]
colors = ['green' if v > 0 else 'red' for v in values]
plt.barh(range(len(labels)), values, color=colors)
plt.yticks(range(len(labels)), labels)
plt.xlabel('Pair Importance Weight')
plt.title('Top 20 Unit Pair Importance')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(PLOTPATH)
plt.close()
print(f"\n前20重要pair可视化已保存到: {PLOTPATH}")

# ----------------- 8. 可视化功能（可选） -----------------
def plot_perturbation_images(original_image, perturbed_images, samples, units, max_cols=4):
    n_samples = len(perturbed_images)
    n_cols = min(max_cols, n_samples + 1)
    n_rows = (n_samples + 1 + n_cols - 1) // n_cols
    plt.figure(figsize=(18, 4*n_rows))
    plt.subplot(n_rows, n_cols, 1)
    plt.imshow(original_image)
    plt.title("Original Image\n(Full Text)", fontsize=10)
    plt.axis('off')
    for i, (img, (ptext, z, pairs)) in enumerate(zip(perturbed_images, samples)):
        removed_units = [units[idx] for idx, val in enumerate(z) if val == 0]
        title = f"Perturb {i+1}\nRemoved: {removed_units or 'None'}"
        plt.subplot(n_rows, n_cols, i+2)
        plt.imshow(img)
        plt.title(title, fontsize=8, color='red' if removed_units else 'green')
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    print(f"有效扰动图像数量：{len(perturbed_images)}")

#plot_perturbation_images(original_image, perturbed_images, samples, units)