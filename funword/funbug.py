import spacy
import numpy as np
from typing import List
from PIL import Image
import torch
import clip
from diffusers import StableDiffusionPipeline
from sklearn.metrics.pairwise import cosine_similarity

# ----------------- 1. 初始化与参数 -----------------
nlp = spacy.load("en_core_web_sm")
device = "cuda" if torch.cuda.is_available() else "cpu"
model_clip, preprocess_clip = clip.load("ViT-B/32", device=device)
model_clip.eval()

text = "A serene mountain landscape at sunrise with golden sunlight illuminating snowcapped peaks and lush green pine forests A crystal clear river winds through the valley eflecting the vibrant colors of the sky Wildflowers in shades of purple yellow and pink blanket the meadows In the distance a small wooden cabin with smoke curling from its chimney sits near the riverbank Birds soar gracefully above the trees while a family of deer grazes peacefully by the water edge The atmosphere is tranquil magical and filled with the soft glow of dawn"
MAX_PHRASE_LENGTH = 4

# ----------------- 2. Unit分割 -----------------
class OrderedPhraseSegmenter:
    def __init__(self, max_phrase_length=5):
        self.max_phrase_length = max_phrase_length
    def segment(self, text: str) -> List[str]:
        doc = nlp(text)
        spans = []
        for sent in doc.sents:
            for chunk in sent.noun_chunks:
                spans.append(chunk.text)
            for token in sent:
                if token.is_punct or token.is_space or any(token.idx >= chunk.start_char and token.idx < chunk.end_char for chunk in sent.noun_chunks):
                    continue
                spans.append(token.text)
        seen, result = set(), []
        for x in spans:
            if x not in seen:
                seen.add(x)
                result.append(x)
        return result

segmenter = OrderedPhraseSegmenter(max_phrase_length=MAX_PHRASE_LENGTH)
units = segmenter.segment(text)
print(f"unit总数: {len(units)}")
for i, u in enumerate(units[:10]):
    print(f"[unit {i+1}] {u}")
if len(units) > 10:
    print("...")

# ----------------- 3. 智能虚词/实词划分 -----------------
def smart_word_type(units):
    doc = nlp(' '.join(units))
    unit_pos = []
    for u in units:
        match = None
        for span in doc.sents:
            if u in span.text:
                for tok in span:
                    if tok.text in u:
                        match = tok
                        break
                if match: break
        pos = match.pos_ if match else 'X'
        unit_pos.append(pos)
    return unit_pos

unit_pos = smart_word_type(units)
content_word_pos = {'NOUN','VERB','ADJ','ADV','PROPN'}
function_word_pos = {'ADP','CCONJ','DET','PART','PRON','SCONJ','AUX','INTJ','NUM','SYM','X'}
content_indices = [i for i, pos in enumerate(unit_pos) if pos in content_word_pos]
function_indices = [i for i, pos in enumerate(unit_pos) if pos in function_word_pos]
print(f"实词数: {len(content_indices)}, 虚词数: {len(function_indices)}")

# ----------------- 4. 生成原始图像及clip特征 -----------------
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

def generate_image(prompt: str, base_latents, pipe) -> Image.Image:
    image = pipe(
        prompt=prompt,
        latents=base_latents.clone(),
        num_inference_steps=25,
        guidance_scale=7.5,
        height=512,
        width=512
    ).images[0]
    return image

def get_clip_feature(image: Image.Image, preprocess_clip, model_clip, device):
    with torch.no_grad():
        img_tensor = preprocess_clip(image).unsqueeze(0).to(device)
        image_feature = model_clip.encode_image(img_tensor)
        image_feature = image_feature.cpu().numpy().flatten()
    return image_feature

print("生成原始图片及特征 ...")
original_image = generate_image(text, base_latents, pipe)
original_feature = get_clip_feature(original_image, preprocess_clip, model_clip, device)

# ----------------- 5. 每个unit单独删一次，生成图片与clip特征 -----------------
unit_del_images = []
unit_del_features = []
for i in range(len(units)):
    perturbed_units = [units[j] for j in range(len(units)) if j != i]
    perturbed_text = ' '.join(perturbed_units)
    img = generate_image(perturbed_text, base_latents, pipe)
    feat = get_clip_feature(img, preprocess_clip, model_clip, device)
    unit_del_images.append(img)
    unit_del_features.append(feat)
print("全部完成。")

# ----------------- 6. 计算pair相似性与和原图距离 -----------------
results = []
for fi in function_indices:
    for ci in content_indices:
        if fi == ci:
            continue
        feat_f = unit_del_features[fi]
        feat_c = unit_del_features[ci]
        sim_fc = cosine_similarity([feat_f], [feat_c])[0][0]
        sim_f_orig = cosine_similarity([feat_f], [original_feature])[0][0]
        sim_c_orig = cosine_similarity([feat_c], [original_feature])[0][0]
        # 只保留“删虚词+删实词的图像clip特征很像，且都和原图不像”的结果
        results.append({
            "虚词unit": units[fi],
            "虚词pos": unit_pos[fi],
            "实词unit": units[ci],
            "实词pos": unit_pos[ci],
            "pair_sim": sim_fc,
            "f_orig_sim": sim_f_orig,
            "c_orig_sim": sim_c_orig
        })

# 设定阈值，筛选最可疑的bug-pair
results = sorted(results, key=lambda x: (x['pair_sim'] - 0.5*(x['f_orig_sim']+x['c_orig_sim'])), reverse=True)
print("\n最可疑的前10组：")
for r in results[:10]:
    print(f"虚词: {r['虚词unit']}({r['虚词pos']}) <-> 实词: {r['实词unit']}({r['实词pos']}) | pair_sim={r['pair_sim']:.2f} f2orig={r['f_orig_sim']:.2f} c2orig={r['c_orig_sim']:.2f}")

# 可选：保存最可疑bug-pair的图片
import os
os.makedirs("bug_pair_imgs", exist_ok=True)
for idx, r in enumerate(results[:5]):
    fi = units.index(r['虚词unit'])
    ci = units.index(r['实词unit'])
    unit_del_images[fi].save(f"bug_pair_imgs/bugpair_{idx+1}_f_{r['虚词unit']}.png")
    unit_del_images[ci].save(f"bug_pair_imgs/bugpair_{idx+1}_c_{r['实词unit']}.png")
print("最可疑bug pair图片已保存到bug_pair_imgs/")


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# 假设你已经有如下变量：
# units: List[str] 所有unit短语
# unit_pos: List[str] 每个unit的词性标记
# content_word_pos, function_word_pos: set，实词/虚词的POS集合
# content_indices, function_indices: List[int]，分别为实词、虚词在units中的索引
# unit_del_features: List[np.ndarray]，每个unit被单独删除时的clip特征
# original_feature: np.ndarray，原始图像的clip特征

N_SHOW = 10  # 展示前N个最可疑pair

results = []
for fi in function_indices:
    for ci in content_indices:
        if fi == ci:
            continue
        feat_f = unit_del_features[fi]
        feat_c = unit_del_features[ci]
        sim_fc = cosine_similarity([feat_f], [feat_c])[0][0]
        sim_f_orig = cosine_similarity([feat_f], [original_feature])[0][0]
        sim_c_orig = cosine_similarity([feat_c], [original_feature])[0][0]
        # 你可以调整这个综合分数公式以控制筛选标准
        bug_score = sim_fc - 0.5 * (sim_f_orig + sim_c_orig)
        results.append({
            "虚词unit": units[fi],
            "虚词pos": unit_pos[fi],
            "实词unit": units[ci],
            "实词pos": unit_pos[ci],
            "pair_sim": sim_fc,
            "f_orig_sim": sim_f_orig,
            "c_orig_sim": sim_c_orig,
            "bug_score": bug_score,
            "fi": fi,
            "ci": ci
        })

# 排序并展示
results = sorted(results, key=lambda x: x['bug_score'], reverse=True)
print("\n最可疑的前N组虚实pair（删后图像特征相似，且都与原始图像距离大）：")
for r in results[:N_SHOW]:
    print(f"虚词: {r['虚词unit']}({r['虚词pos']}) <-> 实词: {r['实词unit']}({r['实词pos']}) | pair_sim={r['pair_sim']:.2f} f2orig={r['f_orig_sim']:.2f} c2orig={r['c_orig_sim']:.2f} bug_score={r['bug_score']:.2f}")

# ----------------- 可视化前N组pair的特征空间分布 -----------------
# 可选：二维降维可视化
from sklearn.decomposition import PCA

features_for_pca = [original_feature] + [unit_del_features[i] for i in range(len(units))]
labels_for_pca = ["原始"] + [f"del[{units[i]}]" for i in range(len(units))]

pca = PCA(n_components=2)
X_pca = pca.fit_transform(np.stack(features_for_pca))

plt.figure(figsize=(8,6))
plt.scatter(X_pca[0,0], X_pca[0,1], color='black', label='原始', marker='*', s=200)
for i in range(1, len(X_pca)):
    color = 'red' if i-1 in function_indices else ('blue' if i-1 in content_indices else 'grey')
    plt.scatter(X_pca[i,0], X_pca[i,1], color=color, alpha=0.7)
    if i-1 in function_indices or i-1 in content_indices:
        plt.text(X_pca[i,0], X_pca[i,1], units[i-1], fontsize=8, color=color)
plt.legend()
plt.title("所有单unit删除后的clip特征（PCA降维）")
plt.tight_layout()
plt.savefig("clip_unit_del_scatter.png")
plt.close()
print("所有删除unit后的clip特征分布已保存为 clip_unit_del_scatter.png")

# ----------------- 保存最可疑pair图片（假设有unit_del_images） -----------------
import os
os.makedirs("bug_pair_imgs", exist_ok=True)
for idx, r in enumerate(results[:N_SHOW]):
    fi = r['fi']
    ci = r['ci']
    # 假设变量 unit_del_images 可用
    try:
        unit_del_images[fi].save(f"bug_pair_imgs/bugpair_{idx+1}_f_{r['虚词unit']}.png")
        unit_del_images[ci].save(f"bug_pair_imgs/bugpair_{idx+1}_c_{r['实词unit']}.png")
    except Exception as e:
        print(f"保存图片bug: {e}")
print("最可疑pair的图像已保存到 bug_pair_imgs/")

# ----------------- 导出csv分析 -----------------
import pandas as pd
df = pd.DataFrame(results[:N_SHOW])
df.to_csv("clip_bug_pairs.csv", index=False)
print("最可疑虚实pair已导出到 clip_bug_pairs.csv")
