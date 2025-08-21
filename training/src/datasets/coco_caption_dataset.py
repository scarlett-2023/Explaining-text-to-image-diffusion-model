import json
import os
from typing import Dict, Any, List, Tuple
from PIL import Image
from datasets import Dataset, Features, Value, Image as HfImage
from tqdm import tqdm
from src.text_utils import remove_english_adverbs

def _load_coco_annotations(ann_path: str) -> Tuple[Dict[int, str], Dict[int, str]]:
    """
    读取 COCO captions 标注，返回：
      - image_id -> 去副词后的 caption
      - image_id -> 原始 caption（首条）
    备注：一个 image_id 可能有多条 caption，这里选第一条用于训练以简化流程。
    """
    with open(ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # COCO captions JSON 结构：annotations 列表包含 image_id, caption
    image_to_caption = {}
    image_to_caption_raw = {}
    for ann in data.get("annotations", []):
        img_id = ann["image_id"]
        if img_id not in image_to_caption:
            raw = ann["caption"].strip()
            image_to_caption_raw[img_id] = raw
            processed = remove_english_adverbs(raw)
            image_to_caption[img_id] = processed
    return image_to_caption, image_to_caption_raw

def _build_image_index(images_dir: str) -> Dict[int, str]:
    """
    根据 COCO 命名规则将 image_id 映射到具体路径。
    train2017/val2017 图片通常命名为 12 位零填充：000000000009.jpg
    """
    idx = {}
    for root, _, files in os.walk(images_dir):
        for fn in files:
            if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            # 尝试从文件名提取 id
            name = os.path.splitext(fn)[0]
            try:
                img_id = int(name)
            except ValueError:
                # 文件名不是纯数字则跳过（COCO 一般是纯数字）
                continue
            idx[img_id] = os.path.join(root, fn)
    return idx

def prepare_hf_dataset(
    raw_root: str,
    split: str,
    processed_root: str
) -> str:
    """
    将 COCO 指定 split 构造成 datasets 格式，并对 caption 去副词。
    返回保存的本地 datasets 路径。
    预期 COCO 目录结构：
      raw_root/
        images/train2017/*.jpg
        images/val2017/*.jpg
        annotations/captions_train2017.json
        annotations/captions_val2017.json
    """
    split = split.lower()
    assert split in {"train", "val"}
    images_dir = os.path.join(raw_root, "images", f"{split}2017")
    ann_path = os.path.join(raw_root, "annotations", f"captions_{split}2017.json")

    if not os.path.isfile(ann_path):
        raise FileNotFoundError(f"COCO captions JSON not found: {ann_path}")
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"COCO images dir not found: {images_dir}")

    print(f"Loading annotations: {ann_path}")
    imgid2cap, imgid2raw = _load_coco_annotations(ann_path)
    print(f"Found {len(imgid2cap)} images with captions")

    print(f"Indexing images under: {images_dir}")
    imgid2path = _build_image_index(images_dir)
    print(f"Indexed {len(imgid2path)} images")

    records: List[Dict[str, Any]] = []
    missing = 0
    for img_id, cap in tqdm(imgid2cap.items(), desc="Building records"):
        p = imgid2path.get(img_id, None)
        if p and os.path.isfile(p):
            records.append({
                "image": p,
                "text": cap,
                "text_raw": imgid2raw.get(img_id, cap),
                "image_id": img_id,
            })
        else:
            missing += 1
    print(f"Valid records: {len(records)}, missing files: {missing}")

    features = Features({
        "image": HfImage(),
        "text": Value("string"),
        "text_raw": Value("string"),
        "image_id": Value("int64"),
    })
    ds = Dataset.from_list(records, features=features)

    os.makedirs(processed_root, exist_ok=True)
    save_path = os.path.join(processed_root, split)
    print(f"Saving dataset to: {save_path}")
    ds.save_to_disk(save_path)
    return save_path