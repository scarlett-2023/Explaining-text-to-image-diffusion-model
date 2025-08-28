import json
import os
import time
from typing import Dict, Any, List, Tuple, Optional
from PIL import Image
from datasets import Dataset, Features, Value, Image as HfImage
from tqdm import tqdm
from src.text_utils import remove_english_adverbs

def _load_coco_annotations(ann_path: str, verbose: bool = False) -> Tuple[Dict[int, str], Dict[int, str]]:
    """
    读取 COCO captions 标注，返回：
      - image_id -> 去副词后的 caption
      - image_id -> 原始 caption（首条）
    备注：一个 image_id 可能有多条 caption，这里选第一条用于训练以简化流程。
    """
    print(f"Opening annotation file: {ann_path}")
    start_time = time.time()
    
    with open(ann_path, "r", encoding="utf-8") as f:
        print("Parsing JSON...")
        data = json.load(f)
    
    json_time = time.time() - start_time
    print(f"JSON parsing completed in {json_time:.2f} seconds")
    
    annotations = data.get("annotations", [])
    print(f"Processing {len(annotations)} annotations...")
    
    image_to_caption = {}
    image_to_caption_raw = {}
    
    # 添加进度条以便监控处理过程
    for ann in tqdm(annotations, desc="Processing captions"):
        img_id = ann["image_id"]
        if img_id not in image_to_caption:
            raw = ann["caption"].strip()
            image_to_caption_raw[img_id] = raw
            processed = remove_english_adverbs(raw)
            if verbose and processed != raw:
                print(f"Original: {raw}")
                print(f"Processed: {processed}\n")
            image_to_caption[img_id] = processed
    
    print(f"Processed {len(image_to_caption)} unique images with captions")
    return image_to_caption, image_to_caption_raw

def _build_image_index(images_dir: str) -> Dict[int, str]:
    """
    根据 COCO 命名规则将 image_id 映射到具体路径。
    train2017/val2017 图片通常命名为 12 位零填充：000000000009.jpg
    train2014 通常命名为 COCO_train2014_000000000009.jpg
    """
    print(f"Indexing images in: {images_dir}")
    start_time = time.time()
    
    # 首先收集所有图像文件
    files = []
    for root, _, filenames in os.walk(images_dir):
        for fn in filenames:
            if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                files.append((root, fn))
    
    print(f"Found {len(files)} image files, building index...")
    
    idx = {}
    for root, fn in tqdm(files, desc="Building image index"):
        # 尝试从文件名提取id
        name = os.path.splitext(fn)[0]
        try:
            # 处理2014格式: COCO_train2014_000000000009
            if name.startswith("COCO_"):
                img_id = int(name.split("_")[-1])
            else:
                # 处理2017格式: 000000000009
                img_id = int(name)
        except ValueError:
            # 文件名格式不正确则跳过
            continue
        idx[img_id] = os.path.join(root, fn)
    
    index_time = time.time() - start_time
    print(f"Indexed {len(idx)} images in {index_time:.2f} seconds")
    return idx

def prepare_hf_dataset(
    raw_root: str,
    split: str,
    processed_root: str,
    verbose: bool = False
) -> str:
    """
    将 COCO 指定 split 构造成 datasets 格式，并对 caption 去副词。
    返回保存的本地 datasets 路径。
    
    参数:
        raw_root: COCO数据集根目录
        split: 数据集分割("train"或"val")
        processed_root: 处理后数据集保存目录
        verbose: 是否显示详细处理信息
    """
    overall_start = time.time()
    split = split.lower()
    assert split in {"train", "val"}
    
    # 根据split确定正确的年份
    year = "2014" if split == "train" else "2017"
    images_dir = os.path.join(raw_root, "images", f"{split}{year}")
    ann_path = os.path.join(raw_root, "annotations", f"captions_{split}{year}.json")
    
    # 检查文件和目录是否存在
    if not os.path.isfile(ann_path):
        raise FileNotFoundError(f"COCO captions JSON not found: {ann_path}")
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"COCO images dir not found: {images_dir}")

    # 加载标注
    imgid2cap, imgid2raw = _load_coco_annotations(ann_path, verbose)
    
    # 构建图像索引
    imgid2path = _build_image_index(images_dir)

    # 构建数据集记录
    print("Building dataset records...")
    records: List[Dict[str, Any]] = []
    missing = 0
    
    for img_id, cap in tqdm(imgid2cap.items(), desc="Preparing records"):
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

    # 创建HuggingFace数据集
    print("Creating HuggingFace Dataset...")
    features = Features({
        "image": HfImage(),
        "text": Value("string"),
        "text_raw": Value("string"),
        "image_id": Value("int64"),
    })
    
    ds = Dataset.from_list(records, features=features)

    # 保存数据集
    os.makedirs(processed_root, exist_ok=True)
    save_path = os.path.join(processed_root, split)
    print(f"Saving dataset to: {save_path}")
    ds.save_to_disk(save_path)
    
    overall_time = time.time() - overall_start
    print(f"Dataset preparation completed in {overall_time:.2f} seconds")
    print(f"Total images: {len(records)}")
    
    return save_path