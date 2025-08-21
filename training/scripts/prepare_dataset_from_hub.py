import os
import argparse
from typing import Dict, Any, List, Set
from datasets import load_dataset, Dataset, Features, Value, Image as HfImage
from tqdm import tqdm
from src.text_utils import remove_english_adverbs

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_root", type=str, default="./data/processed/coco_no_adverbs")
    ap.add_argument("--split", type=str, default="train", choices=["train", "val"])
    ap.add_argument("--take_per_image", type=int, default=1, help="每张图取几条 caption（默认1条，足够训练纠偏）")
    return ap.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.processed_root, exist_ok=True)
    hf_split = "train" if args.split == "train" else "validation"

    print(f"Loading HF dataset: coco_captions 2017 split={hf_split}")
    ds = load_dataset("coco_captions", "2017", split=hf_split)

    # 该数据集每条样本是“一条 caption 对应一张 image”（同图有多条）
    # 我们仅保留每张图的前 N 条（默认 1 条），并对 caption 做去副词。
    seen_count: Dict[int, int] = {}
    records: List[Dict[str, Any]] = []

    for ex in tqdm(ds, desc="Building records"):
        image_id = int(ex["image_id"])
        cnt = seen_count.get(image_id, 0)
        if cnt >= args.take_per_image:
            continue
        raw = ex["caption"].strip()
        processed = remove_english_adverbs(raw)
        records.append({
            "image": ex["image"],   # datasets 的 Image 对象（延迟加载）
            "text": processed,
            "text_raw": raw,
            "image_id": image_id,
        })
        seen_count[image_id] = cnt + 1

    print(f"Unique images kept: {len(seen_count)}; total records: {len(records)}")

    features = Features({
        "image": HfImage(),
        "text": Value("string"),
        "text_raw": Value("string"),
        "image_id": Value("int64"),
    })
    out_ds = Dataset.from_list(records, features=features)

    save_path = os.path.join(args.processed_root, args.split)
    print(f"Saving to {save_path}")
    out_ds.save_to_disk(save_path)
    print("Done.")

if __name__ == "__main__":
    main()