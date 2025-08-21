import os
import argparse
from src.datasets.coco_caption_dataset import prepare_hf_dataset

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", type=str, default="./data/raw/coco")
    ap.add_argument("--processed_root", type=str, default="./data/processed/coco_no_adverbs")
    ap.add_argument("--split", type=str, default="train", choices=["train","val"])
    return ap.parse_args()

def main():
    args = parse_args()
    save_path = prepare_hf_dataset(
        raw_root=args.raw_root,
        split=args.split,
        processed_root=args.processed_root
    )
    print(f"Prepared dataset at: {save_path}")

if __name__ == "__main__":
    main()