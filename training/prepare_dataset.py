import os
import argparse
import time
from src.datasets.coco_caption_dataset import prepare_hf_dataset

def parse_args():
    ap = argparse.ArgumentParser()
    work_base = os.environ.get("WORK_BASE", ".")
    
    default_raw = os.path.join(work_base, "data/raw/coco")
    default_processed = os.path.join(work_base, "data/processed/coco_no_adverbs")
    
    ap.add_argument("--raw_root", type=str, default=default_raw)
    ap.add_argument("--processed_root", type=str, default=default_processed)
    ap.add_argument("--split", type=str, default="train", choices=["train","val"])
    ap.add_argument("--verbose", action="store_true", 
                    help="显示详细处理信息")
    ap.add_argument("--preload_nlp", action="store_true",
                    help="预先加载NLP模型，避免处理时的延迟")
    return ap.parse_args()

def main():
    start_time = time.time()
    args = parse_args()
    print(f"Using raw_root: {args.raw_root}")
    print(f"Using processed_root: {args.processed_root}")
    
    # 预加载NLP模型
    if args.preload_nlp:
        print("Preloading spaCy model...")
        from src.text_utils import _load_en
        nlp = _load_en()
        print("spaCy model loaded successfully")
    
    save_path = prepare_hf_dataset(
        raw_root=args.raw_root,
        split=args.split,
        processed_root=args.processed_root,
        verbose=args.verbose
    )
    
    total_time = time.time() - start_time
    print(f"Prepared dataset at: {save_path}")
    print(f"Total processing time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()