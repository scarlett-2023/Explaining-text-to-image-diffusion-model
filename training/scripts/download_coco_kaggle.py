import os
import zipfile
import shutil
from kaggle.api.kaggle_api_extended import KaggleApi

"""
需要提前配置 Kaggle 凭证：
- 将 kaggle.json 放在 ~/.kaggle/kaggle.json，或设置环境变量 KAGGLE_USERNAME / KAGGLE_KEY
下载的数据集（nikhil7280/coco-image-caption）解压后需整理成：
raw_root/
  images/train2017/*.jpg
  images/val2017/*.jpg
  annotations/captions_train2017.json
  annotations/captions_val2017.json
部分 Kaggle 包可能只包含 train 或不同命名，你需要根据实际内容做轻微调整。
"""

def main(output_dir="./data/raw/coco"):
    os.makedirs(output_dir, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    print("Downloading dataset from Kaggle...")
    api.dataset_download_files("nikhil7280/coco-image-caption", path=output_dir, quiet=False, unzip=True)

    # 尝试自动识别并归位常见结构
    # 若解压后出现多级目录，尽量移动到目标结构
    print("Post-processing files...")
    candidates = []
    for root, dirs, files in os.walk(output_dir):
        for f in files:
            if f.startswith("captions_") and f.endswith(".json"):
                candidates.append(os.path.join(root, f))
    ann_dir = os.path.join(output_dir, "annotations")
    os.makedirs(ann_dir, exist_ok=True)
    for p in candidates:
        shutil.move(p, os.path.join(ann_dir, os.path.basename(p)))

    # 尝试归位图片目录（train2017/val2017）
    for split in ["train2017", "val2017"]:
        split_dir = None
        for root, dirs, files in os.walk(output_dir):
            if os.path.basename(root) == split:
                split_dir = root
                break
        if split_dir:
            target = os.path.join(output_dir, "images", split)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            if os.path.abspath(split_dir) != os.path.abspath(target):
                shutil.move(split_dir, target)

    print("Done. Please verify directory structure as documented in the script docstring.")

if __name__ == "__main__":
    main()