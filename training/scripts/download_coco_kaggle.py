import os
from kaggle.api.kaggle_api_extended import KaggleApi

"""
Script to download and extract COCO dataset from Kaggle.
The data will need to be manually organized into the following structure:
raw_root/
  images/train2017/*.jpg
  images/val2017/*.jpg
  annotations/captions_train2017.json
  annotations/captions_val2017.json
"""

WORK_BASE = os.environ.get("WORK_BASE", "./")  # Default to current directory
DEFAULT_OUTPUT_DIR = os.path.join(WORK_BASE, "data/raw/coco")

def main(output_dir=DEFAULT_OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    
    # Download dataset
    api = KaggleApi()
    api.authenticate()
    print("Downloading dataset from Kaggle...")
    api.dataset_download_files("nikhil7280/coco-image-caption", path=output_dir, quiet=False, unzip=True)
    
    print(f"Download and extraction complete. Files are in: {output_dir}")
    print("\nPlease manually organize the files into the following structure:")
    print("  images/train2017/*.jpg")
    print("  images/val2017/*.jpg")
    print("  annotations/captions_train2017.json")
    print("  annotations/captions_val2017.json")

if __name__ == "__main__":
    main()