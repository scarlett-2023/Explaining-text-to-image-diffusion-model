import os
import sys
import subprocess
import torch

MAIN_SCRIPT = "main_batch.py"
TXT_PATH = "your_texts.txt"
BATCH_SIZE = 8

def get_best_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

with open(TXT_PATH, "r", encoding="utf-8") as f:
    texts = [line.strip() for line in f if line.strip()]

total = len(texts)
print(f"总文本数: {total}")
batches = (total + BATCH_SIZE - 1) // BATCH_SIZE

for i in range(batches):
    start = i * BATCH_SIZE
    end = min((i+1) * BATCH_SIZE, total)
    batch_txt = f"batch_{i+1}.txt"
    with open(batch_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(texts[start:end]))
    print(f"\n=== 处理第{i+1}批: {batch_txt} ({start+1}-{end}) ===")
    subprocess.run(
        [sys.executable, MAIN_SCRIPT, batch_txt, str(i+1)], check=True
    )
    os.remove(batch_txt)