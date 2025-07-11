import os
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt

RESULT_DIR = "importance_eval_results"
score_files = sorted(glob.glob(os.path.join(RESULT_DIR, "batch_*_replaced_scores.npy")))

if not score_files:
    raise RuntimeError("未找到批次分数文件（batch_*_replaced_scores.npy），请确认主脚本每批保存了分数！")

all_scores = []
for f in score_files:
    scores = np.load(f)
    all_scores.append(scores)
all_scores = np.concatenate(all_scores, axis=0)

mean_scores = np.mean(all_scores, axis=0)
std_scores = np.std(all_scores, axis=0)
print("全体Top1/2/3均值：", mean_scores)
print("全体Top1/2/3标准差：", std_scores)

plt.figure(figsize=(12, 7))
x = np.arange(1, all_scores.shape[0]+1)
for i, name in enumerate(['Top1', 'Top2', 'Top3']):
    plt.plot(x, all_scores[:, i], marker='o', label=f'{name} Replaced')
    plt.axhline(mean_scores[i], color=f"C{i}", linestyle="--", label=f'{name} Mean: {mean_scores[i]:.2f}')
plt.xlabel('Text Index')
plt.ylabel('CLIP Similarity (with original)')
plt.title('CLIP Score after Replacing Top1/2/3 Important Units (All)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "final_score_lines_replaced.png"))
plt.close('all')

df = pd.DataFrame(all_scores, columns=["top1", "top2", "top3"])
df.to_csv(os.path.join(RESULT_DIR, "all_scores_replaced.csv"), index=False)
print("最终合并分数表已保存到 all_scores_replaced.csv")
print("线图已保存到 final_score_lines_replaced.png")