# 基础配置，所有脚本都可import此文件
TXT_PATH = "your_texts.txt"              # 测试/分析文本列表

# 只用官方SD1.5模型作为base（如需自定义，可填写本地路径或HuggingFace真实仓库名）
BASE_MODELS = {
    "original": "runwayml/stable-diffusion-v1-5",
    # "finetuned": "your_finetuned_model_path_or_hf_repo",  # 只在本地有权重或HuggingFace有仓库时填写
    # "baseline": "your_baseline_model_path_or_hf_repo",
}
OUTPUT_ROOT = "full_pipeline_outputs"
BATCH_SIZE = 2
EPOCHS = 3
LR = 5e-6
DEVICE = "cuda"  # "cuda", "cpu" 或自动检测
K_IMPORTANCE = 0.3  # 前百分之多少的importance unit
FID_BATCH = 8