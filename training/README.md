# LoRA 微调 Stable Diffusion（只用 diff loss，去副词改数据集）

目标：
- 通过“改数据集而不改模型”来纠正模型关注无关副词的问题：删除 caption 中的副词，将处理后的文本与原图配对训练。
- 使用 LoRA 只微调 UNet（冻结 VAE 与 CLIP 文本编码器）。
- 使用标准扩散噪声预测的 MSE（diff loss）。
- 小学习率 + 大 batch size 平滑梯度。
- 快速实验 50-100 步即可看到效果。

## 1. 准备环境

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

可选：安装并配置 `accelerate`
```bash
accelerate config
```
运行环境设置脚本
```bash
source env.sh 
```

## 2. 准备 COCO 数据

方式 A（自动下载，需要 Kaggle 凭证）：
- 将 `kaggle.json` 放置于 `~/.kaggle/kaggle.json`，或设置环境变量 `KAGGLE_USERNAME` 和 `KAGGLE_KEY`
- 运行：
```bash
python scripts/download_coco_kaggle.py
```
您可以通过以下方式验证数据是否下载到了正确的位置
```bash
ls -la "$WORK_BASE/data/raw/coco" 
```
数据应该会被下载到您设置的环境路径中（默认/projects/bepi/data/raw/coco）

完成后请检查目录结构是否类似：
```
data/raw/coco/
  images/train2014/*.jpg
  images/val2017/*.jpg
  annotations/captions_train2014.json
  annotations/captions_val2017.json
```
如果不类似请手动修改调整

方式 B（已有 COCO 本地）：将你的 COCO 目录指向 `config/train_config.yaml` 中的 `dataset.raw_root`。

## 3. 数据集去副词处理并保存为 datasets

训练集
```bash
python prepare_dataset.py --raw_root ./data/raw/coco --processed_root ./data/processed/coco_no_adverbs --split train
```
验证集
```bash
python prepare_dataset.py --raw_root ./data/raw/coco --processed_root ./data/processed/coco_no_adverbs --split val
```

## 4. 训练

编辑 `config/train_config.yaml` 可设置：
- `model_name`: 基座模型（如 `runwayml/stable-diffusion-v1-5`）
- `train.max_train_steps`: 建议先 50~100 做快速验证
- `train.learning_rate`: 小学习率，例如 5e-6
- `train.train_batch_size`: 尽量大，受限于显存
- `train.mixed_precision`: "fp16"（如果你的 GPU 支持）
- `train.enable_xformers`: True（建议）
- `train.lora_rank`: 8（可根据需求调整）

启动训练：
```bash
python train_lora_diffusion.py --config config/train_config.yaml
```

日志中会打印每步耗时（step_time），你可以据此估算训练样本处理速度与整体耗时。

在 `logging.output_dir` 下，每 `save_steps` 步会保存一次 LoRA 权重（UNet 注意力处理器权重）。

注意：
- 适用于 diffusers 0.34.0 和 PEFT 0.17.0
- 我们只训练 UNet 的 LoRA 权重；
- VAE 与 CLIP 文本编码器均冻结，不会更新；
- 训练损失为纯 “diff loss”（噪声预测 MSE）。

## 5. 推理测试

选择某个 checkpoint 目录，例如 `outputs/lora_sd15_no_adverbs/checkpoint-100`：

```bash
python infer.py \
  --model_name runwayml/stable-diffusion-v1-5 \
  --lora_dir outputs/lora_sd15_no_adverbs/checkpoint-100 \
  --prompt "a photo of a small red car on the street" \
  --num_inference_steps 30 \
  --guidance_scale 7.5 \
  --seed 1234 \
  --output outputs/sample.png
```

## 6. 关于 “只用 diff loss，且不动 text encoder”

- 训练时我们仅最小化 `MSE(noise_pred, noise)`，不加入额外损失；
- 冻结 `pipe.text_encoder`，不会对 CLIP 文本编码器进行任何更新；
- 通过对数据（caption 去副词）做干预，避免模型在训练信号中强化无关副词。

## 7. 适配不同 SD 模型

`config/train_config.yaml` 的 `model_name` 支持替换为不同基座（如 `stabilityai/stable-diffusion-2-1`）。其余流程一致。

## 8. 性能建议

- 如果显存允许，将 `train_batch_size` 调大、并保持较小 `learning_rate`，可获得更平滑的梯度；
- 使用 `xFormers` 与 `fp16` 能显著降低显存与加速训练；
- 初次尝试建议 `max_train_steps=50~100` 快速评估效果与训练速度；
- 观察日志中 `step_time` 的平均值，估算训练总时长。

## 9. 常见问题

- spaCy 报错找不到 `en_core_web_sm`：
  执行 `python -m spacy download en_core_web_sm`
- Kaggle 下载失败：检查 `~/.kaggle/kaggle.json` 或环境变量 `KAGGLE_USERNAME` / `KAGGLE_KEY`
- 显存不足：减少 `train_batch_size` 或启用 `--mixed_precision fp16`，并开启 `enable_xformers`
