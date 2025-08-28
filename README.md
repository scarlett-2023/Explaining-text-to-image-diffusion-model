# Explaining-text-to-image-diffusion-model
# Stable Diffusion LoRA Training Pipeline

This repository contains a complete pipeline for downloading, processing, training, and running inference with a LoRA-enabled Stable Diffusion model.

## Overview

The pipeline consists of four main components:

1. **Dataset Preparation**: Downloads and processes the COCO dataset, removing adverbs from captions
2. **Model Training**: Trains a LoRA adapter for Stable Diffusion 1.5
3. **Inference**: Generates images using the trained LoRA adapter
4. **Monitoring**: Includes memory monitoring for all stages of the pipeline

## Requirements

- Python 3.8+
- PyTorch 1.12+
- CUDA-capable GPU for training (16GB+ VRAM recommended)
- 50GB+ free disk space for the COCO dataset
- Dependencies: `diffusers`, `transformers`, `accelerate`, `datasets`, `safetensors`, `psutil`

To install requirements:

```bash
pip install torch torchvision diffusers transformers accelerate datasets safetensors psutil
```

## Directory Structure

All files are stored in `/work/nvme/bepi/sxie7/training` with the following structure:

```
/work/nvme/bepi/sxie7/training/
├── data/
│   ├── raw/
│   │   └── coco/            # Raw COCO dataset
│   └── processed/
│       └── coco_no_adverbs/ # Processed dataset with adverbs removed
├── outputs/
│   └── lora_sd15_noadverbs_a100/ # Training outputs and checkpoints
├── config/
│   └── train_config.yaml    # Training configuration
└── logs/                    # Log files
```

## Usage

### Full Pipeline

Run the complete pipeline:

```bash
python run_diffusion_model.py --all
```

Or run specific steps:

```bash
# Download COCO dataset
python run_diffusion_model.py --download

# Process dataset
python run_diffusion_model.py --process

# Train model
python run_diffusion_model.py --train

# Run inference with trained model
python run_diffusion_model.py --infer --checkpoint 100 --prompt "a photo of a small red car on the street"
```

### Dataset Preparation

To manually prepare the dataset:

```bash
python prepare_dataset.py --raw_root /work/nvme/bepi/sxie7/training/data/raw/coco --processed_root /work/nvme/bepi/sxie7/training/data/processed/coco_no_adverbs --split train
```

### Training

To manually train the model:

```bash
python train_lora_diffusion.py --config /work/nvme/bepi/sxie7/training/config/train_config.yaml
```

### Inference

To generate images with a trained model:

```bash
python infer.py --model_name runwayml/stable-diffusion-v1-5 --lora_dir /work/nvme/bepi/sxie7/training/outputs/lora_sd15_noadverbs_a100/checkpoint-100 --prompt "a photo of a small red car on the street" --output /work/nvme/bepi/sxie7/training/outputs/sample.png
```

## Memory Monitoring

All scripts include memory monitoring to help avoid out-of-memory and disk space issues. The monitoring tracks:

- Process RAM usage
- GPU memory (when available)
- System memory usage
- Disk space usage

Memory statistics are logged at key points during execution and can be found in the log files.

## Troubleshooting

### Out of Disk Space

If you encounter disk space issues:

1. Check available space: `df -h /work/nvme/bepi/sxie7/training`
2. Free up space by removing old output files or checkpoints
3. Use `--skip-download` flag if the dataset is already downloaded

### Training Crashes

If training crashes:

1. Check the logs for memory-related errors
2. Reduce batch size in the configuration
3. Enable gradient checkpointing for lower memory usage
4. Training will create emergency checkpoints when possible

### Download Issues

If dataset download fails:

1. The script will automatically retry up to 3 times
2. Check network connectivity
3. If partially downloaded, use `--force-extract` to attempt extraction again

## Configuration

Edit the training configuration in `/work/nvme/bepi/sxie7/training/config/train_config.yaml` to adjust:

- Model parameters
- LoRA rank and alpha
- Batch size and learning rate
- Training steps
- Checkpointing frequency

## Notes

- The pipeline creates log files with detailed execution information
- Checkpoint files are saved every N steps (configurable)
- All scripts monitor memory usage and disk space to prevent unexpected failures
- The dataset download and extraction may take 30+ minutes due to its size
