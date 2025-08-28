#!/usr/bin/env bash
# 将所有缓存/数据/输出重定向到大盘
mkdir -p /work/nvme/bepi/sxie7/training
export WORK_BASE="/work/nvme/bepi/sxie7/training"

# 创建目录
mkdir -p "$WORK_BASE"/{data/raw,caches,tmp,outputs,workdir} \
      "$WORK_BASE/caches"/{huggingface,transformers,diffusers,datasets,torch,pip,xdg}

# 缓存与临时目录
export XDG_CACHE_HOME="$WORK_BASE/caches/xdg"
export HF_HOME="$WORK_BASE/caches/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$WORK_BASE/caches/datasets"
export TRANSFORMERS_CACHE="$WORK_BASE/caches/transformers"
export DIFFUSERS_CACHE="$WORK_BASE/caches/diffusers"
export TORCH_HOME="$WORK_BASE/caches/torch"
export PIP_CACHE_DIR="$WORK_BASE/caches/pip"

# 临时目录（解压/中间文件）
export TMPDIR="$WORK_BASE/tmp"
export TEMP="$WORK_BASE/tmp"
export TMP="$WORK_BASE/tmp"

echo "[env] WORK_BASE=$WORK_BASE"
echo "[env] Caches and TMP are redirected to $WORK_BASE"