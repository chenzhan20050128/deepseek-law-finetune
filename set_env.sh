#!/usr/bin/env bash
set -e
cd $(dirname "$0")
echo "working directory $(pwd)"

# install dependencies required for CUDA-enabled PyTorch builds and compilation helpers
apt-get update
apt-get install -y --no-install-recommends build-essential git

export DS_MODELS_ROOT="${DS_MODELS_ROOT:-/root/autodl-tmp/models}"
mkdir -p "${DS_MODELS_ROOT}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
export HF_TOKEN="${HF_TOKEN:-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}"
export HF_DOWNLOAD_THREADS="${HF_DOWNLOAD_THREADS:-4}"
export HUGGINGFACE_HUB_URL="${HUGGINGFACE_HUB_URL:-https://mirrors.tuna.tsinghua.edu.cn/hugging-face-hub}"
export HF_DATASETS_URL="${HF_DATASETS_URL:-https://mirrors.tuna.tsinghua.edu.cn/hugging-face-datasets}"
export DS_HF_FALLBACK_ENDPOINT="${DS_HF_FALLBACK_ENDPOINT:-https://huggingface.co}"

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
