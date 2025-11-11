#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
echo "Working directory: $(pwd)"

: "${DS_MODELS_ROOT:=/root/autodl-tmp/models}"
export DS_MODELS_ROOT

: "${DS_HF_PRIMARY_ENDPOINT:=https://mirrors.tuna.tsinghua.edu.cn/hugging-face-hub}"
export DS_HF_PRIMARY_ENDPOINT

: "${DS_HF_FALLBACK_ENDPOINT:=https://huggingface.co}"
export DS_HF_FALLBACK_ENDPOINT

: "${HF_ENDPOINT:=https://hf-mirror.com}"
export HF_ENDPOINT

: "${HUGGINGFACE_HUB_ENDPOINT:=$HF_ENDPOINT}"
export HUGGINGFACE_HUB_ENDPOINT

: "${HF_HUB_DISABLE_TELEMETRY:=1}"
export HF_HUB_DISABLE_TELEMETRY

: "${PYTHONPATH:=}"
export PYTHONPATH="$(pwd)/src:${PYTHONPATH}"

BASE_MODEL_ID="${BASE_MODEL_ID:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}"
TEXT2VEC_MODEL_ID="${TEXT2VEC_MODEL_ID:-shibing624/text2vec-base-chinese}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-ds_r1_law_1.5B_exp4}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"

CLI_ARGS=(
	--base-model-id "${BASE_MODEL_ID}"
	--text2vec-model-id "${TEXT2VEC_MODEL_ID}"
	--experiment-name "${EXPERIMENT_NAME}"
	--output-dir "${OUTPUT_DIR}"
)

if [[ -n "${BASE_MODEL_PATH:-}" ]]; then
	CLI_ARGS+=(--base-model-path "${BASE_MODEL_PATH}")
fi

if [[ -n "${TEXT2VEC_MODEL_PATH:-}" ]]; then
	CLI_ARGS+=(--text2vec-model-path "${TEXT2VEC_MODEL_PATH}")
fi

python3 -m ds_finetune.cli train "${CLI_ARGS[@]}" "$@"

