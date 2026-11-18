#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
model=${1:?Usage: bash train/run_lora.sh clip-or-siglip gpu-index}
export CUDA_VISIBLE_DEVICES=${2:?GPU index required}
export HF_HOME=${HF_HOME:-/media/hdd2/xiaoying/model-weights/mmrep}
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=8
for rank in 8 16; do
  .venv/bin/python -u -m train.train_lora --model "$model" --rank "$rank" --config configs/lora.yaml
done
