#!/usr/bin/env bash
# run_gpu1.sh — Retarget shard 1 of 2 on cuda:1
# Usage: bash /mnt/data1/rgz/soma-retargeter/run_gpu1.sh
# Input:  /mnt/data1/rgz/soma_uniform/bvh  (files at sorted indices 1,3,5,...)
# Output: /mnt/data1/rgz/soma-retargeter/adam_soma_pkl
# GPU:    cuda:1  |  Shard: 1/2

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "[INFO] Starting GPU1 shard retargeting (shard 1/2, cuda:1)"
echo "[INFO] Working directory: $SCRIPT_DIR"
echo "[INFO] Config: assets/gpu1_bvh_to_csv_converter_config.json"

uv run python ./app/bvh_to_csv_converter.py \
    --config ./assets/gpu1_bvh_to_csv_converter_config.json \
    --viewer null

echo "[INFO] GPU1 shard complete."
