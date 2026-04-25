#!/usr/bin/env bash
# run_gpu0.sh — Retarget shard 0 of 2 on cuda:0
# Usage: bash /mnt/data1/rgz/soma-retargeter/run_gpu0.sh
# Input:  /mnt/data1/rgz/soma_uniform/bvh  (files at sorted indices 0,2,4,...)
# Output: /mnt/data1/rgz/soma-retargeter/adam_soma_pkl
# GPU:    cuda:0  |  Shard: 0/2

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "[INFO] Starting GPU0 shard retargeting (shard 0/2, cuda:0)"
echo "[INFO] Working directory: $SCRIPT_DIR"
echo "[INFO] Config: assets/gpu0_bvh_to_csv_converter_config.json"

uv run python ./app/bvh_to_csv_converter.py \
    --config ./assets/gpu0_bvh_to_csv_converter_config.json \
    --viewer null

echo "[INFO] GPU0 shard complete."
