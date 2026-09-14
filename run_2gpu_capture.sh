#!/usr/bin/env bash
# Launch capture_attention_leakage.py sharded across 2 GPUs.
#
# Usage:
#   ./run_2gpu_capture.sh --exp_name my_exp [any other capture_attention_leakage.py args...]
#
# GPU ids can be overridden via env vars, e.g.:
#   GPU0=2 GPU1=3 ./run_2gpu_capture.sh --exp_name my_exp
set -euo pipefail

GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

echo "Launching shard 0 on GPU ${GPU0}..."
CUDA_VISIBLE_DEVICES="${GPU0}" python "${SCRIPT_DIR}/capture_attention_leakage.py" \
    --num_shards 2 --shard_id 0 "$@" \
    > "${LOG_DIR}/capture_shard0.log" 2>&1 &
PID0=$!

echo "Launching shard 1 on GPU ${GPU1}..."
CUDA_VISIBLE_DEVICES="${GPU1}" python "${SCRIPT_DIR}/capture_attention_leakage.py" \
    --num_shards 2 --shard_id 1 "$@" \
    > "${LOG_DIR}/capture_shard1.log" 2>&1 &
PID1=$!

echo "Shard 0 PID ${PID0} -> ${LOG_DIR}/capture_shard0.log"
echo "Shard 1 PID ${PID1} -> ${LOG_DIR}/capture_shard1.log"

wait "${PID0}" "${PID1}"
echo "Both shards finished."
