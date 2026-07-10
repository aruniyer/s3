#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${GENERATOR_GPU:-3}"

python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --port 8000 \
    --max-model-len 8192 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.65
