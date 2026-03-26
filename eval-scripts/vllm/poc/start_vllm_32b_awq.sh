#!/bin/bash
# Start 2 vLLM servers with Qwen2.5-VL-32B-Instruct-AWQ, TP=2 each
# 32B AWQ (~18GB weights) OOMs on a single A10G (24GB), so we use 2 GPUs
# per server (~46GB available) for comfortable KV cache headroom.
# 2 servers x 2 GPUs = 4 GPUs total on g5.12xlarge

echo "=== Setting up HuggingFace cache symlink ==="
mkdir -p /opt/dlami/nvme/huggingface
echo "HuggingFace cache directory ready at /opt/dlami/nvme/huggingface"

echo ""
echo "=== Starting 2 vLLM servers (32B AWQ, TP=2) ==="
echo "Server 0: GPUs 0,1 -> port 8000"
echo "Server 1: GPUs 2,3 -> port 8001"
echo ""

# NCCL workarounds for multi-GPU communication
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

CUDA_VISIBLE_DEVICES=0,1 vllm serve Qwen/Qwen2.5-VL-32B-Instruct-AWQ \
    --host 0.0.0.0 --port 8000 \
    --tensor-parallel-size 2 \
    --quantization awq \
    --dtype float16 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.95 \
    --enforce-eager \
    > /tmp/vllm_32b_0.log 2>&1 &
echo "Started server on GPUs 0,1 -> port 8000"

CUDA_VISIBLE_DEVICES=2,3 vllm serve Qwen/Qwen2.5-VL-32B-Instruct-AWQ \
    --host 0.0.0.0 --port 8001 \
    --tensor-parallel-size 2 \
    --quantization awq \
    --dtype float16 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.95 \
    --enforce-eager \
    > /tmp/vllm_32b_1.log 2>&1 &
echo "Started server on GPUs 2,3 -> port 8001"

echo ""
echo "Logs: /tmp/vllm_32b_0.log, /tmp/vllm_32b_1.log"
echo ""
echo "Waiting for servers to load (this takes ~3-5 minutes for 32B)..."
echo ""

# Wait and check status
for i in {1..60}; do
    sleep 5
    echo "Checking status... (attempt $i/60, $(($i * 5)) seconds elapsed)"

    ready=0
    for port in 8000 8001; do
        if curl -s "http://localhost:$port/health" > /dev/null 2>&1; then
            ((ready++))
        fi
    done

    echo "  $ready/2 servers ready"

    if [ $ready -eq 2 ]; then
        echo ""
        echo "All 2 servers are ready!"
        echo ""
        echo "Stop all with: pkill -f vllm"
        exit 0
    fi
done

echo ""
echo "Timeout: Not all servers ready after 5 minutes"
echo "Check logs: tail -f /tmp/vllm_32b_*.log"
