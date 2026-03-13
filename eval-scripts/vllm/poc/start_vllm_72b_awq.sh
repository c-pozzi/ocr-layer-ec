#!/bin/bash
# Start 1 vLLM server with Qwen2.5-VL-72B-Instruct-AWQ
# Uses 4 GPUs with tensor parallelism (AWQ fits on 4x A10G)

echo "=== Setting up HuggingFace cache symlink ==="
mkdir -p /opt/dlami/nvme/huggingface
echo "HuggingFace cache directory ready at /opt/dlami/nvme/huggingface"

echo ""
echo "=== Starting vLLM server with Qwen2.5-VL-72B-Instruct-AWQ ==="
echo "Using 4 GPUs with tensor parallelism (TP=4)"
echo ""

# NCCL workarounds for multi-GPU communication
export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve Qwen/Qwen2.5-VL-72B-Instruct-AWQ \
    --host 0.0.0.0 --port 8000 \
    --tensor-parallel-size 4 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.95 \
    --enforce-eager \
    --disable-custom-all-reduce \
    > /tmp/vllm_72b_awq.log 2>&1 &

echo "Started server on GPUs 0-3, port 8000"
echo "Log: /tmp/vllm_72b_awq.log"
echo ""
echo "Waiting for server to load..."
echo ""

# Wait and check status (longer timeout for large model)
for i in {1..60}; do
    sleep 10
    echo "Checking status... (attempt $i/60, $(($i * 10)) seconds elapsed)"

    if curl -s "http://localhost:8000/health" > /dev/null 2>&1; then
        echo ""
        echo "Server is ready on port 8000!"
        echo ""
        echo "Stop with: pkill -f vllm"
        exit 0
    fi
done

echo ""
echo "Timeout: Server not ready after 10 minutes"
echo "Check log: tail -f /tmp/vllm_72b_awq.log"
echo "Model might still be downloading. Check: du -sh /opt/dlami/nvme/huggingface/"
