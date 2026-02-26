#!/bin/bash
# Start 2 vLLM servers with Qwen2.5-VL-72B, using tensor parallelism (4 GPUs each)

echo "=== Setting up HuggingFace cache symlink ==="
mkdir -p /opt/dlami/nvme/huggingface
echo "✅ HuggingFace cache directory ready at /opt/dlami/nvme/huggingface"

echo ""
echo "=== Starting vLLM servers with Qwen2.5-VL-72B ==="
echo "Each server uses 4 GPUs with tensor parallelism"
echo ""

# Run with workarounds, then disable one by one
export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 vllm serve Qwen/Qwen2.5-VL-72B-Instruct \
    --host 0.0.0.0 --port 8000 --dtype bfloat16 \
    --tensor-parallel-size 8 --max-model-len 8192 \
    --disable-custom-all-reduce \
    --enforce-eager \
    --gpu-memory-utilization 0.95 > /tmp/vllm_0.log 2>&1 &

# # Server 1: GPUs 0,1,2,3 on port 8000
# CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve Qwen/Qwen2.5-VL-72B-Instruct \
#     --host 0.0.0.0 \
#     --port 8000 \
#     --dtype bfloat16 \
#     --tensor-parallel-size 4 \
#     --max-model-len 1024 \
#     > /tmp/vllm_0.log 2>&1 &
# echo "Started server 1 on GPUs 0-3, port 8000"

# # Server 2: GPUs 4,5,6,7 on port 8001
# CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve Qwen/Qwen2.5-VL-72B-Instruct \
#     --host 0.0.0.0 \
#     --port 8001 \
#     --dtype bfloat16 \
#     --tensor-parallel-size 4 \
#     --max-model-len 1024 \
#     > /tmp/vllm_1.log 2>&1 &
# echo "Started server 2 on GPUs 4-7, port 8001"

echo ""
echo "Logs: /tmp/vllm_0.log, /tmp/vllm_1.log"
echo ""
echo "⚠️  First run will download ~150GB model weights - this takes 10-20 minutes!"
echo ""
echo "Waiting for servers to load..."
echo ""

# Wait and check status (longer timeout for large model)
for i in {1..60}; do
    sleep 10
    echo "Checking status... (attempt $i/60, $(($i * 10)) seconds elapsed)"
    
    ready=0
    for port in 8000 8001; do
        if curl -s "http://localhost:$port/health" > /dev/null 2>&1; then
            ((ready++))
        fi
    done
    
    echo "  $ready/2 servers ready"
    
    if [ $ready -eq 2 ]; then
        echo ""
        echo "✅ All 2 servers are ready!"
        echo ""
        echo "Stop all with: pkill -f vllm"
        exit 0
    fi
done

echo ""
echo "⚠️  Timeout: Not all servers ready after 10 minutes"
echo "Check logs: tail -f /tmp/vllm_*.log"
echo "Model might still be downloading. Check: du -sh /opt/dlami/nvme/huggingface/"