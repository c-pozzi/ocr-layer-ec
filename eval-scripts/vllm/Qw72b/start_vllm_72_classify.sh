#!/bin/bash

# Start vLLM server with Qwen 72B using tensor parallelism across 8 GPUs
# Single server instance, all 8 GPUs working together

echo "=== Setting up HuggingFace cache ==="
mkdir -p /opt/dlami/nvme/huggingface
echo "✅ HuggingFace cache directory ready"
echo ""

# Kill any existing vLLM processes
echo "=== Stopping any existing vLLM servers ==="
pkill -f vllm || true
sleep 2
echo ""

MODEL="Qwen/Qwen2.5-VL-72B-Instruct"

echo "=== Starting Qwen 72B with tensor parallelism (8 GPUs) ==="
echo "Model: $MODEL"
echo "Port: 8000"
echo ""

# Launch with tensor parallelism across all 8 GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 vllm serve $MODEL \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype bfloat16 \
    --tensor-parallel-size 8 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.85 \
    > /tmp/vllm_72b.log 2>&1 &

echo "Server starting in background..."
echo "Log: /tmp/vllm_72b.log"
echo ""
echo "Waiting for server to load (72B takes ~3-5 minutes)..."
echo ""

# Wait and check status
for i in {1..60}; do
    sleep 5
    echo "Checking status... (attempt $i/60)"
    
    if curl -s "http://localhost:8000/health" > /dev/null 2>&1; then
        echo ""
        echo "✅ Server is ready!"
        echo ""
        echo "Endpoint: http://localhost:8000"
        echo ""
        echo "Stop with: pkill -f vllm"
        exit 0
    fi
    
    # Check if process died
    if ! pgrep -f "vllm serve" > /dev/null; then
        echo ""
        echo "❌ Server process died. Check log:"
        echo "tail -100 /tmp/vllm_72b.log"
        exit 1
    fi
done

echo ""
echo "⚠️  Timeout: Server not ready after 5 minutes"
echo "Check log: tail -f /tmp/vllm_72b.log"