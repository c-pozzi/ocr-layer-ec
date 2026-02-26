#!/bin/bash
# Start 8 vLLM servers with configurable parameters

echo "=== Setting up HuggingFace cache symlink ==="
# Fix symlink - create target directory if it doesn't exist
mkdir -p /opt/dlami/nvme/huggingface
echo "✅ HuggingFace cache directory ready at /opt/dlami/nvme/huggingface"

# === CONFIGURABLE PARAMETERS ===
MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-16}
GPU_MEMORY_UTIL=${GPU_MEMORY_UTIL:-0.9}
DTYPE="bfloat16"
BASE_PORT=8000

echo "=== Starting 8 vLLM servers ==="
echo "Model:           $MODEL"
echo "max-model-len:   $MAX_MODEL_LEN"
echo "max-num-seqs:    $MAX_NUM_SEQS"
echo "gpu-memory-util: $GPU_MEMORY_UTIL"
echo ""

# Kill any existing vLLM processes
pkill -f "vllm serve" 2>/dev/null
sleep 2

# Start 8 servers
for i in {0..7}; do
    PORT=$((BASE_PORT + i))
    LOG="/tmp/vllm_${i}.log"
    
    CUDA_VISIBLE_DEVICES=$i vllm serve $MODEL \
        --host 0.0.0.0 \
        --port $PORT \
        --dtype $DTYPE \
        --max-model-len $MAX_MODEL_LEN \
        --max-num-seqs $MAX_NUM_SEQS \
        --gpu-memory-utilization $GPU_MEMORY_UTIL \
        > $LOG 2>&1 &
    
    echo "Started GPU $i on port $PORT (log: $LOG)"
done

echo ""
echo "Waiting for servers to be ready..."

# Wait for all servers
for attempt in {1..60}; do
    ready=0
    for i in {0..7}; do
        PORT=$((BASE_PORT + i))
        if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
            ((ready++))
        fi
    done
    
    echo "  $ready/8 servers ready (attempt $attempt/60)"
    
    if [ $ready -eq 8 ]; then
        echo ""
        echo "✅ All 8 servers are ready!"
        echo ""
        echo "Ports: 8000-8007"
        echo "Stop all: pkill -f 'vllm serve'"
        exit 0
    fi
    
    sleep 5
done

echo "⚠️ Timeout: Not all servers ready"
echo "Check logs: tail -f /tmp/vllm_*.log"
exit 1