#!/bin/bash

# Start 8 vLLM servers, one per GPU
# Same model handles both classification and OCR (different prompts)

echo "=== Setting up HuggingFace cache ==="
mkdir -p /opt/dlami/nvme/huggingface
echo "✅ HuggingFace cache directory ready at /opt/dlami/nvme/huggingface"
echo ""

# Single model for both tasks
MODEL="Qwen/Qwen2.5-VL-7B-Instruct"

# Ports: 8000-8007 (GPU 0-7)

echo "=== Starting vLLM servers on GPUs 0-7 ==="
for gpu in {0..7}; do
    port=$((8000 + gpu))
    CUDA_VISIBLE_DEVICES=$gpu vllm serve $MODEL \
        --host 0.0.0.0 \
        --port $port \
        --dtype bfloat16 \
        --max-model-len 16384 \
        --gpu-memory-utilization 0.85 \
        > /tmp/vllm_$gpu.log 2>&1 &
    echo "Started server on GPU $gpu, port $port"
done

echo ""
echo "Logs: /tmp/vllm_*.log"
echo ""
echo "Waiting for servers to load (this takes ~1-2 minutes)..."
echo ""

# Wait and check status
TOTAL_SERVERS=8
for i in {1..30}; do
    sleep 5
    echo "Checking status... (attempt $i/30)"
    
    ready=0
    for port in {8000..8007}; do
        if curl -s "http://localhost:$port/health" > /dev/null 2>&1; then
            ((ready++))
        fi
    done
    
    echo "  $ready/$TOTAL_SERVERS servers ready"
    
    if [ $ready -eq $TOTAL_SERVERS ]; then
        echo ""
        echo "✅ All $TOTAL_SERVERS servers are ready!"
        echo ""
        echo "Endpoints: http://localhost:8000-8007"
        echo ""
        echo "Usage:"
        echo "  - Send classify prompt → get JSON tags"
        echo "  - Send OCR prompt (based on tags) → get transcription"
        echo ""
        echo "Stop all with: pkill -f vllm"
        exit 0
    fi
done

echo ""
echo "⚠️  Timeout: Not all servers ready after 2.5 minutes"
echo "Check logs: tail -f /tmp/vllm_*.log"