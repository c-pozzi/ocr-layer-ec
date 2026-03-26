#!/bin/bash
# Start 4 vLLM servers with INT4 (AWQ) quantized model, one per GPU

echo "=== Setting up HuggingFace cache symlink ==="
# Fix symlink - create target directory if it doesn't exist
mkdir -p /opt/dlami/nvme/huggingface
echo "✅ HuggingFace cache directory ready at /opt/dlami/nvme/huggingface"

echo ""
echo "=== Starting vLLM servers (INT4 AWQ) on 4 GPUs ==="

CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen2.5-VL-7B-Instruct-AWQ --host 0.0.0.0 --port 8000 --quantization awq --dtype float16 --max-model-len 32768 > /tmp/vllm_0.log 2>&1 &
echo "Started server on GPU 0, port 8000"

CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen2.5-VL-7B-Instruct-AWQ --host 0.0.0.0 --port 8001 --quantization awq --dtype float16 --max-model-len 32768 > /tmp/vllm_1.log 2>&1 &
echo "Started server on GPU 1, port 8001"

CUDA_VISIBLE_DEVICES=2 vllm serve Qwen/Qwen2.5-VL-7B-Instruct-AWQ --host 0.0.0.0 --port 8002 --quantization awq --dtype float16 --max-model-len 32768 > /tmp/vllm_2.log 2>&1 &
echo "Started server on GPU 2, port 8002"

CUDA_VISIBLE_DEVICES=3 vllm serve Qwen/Qwen2.5-VL-7B-Instruct-AWQ --host 0.0.0.0 --port 8003 --quantization awq --dtype float16 --max-model-len 32768 > /tmp/vllm_3.log 2>&1 &
echo "Started server on GPU 3, port 8003"

echo ""
echo "Logs: /tmp/vllm_0.log, /tmp/vllm_1.log, /tmp/vllm_2.log, /tmp/vllm_3.log"
echo ""
echo "Waiting for servers to load (this takes ~1-2 minutes)..."
echo ""

# Wait and check status
for i in {1..24}; do
    sleep 5
    echo "Checking status... (attempt $i/24)"

    ready=0
    for port in 8000 8001 8002 8003; do
        if curl -s "http://localhost:$port/health" > /dev/null 2>&1; then
            ((ready++))
        fi
    done

    echo "  $ready/4 servers ready"

    if [ $ready -eq 4 ]; then
        echo ""
        echo "✅ All 4 servers are ready!"
        echo ""
        echo "Stop all with: pkill -f vllm"
        exit 0
    fi
done

echo ""
echo "⚠️  Timeout: Not all servers ready after 2 minutes"
echo "Check logs: tail -f /tmp/vllm_*.log"
