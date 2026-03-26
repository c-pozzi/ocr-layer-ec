#!/bin/bash
# Start 4x MAX serve instances with Qwen2.5-VL-7B-Instruct (one per A10G)
# Mirrors the vLLM 4-server setup for throughput comparison.

set -e

MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
QUANT="q4_k"
MAX_LENGTH=8192
BASE_PORT=8000
NUM_GPUS=4

echo "=== Cleaning up ==="
pkill -9 -f "max serve" 2>/dev/null || true
sleep 2

echo ""
echo "=== GPU Status ==="
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv
echo ""

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate max-ocr

# Use NVMe for HF cache
export HF_HOME=/opt/dlami/nvme/huggingface

echo "=== Starting $NUM_GPUS MAX serve instances ==="
echo "Model: $MODEL"
echo "Quantization: $QUANT"
echo "Max length: $MAX_LENGTH"
echo "Ports: $BASE_PORT-$((BASE_PORT + NUM_GPUS - 1))"
echo ""

for i in $(seq 0 $((NUM_GPUS - 1))); do
    PORT=$((BASE_PORT + i))
    LOG="/tmp/max_7b_gpu${i}.log"

    echo "Starting GPU $i on port $PORT..."
    nohup max serve \
        --model-path "$MODEL" \
        --quantization-encoding "$QUANT" \
        --max-length "$MAX_LENGTH" \
        --devices "gpu:$i" \
        --port "$PORT" \
        > "$LOG" 2>&1 &
    disown
    echo "  PID: $! | Log: $LOG"
done

echo ""
echo "Waiting for servers to be ready..."

# Wait for all servers to become healthy
ALL_READY=false
for attempt in $(seq 1 120); do
    sleep 5
    READY=0
    for i in $(seq 0 $((NUM_GPUS - 1))); do
        PORT=$((BASE_PORT + i))
        if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
            READY=$((READY + 1))
        fi
    done

    echo "  Attempt $attempt: $READY/$NUM_GPUS servers ready (${attempt}x5s elapsed)"

    if [ "$READY" -eq "$NUM_GPUS" ]; then
        ALL_READY=true
        break
    fi

    # Show logs periodically
    if [ $((attempt % 12)) -eq 0 ]; then
        for i in $(seq 0 $((NUM_GPUS - 1))); do
            echo "--- GPU $i last 3 lines ---"
            tail -3 "/tmp/max_7b_gpu${i}.log"
        done
    fi
done

echo ""
if $ALL_READY; then
    echo "All $NUM_GPUS servers ready!"
    echo ""
    echo "Test with:"
    for i in $(seq 0 $((NUM_GPUS - 1))); do
        echo "  curl http://localhost:$((BASE_PORT + i))/v1/models"
    done
    echo ""
    echo "Run OCR benchmark:"
    echo "  cd ~/eval-scripts/vllm/poc"
    echo "  python poc_3_ocr.py --run-dir ~/eval-scripts/max/results/max-7b__BAC-0002-1971 --tier simple --model $MODEL --servers http://localhost:8000,http://localhost:8001,http://localhost:8002,http://localhost:8003"
else
    echo "WARNING: Not all servers ready after 10 minutes."
    echo "Check logs: tail -f /tmp/max_7b_gpu*.log"
fi

echo ""
echo "Stop all with: pkill -f 'max serve'"
