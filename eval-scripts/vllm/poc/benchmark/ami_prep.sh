#!/usr/bin/env bash
# =============================================================================
# AMI Preparation Script for p4d.24xlarge
#
# Run this ONCE on a fresh p4d instance before benchmarking:
#   bash ami_prep.sh
#
# Checks: model weights, Python deps, GPU topology, smoke tests.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export HF_HOME=/opt/dlami/nvme/huggingface

echo "============================================"
echo "  AMI Prep: p4d.24xlarge for vLLM Benchmark"
echo "============================================"
echo ""

# ---- 1. GPU Topology ----
echo "=== Step 1: GPU Topology ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo ""
echo "GPU Topology (NVLink check):"
nvidia-smi topo -m
echo ""

# Verify 8 GPUs
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
if [ "$GPU_COUNT" -ne 8 ]; then
    echo "ERROR: Expected 8 GPUs, found $GPU_COUNT"
    exit 1
fi
echo "OK: $GPU_COUNT GPUs detected"
echo ""

# ---- 2. Download Models ----
echo "=== Step 2: Download Models to NVMe ==="
echo "HF_HOME=$HF_HOME"
mkdir -p "$HF_HOME"

echo "Downloading Qwen2.5-VL-7B-Instruct-AWQ..."
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct-AWQ

echo ""
echo "Downloading Qwen2.5-VL-32B-Instruct-AWQ..."
huggingface-cli download Qwen/Qwen2.5-VL-32B-Instruct-AWQ

echo ""
echo "OK: Models downloaded"
echo ""

# ---- 3. Verify Python Dependencies ----
echo "=== Step 3: Python Dependencies ==="

eval "$(conda shell.bash hook)"
conda activate deepseek-ocr

echo "Python: $(which python) ($(python --version))"

DEPS=(vllm aiohttp fitz pandas)
MISSING=()
for dep in "${DEPS[@]}"; do
    if python -c "import $dep" 2>/dev/null; then
        echo "  OK: $dep"
    else
        echo "  MISSING: $dep"
        MISSING+=("$dep")
    fi
done

if [ ${#MISSING[@]} -gt 0 ]; then
    echo ""
    echo "ERROR: Missing dependencies: ${MISSING[*]}"
    echo "Install with: pip install ${MISSING[*]}"
    exit 1
fi
echo ""

# ---- 4. Smoke Test: 7B on 1 GPU ----
echo "=== Step 4: Smoke Test — 7B AWQ on 1 GPU ==="

CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen2.5-VL-7B-Instruct-AWQ \
    --port 9999 --quantization awq --dtype float16 --max-model-len 8192 \
    --gpu-memory-utilization 0.90 &
SMOKE_PID=$!

echo "  Waiting for server (PID $SMOKE_PID)..."
for i in $(seq 1 60); do
    if curl -s http://localhost:9999/health | grep -q "ok\|200\|healthy" 2>/dev/null; then
        echo "  OK: 7B server healthy after ${i}s"
        break
    fi
    if [ "$i" -eq 60 ]; then
        echo "  ERROR: 7B server did not start within 60s"
        kill $SMOKE_PID 2>/dev/null || true
        wait $SMOKE_PID 2>/dev/null || true
        exit 1
    fi
    sleep 1
done

kill $SMOKE_PID 2>/dev/null || true
wait $SMOKE_PID 2>/dev/null || true
echo "  7B smoke test passed"
echo ""

# ---- 5. Smoke Test: 32B TP=1 on 1 GPU (A100 80GB) ----
echo "=== Step 5: Smoke Test — 32B AWQ TP=1 on 1 GPU ==="
echo "  (This tests whether 32B fits in a single A100 80GB)"

CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen2.5-VL-32B-Instruct-AWQ \
    --port 9999 --quantization awq --dtype float16 --max-model-len 16384 \
    --gpu-memory-utilization 0.90 &
SMOKE_PID=$!

echo "  Waiting for server (PID $SMOKE_PID)..."
SMOKE_OK=false
for i in $(seq 1 120); do
    if curl -s http://localhost:9999/health | grep -q "ok\|200\|healthy" 2>/dev/null; then
        echo "  OK: 32B TP=1 server healthy after ${i}s — FITS on single A100!"
        SMOKE_OK=true
        break
    fi
    # Check if process died
    if ! kill -0 $SMOKE_PID 2>/dev/null; then
        echo "  WARNING: 32B TP=1 process exited — may not fit on single A100"
        break
    fi
    sleep 1
done

kill $SMOKE_PID 2>/dev/null || true
wait $SMOKE_PID 2>/dev/null || true

if [ "$SMOKE_OK" = false ]; then
    echo "  NOTE: 32B TP=1 did not start. Will need TP>=2."
fi
echo ""

# ---- 6. Check Input Data ----
echo "=== Step 6: Input Data ==="
INPUT_DIR="/home/ubuntu/ocr-input"
if [ -d "$INPUT_DIR" ]; then
    PDF_COUNT=$(find "$INPUT_DIR" -name "*.pdf" | wc -l)
    echo "  OK: $PDF_COUNT PDFs in $INPUT_DIR"
else
    echo "  WARNING: $INPUT_DIR not found. Copy input data before benchmarking."
fi
echo ""

# ---- Done ----
echo "============================================"
echo "  AMI Prep Complete"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Verify input data is in $INPUT_DIR"
echo "  2. Run: bash $SCRIPT_DIR/start_benchmark.sh"
