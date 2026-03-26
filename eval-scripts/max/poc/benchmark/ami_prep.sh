#!/usr/bin/env bash
# =============================================================================
# AMI Preparation for MAX Serve Benchmark on p4d.24xlarge
#
# Run ONCE before benchmarking:
#   bash ami_prep.sh
# =============================================================================

set -euo pipefail

export HF_HOME=/opt/dlami/nvme/huggingface

echo "============================================"
echo "  AMI Prep: MAX Serve Benchmark"
echo "============================================"
echo ""

# ---- 1. GPU Topology ----
echo "=== Step 1: GPU Topology ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo ""

GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
if [ "$GPU_COUNT" -ne 8 ]; then
    echo "ERROR: Expected 8 GPUs, found $GPU_COUNT"
    exit 1
fi
echo "OK: $GPU_COUNT GPUs detected"
echo ""

# ---- 2. Download Base Models ----
echo "=== Step 2: Download Models ==="
echo "HF_HOME=$HF_HOME"
mkdir -p "$HF_HOME"

eval "$(conda shell.bash hook)"
conda activate max-ocr

echo "Downloading Qwen2.5-VL-7B-Instruct (base, not AWQ)..."
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct

echo ""
echo "Downloading Qwen2.5-VL-32B-Instruct (base, not AWQ)..."
huggingface-cli download Qwen/Qwen2.5-VL-32B-Instruct

echo ""
echo "OK: Models downloaded"
echo ""

# ---- 3. Verify Python Dependencies ----
echo "=== Step 3: Python Dependencies ==="
echo "Python: $(which python) ($(python --version))"
echo "MAX: $(max --version 2>&1 | head -1)"

DEPS=(aiohttp fitz pandas)
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
    exit 1
fi
echo ""

# ---- 4. Smoke Test: 7B q4_k on 1 GPU ----
echo "=== Step 4: Smoke Test — 7B q4_k on 1 GPU ==="

max serve \
    --model-path Qwen/Qwen2.5-VL-7B-Instruct \
    --quantization-encoding q4_k \
    --max-length 8192 \
    --devices gpu:0 \
    --port 9999 &
SMOKE_PID=$!

echo "  Waiting for server (PID $SMOKE_PID)..."
SMOKE_OK=false
for i in $(seq 1 120); do
    if curl -s http://localhost:9999/health > /dev/null 2>&1; then
        echo "  OK: 7B q4_k server healthy after ${i}s"
        SMOKE_OK=true
        break
    fi
    if ! kill -0 $SMOKE_PID 2>/dev/null; then
        echo "  ERROR: 7B server process exited"
        break
    fi
    sleep 1
done

kill $SMOKE_PID 2>/dev/null || true
wait $SMOKE_PID 2>/dev/null || true

if [ "$SMOKE_OK" = false ]; then
    echo "  WARNING: 7B smoke test failed"
fi
echo ""

# ---- 5. Smoke Test: 32B q4_k on 1 GPU ----
echo "=== Step 5: Smoke Test — 32B q4_k on 1 GPU ==="
echo "  (Tests whether 32B q4_k fits in single A100 40GB)"

max serve \
    --model-path Qwen/Qwen2.5-VL-32B-Instruct \
    --quantization-encoding q4_k \
    --max-length 16384 \
    --devices gpu:0 \
    --port 9999 &
SMOKE_PID=$!

echo "  Waiting for server (PID $SMOKE_PID)..."
SMOKE_OK=false
for i in $(seq 1 180); do
    if curl -s http://localhost:9999/health > /dev/null 2>&1; then
        echo "  OK: 32B q4_k server healthy after ${i}s — FITS on single A100!"
        SMOKE_OK=true
        break
    fi
    if ! kill -0 $SMOKE_PID 2>/dev/null; then
        echo "  WARNING: 32B q4_k process exited — may not fit on single A100"
        break
    fi
    sleep 1
done

kill $SMOKE_PID 2>/dev/null || true
wait $SMOKE_PID 2>/dev/null || true

if [ "$SMOKE_OK" = false ]; then
    echo "  NOTE: 32B q4_k on 1 GPU did not start. Will need 2+ GPUs."
fi
echo ""

# ---- 6. Check Input Data ----
echo "=== Step 6: Input Data ==="
INPUT_DIR="/home/ubuntu/ocr-input"
if [ -d "$INPUT_DIR" ] || [ -L "$INPUT_DIR" ]; then
    PDF_COUNT=$(find -L "$INPUT_DIR" -name "*.pdf" | wc -l)
    echo "  OK: $PDF_COUNT PDFs in $INPUT_DIR"
else
    echo "  WARNING: $INPUT_DIR not found"
fi
echo ""

# ---- Done ----
echo "============================================"
echo "  AMI Prep Complete"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Run: bash $(dirname "$0")/start_benchmark.sh"
