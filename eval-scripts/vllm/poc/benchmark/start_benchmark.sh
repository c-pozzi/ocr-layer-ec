#!/usr/bin/env bash
# =============================================================================
# A100 Benchmark Entry Point
#
# Sets environment, activates conda, and invokes the benchmark runner.
#
# Usage:
#   bash start_benchmark.sh                    # Run all configs
#   bash start_benchmark.sh --configs 7b       # Run only 7B configs
#   bash start_benchmark.sh --configs 32b      # Run only 32B configs
#   bash start_benchmark.sh --configs 7b_tp1_graphs_m95,32b_tp2_graphs_m95
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- Environment ----
export HF_HOME=/opt/dlami/nvme/huggingface
export TOKENIZERS_PARALLELISM=false

# Disable memory-hungry NCCL debug logging
export NCCL_DEBUG=WARN

# ---- Activate conda env ----
eval "$(conda shell.bash hook)"
conda activate deepseek-ocr

echo "============================================"
echo "  A100 vLLM Benchmark"
echo "============================================"
echo "  Python:    $(which python)"
echo "  HF_HOME:   $HF_HOME"
echo "  GPUs:      $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) x$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
echo "  Timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================"
echo ""

# ---- Run benchmark ----
python "$SCRIPT_DIR/benchmark_runner.py" "$@"

# ---- Generate report ----
echo ""
echo "Generating report..."
python "$SCRIPT_DIR/benchmark_report.py"

echo ""
echo "Done. Results in: eval-scripts/vllm/poc/results/benchmark_a100/"
