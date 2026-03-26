#!/usr/bin/env bash
# =============================================================================
# A100 MAX Serve Benchmark Entry Point
#
# Usage:
#   bash start_benchmark.sh                    # Run all configs
#   bash start_benchmark.sh --configs 7b       # Run only 7B configs
#   bash start_benchmark.sh --configs 32b      # Run only 32B configs
#   bash start_benchmark.sh --configs max_7b_q4k,max_32b_q4k_1gpu
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- Environment ----
export HF_HOME=/opt/dlami/nvme/huggingface
export TOKENIZERS_PARALLELISM=false

# ---- Activate conda env ----
eval "$(conda shell.bash hook)"
conda activate max-ocr

echo "============================================"
echo "  A100 MAX Serve Benchmark"
echo "============================================"
echo "  Python:    $(which python)"
echo "  MAX:       $(max --version 2>&1 | head -1)"
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
echo "Done. Results in: eval-scripts/max/poc/results/benchmark_a100/"
