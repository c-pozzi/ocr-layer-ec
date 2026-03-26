#!/bin/bash
# =============================================================================
# OCR Pipeline Boot Entrypoint
#
# Intended to run via systemd on instance start, or manually for testing.
# Activates conda env, runs the pipeline, captures all output to a log file.
#
# Usage:
#   bash start_pipeline.sh                         # default S3 URIs
#   bash start_pipeline.sh --local-input /path     # local test mode
#   bash start_pipeline.sh --no-shutdown            # don't stop instance
# =============================================================================
set -euo pipefail

export HF_HOME=/opt/dlami/nvme/huggingface
export TOKENIZERS_PARALLELISM=false

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate deepseek-ocr

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="/opt/dlami/nvme/pipeline_logs/pipeline_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$LOG_FILE")"

echo "Starting OCR deployment pipeline at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Log: $LOG_FILE"

python "$SCRIPT_DIR/deploy_pipeline.py" "$@" 2>&1 | tee "$LOG_FILE"
