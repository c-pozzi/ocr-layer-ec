#!/bin/bash
# Start MAX serve with Qwen2.5-VL-7B for testing

echo "=== Cleaning up ==="
pkill -9 -f "max serve" 2>/dev/null
pkill -9 -f vllm 2>/dev/null
sleep 2

# Clear caches (not HF model weights)
rm -rf ~/.cache/vllm/
rm -rf ~/.cache/max/
rm -rf ~/.cache/torch_compile_cache/
rm -rf /tmp/vllm* /tmp/max* /tmp/torch*

echo "=== Setting up HuggingFace cache symlink ==="
# # Use fast NVMe storage for HF cache
# if [ ! -L ~/.cache/huggingface ] && [ -d ~/.cache/huggingface ]; then
#     # Move existing cache to NVMe if not already a symlink
#     mkdir -p /opt/dlami/nvme/huggingface
#     mv ~/.cache/huggingface/* /opt/dlami/nvme/huggingface/ 2>/dev/null
#     rm -rf ~/.cache/huggingface
#     ln -s /opt/dlami/nvme/huggingface ~/.cache/huggingface
#     echo "✅ Moved HF cache to NVMe and created symlink"
# elif [ -L ~/.cache/huggingface ]; then
#     echo "✅ HF cache symlink already exists"
# else
#     mkdir -p /opt/dlami/nvme/huggingface
#     ln -s /opt/dlami/nvme/huggingface ~/.cache/huggingface
#     echo "✅ Created HF cache symlink to NVMe"
# fi
echo "=== Setting up HuggingFace cache symlink ==="
# Fix symlink - create target directory if it doesn't exist
mkdir -p /opt/dlami/nvme/huggingface
echo "✅ HuggingFace cache directory ready at /opt/dlami/nvme/huggingface"


echo ""
echo "=== GPU Status ==="
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv
echo ""

echo "=== Starting MAX serve with Qwen2.5-VL-7B ==="
echo "Model: Qwen/Qwen2.5-VL-7B-Instruct"
echo "Using single GPU for 7B model"
echo ""

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate max-ocr

# Start MAX serve (single GPU for 7B is enough)
nohup max serve \
    --model-path Qwen/Qwen2.5-VL-7B-Instruct \
    --max-length 8192 \
    --devices gpu:0 \
    > /tmp/max_7b.log 2>&1 &
disown

echo "Server starting in background..."
echo "Log: /tmp/max_7b.log"
echo ""
echo "Waiting for server to be ready..."

# Wait for server to start
for i in {1..60}; do
    sleep 5
    echo "Checking status... (attempt $i/60, $((i * 5))s elapsed)"
    
    # Check if process is still running
    if ! pgrep -f "max serve" > /dev/null; then
        echo "❌ MAX serve process died. Check log:"
        tail -50 /tmp/max_7b.log
        exit 1
    fi
    
    # Check health endpoint
    if curl -s "http://localhost:8000/health" > /dev/null 2>&1; then
        echo ""
        echo "✅ Server is ready!"
        echo ""
        echo "Test with:"
        echo "  curl http://localhost:8000/v1/models"
        echo ""
        echo "Stop with:"
        echo "  pkill -f 'max serve'"
        exit 0
    fi
    
    # Show last few log lines periodically
    if [ $((i % 6)) -eq 0 ]; then
        echo "--- Recent log ---"
        tail -5 /tmp/max_7b.log
        echo "------------------"
    fi
done

echo ""
echo "⚠️ Timeout after 5 minutes. Server may still be loading."
echo "Check log: tail -f /tmp/max_7b.log"
EOF

chmod +x ~/start_max_7b.sh
echo "Created ~/start_max_7b.sh"