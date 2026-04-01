"""
Simple standalone OCR test: send one image to Qwen 7B AWQ and 32B AWQ servers.

Usage:
    # Start servers first (see below), then:
    python test_ocr_simple.py /path/to/image.png

Servers:
    # GPU 0: 7B AWQ on port 8000
    CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen2.5-VL-7B-Instruct-AWQ \
        --port 8000 --quantization awq --dtype float16 --max-model-len 8192

    # GPUs 1,2: 32B AWQ on port 8001 (TP=2)
    NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
    CUDA_VISIBLE_DEVICES=1,2 vllm serve Qwen/Qwen2.5-VL-32B-Instruct-AWQ \
        --port 8001 --tensor-parallel-size 2 --quantization awq --dtype float16 \
        --max-model-len 16384 --gpu-memory-utilization 0.95 --enforce-eager
"""

import sys
import base64
import requests
from pathlib import Path

PROMPT = "Transcribe all the text in this image exactly as written. Preserve formatting, punctuation, and capitalization."

SERVERS = {
    "Qwen 7B AWQ":  {"url": "http://localhost:8000/v1/chat/completions", "model": "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"},
    "Qwen 32B AWQ": {"url": "http://localhost:8001/v1/chat/completions", "model": "Qwen/Qwen2.5-VL-32B-Instruct-AWQ"},
}


def encode_image(path: str) -> str:
    data = Path(path).read_bytes()
    return base64.b64encode(data).decode()


def ocr_request(url: str, model: str, image_b64: str) -> str:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                    {"type": "text", "text": PROMPT},
                ],
            }
        ],
        "max_tokens": 4096,
        "temperature": 0.0,
    }
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    if not Path(image_path).exists():
        print(f"File not found: {image_path}")
        sys.exit(1)

    print(f"Image: {image_path}\n")
    image_b64 = encode_image(image_path)

    for name, cfg in SERVERS.items():
        print(f"{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")
        try:
            result = ocr_request(cfg["url"], cfg["model"], image_b64)
            print(result)
        except requests.ConnectionError:
            print(f"  [SERVER NOT RUNNING on {cfg['url']}]")
        except Exception as e:
            print(f"  [ERROR: {e}]")
        print()


if __name__ == "__main__":
    main()
