# OCR Evaluation Scripts

Benchmarking OCR quality on historical European Commission documents (BAC/INV series). Compares AI vision-language model OCR against existing PDF text layers, using human-annotated ground truth as the reference.

**Primary metric:** Character Error Rate (CER)

## Directory Layout

```
eval-scripts/
├── vllm/
│   ├── Qw7b/                    # Qwen2.5-VL-7B-Instruct (original evals)
│   ├── Qw72b/                   # Qwen2.5-VL-72B-Instruct (original evals)
│   └── poc/                     # Active multi-model POC pipeline
│       ├── prompts.py               # Shared prompt system (versioned)
│       ├── poc_utils.py             # Shared utilities
│       ├── poc_1_manifest.py        # Phase 1: build page manifest
│       ├── poc_2_classify.py        # Phase 2: classify pages
│       ├── poc_3_ocr.py             # Phase 3: two-tier OCR
│       ├── cer_eval.py              # CER/WER evaluation
│       ├── compare_app.py           # Streamlit comparison viewer
│       ├── run_20260216.py          # Quick runner for 20260216 dataset
│       ├── benchmark/               # A100 throughput benchmark suite
│       └── results/                 # Evaluation results
│
└── max/                         # Modular MAX serve variant (future)
```

## Two-Tier OCR Pipeline

The active pipeline (`vllm/poc/`) uses a classify-then-OCR approach:

1. **Manifest** — Render PDF pages and build a page inventory
2. **Classify** — Detect features: `multi_column`, `has_tables`, `poor_quality`, `latin_script`
3. **Route** — Simple pages → 7B AWQ (fast) | Complex pages → 32B/72B AWQ (quality)
4. **OCR** — Modular prompts assembled from classification flags
5. **Evaluate** — CER/WER against ground truth, Streamlit diff viewer

A page is routed to the complex model if any of: `has_tables`, `handwritten`, `poor_quality`, or non-Latin script.

## Quick Start

### Prerequisites

- Conda environment: `deepseek-ocr`
- Python packages: `vllm`, `aiohttp`, `pymupdf`, `pandas`, `streamlit`, `jiwer`
- System: `pdftotext`
- Model weights cached at `/opt/dlami/nvme/huggingface` (`HF_HOME`)

### Running the Pipeline (20260216 dataset)

```bash
# 1. Start 7B AWQ servers (4× A10G)
for i in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$i vllm serve Qwen/Qwen2.5-VL-7B-Instruct-AWQ \
        --port $((8000+i)) --quantization awq --dtype float16 --max-model-len 8192
done

# 2. Classify + OCR simple pages
cd ~/eval-scripts/vllm/poc
python run_20260216.py --phase all-7b --prompt-version v1

# 3. Switch to 32B AWQ for complex pages (2 servers, TP=2)
pkill -f vllm
bash start_vllm_32b_awq.sh

# 4. OCR complex pages
python run_20260216.py --phase ocr-complex --prompt-version v2 \
    --servers http://localhost:8000,http://localhost:8001 \
    --model Qwen/Qwen2.5-VL-32B-Instruct-AWQ --max-tokens 8192

# 5. View results
streamlit run compare_app.py
```

### Running the A100 Benchmark

```bash
# Prepare AMI (once)
bash eval-scripts/vllm/poc/benchmark/ami_prep.sh

# Run all benchmarks (~1.5–2 hours on p4d.24xlarge)
bash eval-scripts/vllm/poc/benchmark/start_benchmark.sh

# Or run a subset
bash eval-scripts/vllm/poc/benchmark/start_benchmark.sh --configs 7b
```

## Key Results

| Model / Config | Mean CER | Median CER |
|---|---|---|
| Qwen 7B (150 dpi PDF) | ~5.5% | ~1.3% |
| Qwen 7B (full-res TIF) | ~4.8% | ~1.1% |
| Qwen 72B (8-GPU TP) | ~3.5% | ~0.8% |
| PDF embedded OCR (baseline) | ~12.2% | ~6.7% |

## Document Format

- **BAC-XXXX-YYYY-ZZZZ_PPPP** — Historical commission documents (1960s–1990s, mostly French)
- **INV-XXXX-YYYY-ZZZZ_PPPP** — Investigation/inventory documents (2018–2023)
- Each document has three files: `.pdf`, `.tif`, `_ground_truth.txt`

## Serving Configs

| Config | GPUs | Notes |
|---|---|---|
| 7B AWQ | 4× A10G, 1 per server | `--max-model-len 8192` |
| 32B AWQ | 2 servers, TP=2 each | `--max-model-len 16384 --enforce-eager` |
| 72B | 1 server, 8 GPUs TP | Full tensor parallelism |

32B AWQ multi-GPU requires `NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1` (except on NVLink systems like p4d).
