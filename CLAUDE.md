# OCR Evaluation Project

## Purpose

Benchmarking OCR quality on historical European Commission documents (BAC/INV series). Compares AI vision-language model OCR against existing PDF text layers, using human-annotated ground truth as the reference. The primary metric is **Character Error Rate (CER)**.

## Directory Structure

```
~/
├── ocr-evaluation-samples/          # INPUT ONLY — document triplets
│   ├── <doc_id>.pdf                 # Scanned document PDFs with embedded OCR layer
│   ├── <doc_id>.tif                 # High-resolution TIFF scans (~2500x3500px)
│   ├── <doc_id>_ground_truth.txt    # Human-annotated reference text
│   ├── 20260216_OCR_files/          # Second batch (49 docs, nested subdir)
│   └── metadata.xlsx                # Document metadata
│
├── eval-scripts/                    # All evaluation scripts and results
│   ├── vllm/
│   │   ├── Qw7b/                    # Qwen2.5-VL-7B-Instruct scripts (original evals)
│   │   │   ├── ocr-eval-vllm.py        # Main eval: PDF OCR vs Qwen vs ground truth
│   │   │   ├── ocr-eval-vllm-tif.py    # Full-resolution TIF variant
│   │   │   ├── classify_prompts.py      # Document classification (legacy)
│   │   │   ├── ocr_pipeline.py          # Classification-based OCR pipeline (legacy)
│   │   │   ├── start_vllm_servers.sh    # Launch 4x 7B bf16 servers
│   │   │   ├── results/                 # Qwen 7B results (150dpi PDF)
│   │   │   ├── results_fullres/         # Qwen 7B results (full-res TIF)
│   │   │   └── api/
│   │   │       └── ocr_comparison_app.py  # Streamlit diff viewer (old)
│   │   │
│   │   ├── Qw72b/                   # Qwen2.5-VL-72B-Instruct scripts (original evals)
│   │   │   ├── ocr-eval-vllm.py
│   │   │   ├── start_vllm_72.sh
│   │   │   └── results/
│   │   │
│   │   └── poc/                     # ** ACTIVE ** Multi-model POC pipeline
│   │       ├── prompts.py               # Shared prompt system (classify + OCR blocks + versioning)
│   │       ├── poc_utils.py             # Shared utils (rendering, vLLM client, classification)
│   │       ├── poc_1_manifest.py        # Phase 1: build page manifest from PDFs
│   │       ├── poc_2_classify.py        # Phase 2: classify pages (async, producer-consumer)
│   │       ├── poc_3_ocr.py             # Phase 3: two-tier OCR (simple/complex routing)
│   │       ├── run_20260216.py          # Quick pipeline for 20260216 dataset
│   │       ├── compare_app.py           # Streamlit: AI OCR vs legacy PDF text vs ground truth
│   │       ├── start_vllm_32b_awq.sh    # Launch 2x 32B AWQ servers (TP=2, ports 8000-8001)
│   │       ├── start_vllm_72b_awq.sh    # Launch 72B AWQ server
│   │       ├── benchmark/               # A100 throughput benchmark suite
│   │       │   ├── benchmark_config.py      # Config matrix (15 configs: 8×7B + 7×32B)
│   │       │   ├── benchmark_server.py      # Server lifecycle: start/stop/health/GPU memory
│   │       │   ├── benchmark_runner.py      # Orchestrator: sample pages, loop configs, run workloads
│   │       │   ├── benchmark_report.py      # Summary tables, cost analysis, baseline comparison
│   │       │   ├── start_benchmark.sh       # Entry point (env setup, conda, run + report)
│   │       │   └── ami_prep.sh              # Pre-AMI: download models, verify deps, smoke tests
│   │       └── results/
│   │           ├── awq__BAC-0002-1971/          # 7B AWQ, first batch
│   │           ├── 32b-awq__BAC-0002-1971/      # 32B AWQ, first batch
│   │           ├── awq-lite__BAC-0002-1971/     # 7B lite classifier, first batch
│   │           └── 20260216/                     # Second batch results
│   │               ├── classification.csv            # Lite classifier output
│   │               ├── ocr_simple_v1/                # 7B AWQ, prompt v1
│   │               ├── ocr_complex_v1/               # 32B AWQ, prompt v1
│   │               └── ocr_complex_v2/               # 32B AWQ, prompt v2
│   │
│   └── max/                         # Modular MAX serve variant (future)
│       ├── ocr-eval-max.py
│       └── start_max_7b.sh
│
├── ocr-input/                       # Additional document folders (BAC/INV series, not in git)
└── archive/                         # Archived experiments (not in git)
```

## Document Naming Convention

- **BAC-XXXX-YYYY-ZZZZ_PPPP**: Historical commission documents (1960s-1990s, mostly French)
- **INV-XXXX-YYYY-ZZZZ_PPPP**: Investigation/inventory documents (2018-2023)
- Each doc has three files: `.pdf`, `.tif`, `_ground_truth.txt`

## Key Evaluation Results (from successful runs)

| Model/Config | Mean CER | Median CER | Notes |
|---|---|---|---|
| Qwen 7B (vLLM, 150dpi PDF) | ~5.5% | ~1.3% | `eval-scripts/vllm/Qw7b/results/` |
| Qwen 7B (full-res TIF) | ~4.8% | ~1.1% | `eval-scripts/vllm/Qw7b/results_fullres/` |
| Qwen 72B (8-GPU TP) | ~3.5% | ~0.8% | `eval-scripts/vllm/Qw72b/results/` |
| PDF embedded OCR layer | ~12.2% | ~6.7% | Baseline from existing PDFs |

## Two-Tier OCR Pipeline (POC)

The active pipeline in `eval-scripts/vllm/poc/` uses a classify-then-OCR approach:

1. **Classify** (lite profile): Detect 4 features — `multi_column`, `has_tables`, `poor_quality`, `latin_script`
2. **Route**: Simple pages → 7B AWQ (fast), Complex pages → 32B/72B AWQ (quality)
3. **OCR**: Modular prompts assembled from classification flags
4. **Compare**: Streamlit app for side-by-side inspection

### Complexity Routing (`is_complex`)

A page is complex if any of:
- `has_tables` is True
- `handwritten` is True
- `poor_quality` is True
- `latin_script` is False (non-Latin scripts like Greek, Arabic)

### Prompt Versioning

Prompts are versioned in `prompts.py` via `PROMPT_VERSIONS` dict. Each version can override base prompt, individual blocks, or the entire build function. Output goes to versioned directories (e.g., `ocr_simple_v1/`, `ocr_complex_v2/`).

- **v1**: Original prompts with all annotation tags and modular blocks
- **v2**: Updated TABLE_BLOCK with sparse form handling (skip empty columns)

### Running the POC Pipeline (20260216 dataset)

```bash
# Start 7B AWQ servers (4x A10G)
for i in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$i vllm serve Qwen/Qwen2.5-VL-7B-Instruct-AWQ \
        --port $((8000+i)) --quantization awq --dtype float16 --max-model-len 8192
done

# Classify + OCR simple pages
cd ~/eval-scripts/vllm/poc
python run_20260216.py --phase all-7b --prompt-version v1

# Stop 7B, start 32B AWQ (2 servers, TP=2)
pkill -f vllm
bash start_vllm_32b_awq.sh  # needs --max-model-len 16384 for complex pages

# OCR complex pages
python run_20260216.py --phase ocr-complex --prompt-version v2 \
    --servers http://localhost:8000,http://localhost:8001 \
    --model Qwen/Qwen2.5-VL-32B-Instruct-AWQ --max-tokens 8192

# Compare results
streamlit run compare_app.py
```

## Serving Infrastructure

- **vLLM 7B AWQ**: 4 servers on ports 8000-8003, one A10G each (`--quantization awq --dtype float16 --max-model-len 8192`)
- **vLLM 32B AWQ**: 2 servers on ports 8000-8001, TP=2 each (`--max-model-len 16384 --enforce-eager`)
- **vLLM 72B**: 1 server, 8 GPUs with tensor parallelism
- **MAX serve**: Single GPU alternative (future)
- Model weights cached at `/opt/dlami/nvme/huggingface` (set `HF_HOME`)
- All servers expose OpenAI-compatible `/v1/chat/completions` endpoint
- 32B AWQ needs `NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1` for multi-GPU

## Text Normalization (for CER)

Applied before CER comparison (`normalize_text()`):
- Remove markdown table formatting (`|`, `+`, `-`, `:`)
- Remove `<br>` tags
- Lowercase everything
- Collapse whitespace
- The classify pipeline also strips `[[TAG]]` annotation markers

## Ground Truth Format

- Plain text with preserved layout
- Tables use markdown pipe syntax with `<br>` for in-cell line breaks
- Page numbers included (e.g., "0258 - 3 -")
- Languages: primarily French, some Greek, German, Italian, Arabic

## Python Dependencies

- `vllm`, `aiohttp`, `asyncio` (serving & async HTTP)
- `pymupdf` (fitz) for PDF rendering to base64
- `pdftotext` (system) for legacy PDF text extraction
- `jiwer` or `python-Levenshtein` (CER calculation)
- `streamlit`, `pandas` (comparison UI)

## Path Convention

All scripts use `Path(__file__).resolve().parent` to derive paths relative to their own location. Input data is read from `ocr-evaluation-samples/` and results are written under `eval-scripts/`. The exception is `run_20260216.py` which has a hardcoded `INPUT_DIR` for the 20260216 dataset path.

## Known Issues

- `eval-scripts/vllm/Qw7b/results_failed_run/` — all CER=1.0, servers were down
- Legacy scripts in `Qw7b/` that imported `prompts.py` (now in `poc/`) are broken — `classify_prompts.py`, `ocr_pipeline.py`
- 32B AWQ with `--max-model-len 8192` causes context overflow for complex pages (image tokens + prompt + max_tokens). Use 16384.
- Some simple pages produce very short output (e.g., INV-0015-2019-0852_0437: 29 chars) — model may parrot prompt examples instead of OCR'ing

## A100 Benchmark Suite

Located in `eval-scripts/vllm/poc/benchmark/`. Tests optimal vLLM serving configs on p4d.24xlarge (8× A100 80GB) for the three OCR pipeline workloads.

### Goal

Find optimal throughput/latency/cost to handle ~9,000 pages/day. Baselines from g5.12xlarge (4× A10G): 7B=0.36 pages/sec, 32B=0.10 pages/sec.

### Config Matrix (15 configs)

**7B AWQ (8 configs)** — classifier + simple OCR workloads:
- Varies: eager vs CUDA graphs, gpu_mem_util (0.90/0.95), max_num_seqs (64/128/256), max_batched_tokens (default/16K/32K), TP=1 vs TP=2
- Each: 200 pages × 3 runs per workload

**32B AWQ (7 configs)** — complex OCR workload:
- Key test: TP=1 (impossible on A10G 24GB, may fit on A100 80GB → 8 servers)
- Varies: TP (1/2/4/8), eager vs graphs, max_num_seqs (32/64/128)
- Each: 100 pages × 3 runs

### Running on p4d.24xlarge

```bash
# 1. Prepare the AMI (once)
bash eval-scripts/vllm/poc/benchmark/ami_prep.sh

# 2. Run all benchmarks (~1.5-2 hours)
bash eval-scripts/vllm/poc/benchmark/start_benchmark.sh

# 3. Run subset
bash eval-scripts/vllm/poc/benchmark/start_benchmark.sh --configs 7b
bash eval-scripts/vllm/poc/benchmark/start_benchmark.sh --configs 32b_tp1_graphs_m95

# 4. Generate report only (after benchmark)
python eval-scripts/vllm/poc/benchmark/benchmark_report.py
```

### Architecture

- **benchmark_config.py**: `BenchmarkConfig` dataclass generates vllm serve args, CUDA_VISIBLE_DEVICES, server URLs
- **benchmark_server.py**: Subprocess management, health polling, nvidia-smi snapshots. NVLink-aware (no NCCL_P2P_DISABLE on p4d)
- **benchmark_runner.py**: Pre-renders 9K sampled pages to base64 in memory, then loops configs with semaphore-based async concurrency (8/server). Reuses `poc_utils.ocr_image_async`, `classify_image_async`, `prompts.build_simple_prompt_versioned`
- **benchmark_report.py**: Aggregates 3 runs → mean/std throughput, p50/p95/p99 latency, cost_per_1000_pages ($32.77/hr p4d rate), baseline speedup comparison

### Data

- Input: `/home/ubuntu/ocr-input/` — 2,400 PDFs, ~598K pages
- Samples 9,000 pages (seed=42), pre-renders at 150 DPI
- Results: `eval-scripts/vllm/poc/results/benchmark_a100/<config_id>/meta.json` + per-run CSVs
- Summary: `eval-scripts/vllm/poc/results/benchmark_a100/summary.csv`

### Python env

`deepseek-ocr` (conda). Deps: vllm, aiohttp, pymupdf (fitz), pandas.

## Current Branch

`refactor/scripts` — active development of POC pipeline
