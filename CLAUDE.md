# OCR Evaluation Project

## Purpose

Benchmarking OCR quality on historical European Commission documents (BAC/INV series). Compares AI vision-language model OCR against existing PDF text layers, using human-annotated ground truth as the reference. The primary metric is **Character Error Rate (CER)**.

## Directory Structure

```
~/
├── ocr-evaluation-samples/          # INPUT ONLY — 31 document triplets
│   ├── <doc_id>.pdf                 # Scanned document PDFs with embedded OCR layer
│   ├── <doc_id>.tif                 # High-resolution TIFF scans (~2500x3500px)
│   ├── <doc_id>_ground_truth.txt    # Human-annotated reference text
│   ├── metadata.xlsx                # Document metadata
│   └── check_tif_def.py             # TIF/PDF resolution inspector utility
│
├── eval-scripts/                    # All evaluation scripts and results
│   ├── vllm/
│   │   ├── Qw7b/                    # Qwen2.5-VL-7B-Instruct scripts
│   │   │   ├── ocr-eval-vllm.py        # Main eval: PDF OCR vs Qwen vs ground truth
│   │   │   ├── ocr-eval-vllm-tif.py    # Full-resolution TIF variant
│   │   │   ├── ocr-eval-vllm-old.py    # Older version
│   │   │   ├── prompts.py              # Modular prompt system (classify-then-OCR)
│   │   │   ├── classify_prompts.py      # Document classification (multi-col, tables, etc.)
│   │   │   ├── ocr_pipeline.py          # Classification-based OCR pipeline
│   │   │   ├── benchmark_batching.py    # Throughput benchmarking
│   │   │   ├── start_vllm_servers.sh    # Launch 4 vLLM servers (1 per GPU, ports 8000-8003)
│   │   │   ├── start_vllm_servers_rc.sh
│   │   │   ├── start_vlllm_servers_nvidia.sh
│   │   │   ├── results/                 # Qwen 7B results (150dpi PDF)
│   │   │   ├── results_fullres/         # Qwen 7B results (full-res TIF, no downscaling)
│   │   │   ├── results_failed_run/      # Dead run (servers down, all CER=1.0)
│   │   │   ├── results_classify/        # Results from classification-based prompt pipeline
│   │   │   └── api/
│   │   │       └── ocr_comparison_app.py  # Streamlit diff viewer
│   │   └── Qw72b/                   # Qwen2.5-VL-72B-Instruct scripts
│   │       ├── ocr-eval-vllm.py         # 72B eval (tensor parallelism, 8 GPUs)
│   │       ├── start_vllm_72.sh         # Launch 72B server (all 8 GPUs)
│   │       ├── results/                 # Qwen 72B results
│   │       └── api/
│   │           └── ocr_comparison_app.py
│   └── max/                         # Modular MAX serve variant
│       ├── ocr-eval-max.py              # Same pipeline but targets MAX serve
│       └── start_max_7b.sh             # Launch MAX server (single GPU)
│
├── ocr-input/                       # Additional document folders (BAC/INV series, not in git)
└── archive/                         # Archived experiments (deepseek-ocr-vllm, not in git)
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

All AI model configs consistently beat PDF OCR layers (30/31 documents better, except INV-0013-2019-0926_0047 which is a difficult multi-page table document).

## Evaluation Pipeline

1. **Input**: PDF/TIF + ground truth pairs from `ocr-evaluation-samples/`
2. **PDF OCR extraction**: PyMuPDF (`fitz`) or pdfplumber to extract embedded text layer
3. **Image preparation**: PDF->image via `pdf2image` at configurable DPI, or native TIF
4. **VLM inference**: Images sent as base64 to vLLM/MAX OpenAI-compatible API
5. **CER calculation**: `jiwer` or `python-Levenshtein` on normalized text
6. **Output**: CSV results, per-document OCR text (raw + normalized), summary stats — written alongside each script in `eval-scripts/`

## Text Normalization (for CER)

Applied before CER comparison (`normalize_text()`):
- Remove markdown table formatting (`|`, `+`, `-`, `:`)
- Remove `<br>` tags
- Lowercase everything
- Collapse whitespace
- The classify pipeline also strips `[[TAG]]` annotation markers

## Serving Infrastructure

- **vLLM 7B**: 4 servers on ports 8000-8003, one GPU each (`--dtype bfloat16 --max-model-len 32768`)
- **vLLM 72B**: 1 server on port 8000, 8 GPUs with tensor parallelism (`--tensor-parallel-size 8`)
- **MAX serve**: Single GPU alternative (`max serve --model-path ... --max-length 8192`)
- Model weights cached at `/opt/dlami/nvme/huggingface`
- All servers expose OpenAI-compatible `/v1/chat/completions` endpoint

## Classification-Based Prompts (Advanced Pipeline)

`prompts.py` implements a two-pass approach:
1. **Classify**: Detect document features (multi-column, tables, handwritten, stamps, poor quality, strikethrough, non-Latin script, footnotes, forms)
2. **Build prompt**: Assemble modular prompt blocks based on classification results
3. Uses annotation tags: `[[H]]`, `[[C1]]`, `[[STAMP]]`, `[[S]]`, `[[LANG:xx]]`, `[[FN]]`, `[[HEADER]]`, `[[FOOTER]]`, `[[ILLEGIBLE]]`, `[[SUP]]`

## Ground Truth Format

- Plain text with preserved layout
- Tables use markdown pipe syntax with `<br>` for in-cell line breaks
- Page numbers included (e.g., "0258 - 3 -")
- Languages: primarily French, some Greek, German

## Python Dependencies

- `vllm`, `aiohttp`, `asyncio` (serving & async HTTP)
- `pdf2image`, `Pillow` (image handling)
- `pymupdf` (fitz), `pdfplumber` (PDF text extraction)
- `jiwer` or `python-Levenshtein` (CER calculation)
- `streamlit`, `pandas` (comparison UI)
- Conda environments: `max-ocr` (for MAX serve)

## Running Evaluations

```bash
# Start vLLM servers
bash ~/eval-scripts/vllm/Qw7b/start_vllm_servers.sh

# Run 7B evaluation
python ~/eval-scripts/vllm/Qw7b/ocr-eval-vllm.py ~/ocr-evaluation-samples/

# Run 72B evaluation
bash ~/eval-scripts/vllm/Qw72b/start_vllm_72.sh
python ~/eval-scripts/vllm/Qw72b/ocr-eval-vllm.py ~/ocr-evaluation-samples/

# Run classification-based pipeline
python ~/eval-scripts/vllm/Qw7b/classify_prompts.py   # First: classify docs
python ~/eval-scripts/vllm/Qw7b/ocr_pipeline.py       # Then: OCR with tailored prompts

# Launch comparison UI
streamlit run ~/eval-scripts/vllm/Qw7b/api/ocr_comparison_app.py
```

## Path Convention

All scripts use `Path(__file__).resolve().parent` to derive paths relative to their own location. Input data is read from `ocr-evaluation-samples/` and results are written alongside each script under `eval-scripts/`. No absolute `/home/ubuntu` paths exist in any Python script.

## Known Issues

- `eval-scripts/vllm/Qw7b/results_failed_run/` shows all CER=1.0 with 0s inference time — this was a run where vLLM servers were not responding (Qwen OCR outputs are empty)
- The `ocr_outputs` file in `ocr-evaluation-samples/` root is an empty file (0 bytes), not a directory
