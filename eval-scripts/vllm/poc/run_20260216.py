#!/usr/bin/env python3
"""
Quick OCR pipeline for 20260216_OCR_files samples.

Single-page PDFs: classify (lite) → route → OCR → save text for inspection.
Simple pages → 7B AWQ, Complex pages → 32B AWQ (or configurable).

Usage:
    # Step 1: Start 7B AWQ servers (4x A10G)
    bash start_vllm_servers.sh   # from parent dir

    # Step 2: Classify + OCR simple pages (7B)
    python run_20260216.py --phase all-7b

    # Step 3: Stop 7B, start 32B AWQ servers (2x TP=2)
    pkill -f vllm && bash start_vllm_32b_awq.sh

    # Step 4: OCR complex pages (32B)
    python run_20260216.py --phase ocr-complex

    # Or do everything in one shot if 7B servers are up:
    python run_20260216.py --phase classify
    python run_20260216.py --phase ocr-simple
    python run_20260216.py --phase ocr-complex --servers http://localhost:8000,http://localhost:8001 --model Qwen/Qwen2.5-VL-32B-Instruct-AWQ

Output:
    results/20260216/
    ├── classification.csv        # lite classification results
    ├── ocr_simple/               # OCR text files for simple pages
    │   ├── BAC-0042-1988-1454_0074.txt
    │   └── ...
    ├── ocr_complex/              # OCR text files for complex pages
    │   ├── BAC-xxxx-...txt
    │   └── ...
    ├── summary.csv               # one row per doc: classification + ocr status
    └── ocr_all/                  # symlinks/copies for easy browsing
"""

import argparse
import asyncio
import csv
import json
import sys
import time
from pathlib import Path

import aiohttp

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from poc_utils import (
    render_page_to_base64,
    parse_classification_json,
    is_complex,
    check_server_health,
    VLLM_ENDPOINT,
)
from prompts import (
    get_classify_prompt,
    build_prompt_versioned,
    build_simple_prompt_versioned,
    list_prompt_versions,
)

# =============================================================================
# DEFAULTS
# =============================================================================

INPUT_DIR = Path("/home/ubuntu/ocr-evaluation-samples/20260216_OCR_files/20260216_OCR_files")
RESULTS_DIR = SCRIPT_DIR / "results" / "20260216"

DEFAULT_7B_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"
DEFAULT_32B_MODEL = "Qwen/Qwen2.5-VL-32B-Instruct-AWQ"
DEFAULT_7B_SERVERS = [f"http://localhost:{8000 + i}" for i in range(4)]
DEFAULT_32B_SERVERS = ["http://localhost:8000", "http://localhost:8001"]


def parse_args():
    parser = argparse.ArgumentParser(description="Quick OCR pipeline for 20260216 samples")
    parser.add_argument(
        "--phase",
        choices=["classify", "ocr-simple", "ocr-complex", "all-7b"],
        required=True,
        help="Which phase to run",
    )
    parser.add_argument("--input-dir", type=str, default=str(INPUT_DIR))
    parser.add_argument("--results-dir", type=str, default=str(RESULTS_DIR))
    parser.add_argument("--model", type=str, default=None, help="Override model name")
    parser.add_argument("--servers", type=str, default=None, help="Comma-separated server URLs")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--concurrency", type=int, default=4, help="Total concurrent requests")
    parser.add_argument("--prompt-version", type=str, default="v1", help="Prompt version (default: v1)")
    parser.add_argument("--list-versions", action="store_true", help="List available prompt versions and exit")
    return parser.parse_args()


# =============================================================================
# SCAN INPUT
# =============================================================================

def scan_samples(input_dir: Path) -> list[dict]:
    """Find all PDFs that have a matching ground truth file.

    Only returns docs with both PDF and ground truth (the 47 complete
    evaluation pairs). Docs without GT are skipped since we can't
    compute CER/WER for them.
    """
    samples = []
    for pdf_path in sorted(input_dir.glob("*.pdf")):
        doc_id = pdf_path.stem
        gt_path = input_dir / f"{doc_id}_ground_truth.txt"
        if not gt_path.exists():
            print(f"  SKIP {doc_id}: no ground truth")
            continue
        tif_path = input_dir / f"{doc_id}.tif"
        samples.append({
            "doc_id": doc_id,
            "pdf_path": pdf_path,
            "tif_path": tif_path if tif_path.exists() else None,
            "gt_path": gt_path,
        })
    return samples


# =============================================================================
# CLASSIFY
# =============================================================================

async def classify_one(
    session: aiohttp.ClientSession,
    server_url: str,
    image_b64: str,
    model_name: str,
) -> dict:
    """Classify a single image, return parsed dict."""
    prompt = get_classify_prompt("lite")
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "max_tokens": 256,
        "temperature": 0.0,
    }

    t0 = time.monotonic()
    try:
        async with session.post(
            f"{server_url}{VLLM_ENDPOINT}",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=60),
        ) as resp:
            elapsed = time.monotonic() - t0
            if resp.status != 200:
                return {"error": f"HTTP {resp.status}", "time": elapsed}
            result = await resp.json()
            content = result["choices"][0]["message"]["content"]
            parsed = parse_classification_json(content)
            parsed["time"] = elapsed
            return parsed
    except Exception as e:
        return {"error": str(e), "time": time.monotonic() - t0}


async def run_classify(args):
    """Classify all samples with lite profile."""
    input_dir = Path(args.input_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    samples = scan_samples(input_dir)
    print(f"Found {len(samples)} PDFs in {input_dir}")

    model_name = args.model or DEFAULT_7B_MODEL
    servers = args.servers.split(",") if args.servers else DEFAULT_7B_SERVERS

    print(f"Checking server health...")
    healthy = await check_server_health(servers)
    if not healthy:
        print("ERROR: No healthy servers. Start vLLM first.")
        sys.exit(1)
    print(f"  {len(healthy)}/{len(servers)} servers healthy")

    # Render all pages first (they're single-page PDFs)
    print("Rendering pages...")
    loop = asyncio.get_event_loop()
    rendered = []
    for s in samples:
        try:
            b64 = await loop.run_in_executor(None, render_page_to_base64, s["pdf_path"], 0, args.dpi)
            rendered.append((s, b64))
        except Exception as e:
            print(f"  WARN: Could not render {s['doc_id']}: {e}")
            rendered.append((s, None))

    # Classify with concurrency
    print(f"Classifying {len(rendered)} pages (model={model_name})...")
    sem = asyncio.Semaphore(args.concurrency)
    connector = aiohttp.TCPConnector(limit=args.concurrency + 2)
    results = []

    async with aiohttp.ClientSession(connector=connector) as session:
        async def classify_with_sem(sample, b64, idx):
            if b64 is None:
                return sample, {"error": "render failed"}
            server = healthy[idx % len(healthy)]
            async with sem:
                cls = await classify_one(session, server, b64, model_name)
                return sample, cls

        tasks = [classify_with_sem(s, b64, i) for i, (s, b64) in enumerate(rendered)]
        for coro in asyncio.as_completed(tasks):
            sample, cls = await coro
            complex_flag = is_complex(cls) if "error" not in cls else None
            results.append((sample, cls, complex_flag))
            status = "complex" if complex_flag else ("simple" if complex_flag is not None else "ERROR")
            print(f"  {sample['doc_id']}: {status} ({cls.get('time', 0):.1f}s)")

    # Sort by doc_id for consistent output
    results.sort(key=lambda x: x[0]["doc_id"])

    # Write classification CSV
    csv_path = results_dir / "classification.csv"
    lite_fields = ["multi_column", "has_tables", "poor_quality", "has_non_latin"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["doc_id", *lite_fields, "is_complex", "error"])
        writer.writeheader()
        for sample, cls, is_cx in results:
            row = {
                "doc_id": sample["doc_id"],
                **{field: cls.get(field) for field in lite_fields},
                "is_complex": is_cx,
                "error": cls.get("error"),
            }
            writer.writerow(row)

    n_simple = sum(1 for _, _, c in results if c is False)
    n_complex = sum(1 for _, _, c in results if c is True)
    n_error = sum(1 for _, _, c in results if c is None)
    print(f"\nClassification done: {n_simple} simple, {n_complex} complex, {n_error} errors")
    print(f"Saved to {csv_path}")


# =============================================================================
# OCR
# =============================================================================

async def ocr_one(
    session: aiohttp.ClientSession,
    server_url: str,
    image_b64: str,
    prompt: str,
    model_name: str,
    max_tokens: int,
) -> tuple[str, float]:
    """OCR a single image, return (text, time)."""
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }

    t0 = time.monotonic()
    try:
        async with session.post(
            f"{server_url}{VLLM_ENDPOINT}",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=180),
        ) as resp:
            elapsed = time.monotonic() - t0
            if resp.status != 200:
                error_text = await resp.text()
                return f"ERROR: HTTP {resp.status}: {error_text}", elapsed
            result = await resp.json()
            content = result["choices"][0]["message"]["content"]
            return content, elapsed
    except Exception as e:
        return f"ERROR: {e}", time.monotonic() - t0


def load_classification(results_dir: Path) -> dict:
    """Load classification.csv into dict keyed by doc_id."""
    csv_path = results_dir / "classification.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. Run --phase classify first.")
        sys.exit(1)

    rows = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            doc_id = row["doc_id"]
            # Parse booleans
            for field in ["multi_column", "has_tables", "poor_quality", "has_non_latin", "is_complex"]:
                val = row.get(field, "")
                if val in ("True", "true", "1"):
                    row[field] = True
                elif val in ("False", "false", "0"):
                    row[field] = False
                else:
                    row[field] = None
            rows[doc_id] = row
    return rows


def build_simple_prompt(version: str = "v1") -> str:
    return build_simple_prompt_versioned(version)


def build_complex_prompt_from_cls(cls_row: dict, version: str = "v1") -> str:
    """Build prompt from classification row (lite fields mapped to full)."""
    cls_dict = {
        "multi_column": cls_row.get("multi_column", False),
        "has_tables": cls_row.get("has_tables", False),
        "handwritten": False,  # not in lite
        "has_stamps": False,
        "poor_quality": cls_row.get("poor_quality", False),
        "has_strikethrough": False,
        "latin_script": not cls_row.get("has_non_latin", False),
        "has_footnotes": False,
        "has_forms": False,
    }
    return build_prompt_versioned(cls_dict, version)


async def run_ocr(args, tier: str):
    """Run OCR for simple or complex tier."""
    input_dir = Path(args.input_dir)
    results_dir = Path(args.results_dir)
    version = args.prompt_version

    classification = load_classification(results_dir)
    samples = scan_samples(input_dir)

    # Filter by tier
    if tier == "simple":
        pages = [s for s in samples if classification.get(s["doc_id"], {}).get("is_complex") is False]
        default_model = DEFAULT_7B_MODEL
        default_servers = DEFAULT_7B_SERVERS
    else:
        pages = [s for s in samples if classification.get(s["doc_id"], {}).get("is_complex") is True]
        default_model = DEFAULT_32B_MODEL
        default_servers = DEFAULT_32B_SERVERS

    if not pages:
        print(f"No {tier} pages to process.")
        return

    model_name = args.model or default_model
    servers = args.servers.split(",") if args.servers else default_servers

    print(f"OCR {tier} tier: {len(pages)} pages, model={model_name}, prompt={version}")

    healthy = await check_server_health(servers)
    if not healthy:
        print("ERROR: No healthy servers.")
        sys.exit(1)
    print(f"  {len(healthy)}/{len(servers)} servers healthy")

    # Output dir — versioned
    out_dir = results_dir / f"ocr_{tier}_{version}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Skip already-done pages
    already_done = {p.stem for p in out_dir.glob("*.txt")}
    todo = [s for s in pages if s["doc_id"] not in already_done]
    if already_done:
        print(f"  Skipping {len(already_done)} already done, {len(todo)} remaining")
    if not todo:
        print("All pages already done.")
        return

    # Render
    print("Rendering pages...")
    loop = asyncio.get_event_loop()
    rendered = []
    for s in todo:
        try:
            b64 = await loop.run_in_executor(None, render_page_to_base64, s["pdf_path"], 0, args.dpi)
            rendered.append((s, b64))
        except Exception as e:
            print(f"  WARN: render failed {s['doc_id']}: {e}")

    # Build prompts
    simple_prompt = build_simple_prompt(version)

    # OCR with concurrency
    sem = asyncio.Semaphore(args.concurrency)
    connector = aiohttp.TCPConnector(limit=args.concurrency + 2)

    async with aiohttp.ClientSession(connector=connector) as session:
        async def ocr_with_sem(sample, b64, idx):
            server = healthy[idx % len(healthy)]
            if tier == "simple":
                prompt = simple_prompt
            else:
                cls_row = classification[sample["doc_id"]]
                prompt = build_complex_prompt_from_cls(cls_row, version)
            async with sem:
                text, elapsed = await ocr_one(session, server, b64, prompt, model_name, args.max_tokens)
                return sample, text, elapsed

        tasks = [ocr_with_sem(s, b64, i) for i, (s, b64) in enumerate(rendered)]
        done_count = 0
        for coro in asyncio.as_completed(tasks):
            sample, text, elapsed = await coro
            done_count += 1

            # Save OCR text
            out_path = out_dir / f"{sample['doc_id']}.txt"
            out_path.write_text(text, encoding="utf-8")

            is_err = text.startswith("ERROR:")
            status = "ERROR" if is_err else f"{len(text)} chars"
            print(f"  [{done_count}/{len(rendered)}] {sample['doc_id']}: {status} ({elapsed:.1f}s)")

    print(f"\nOCR {tier} done. Output in {out_dir}")


# =============================================================================
# ALL-7B: classify + OCR simple in one go
# =============================================================================

async def run_all_7b(args):
    """Classify + OCR simple pages, all using 7B servers."""
    await run_classify(args)
    await run_ocr(args, "simple")


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_args()

    if args.list_versions:
        print("Available prompt versions:")
        list_prompt_versions()
        return

    if args.phase == "classify":
        asyncio.run(run_classify(args))
    elif args.phase == "ocr-simple":
        asyncio.run(run_ocr(args, "simple"))
    elif args.phase == "ocr-complex":
        asyncio.run(run_ocr(args, "complex"))
    elif args.phase == "all-7b":
        asyncio.run(run_all_7b(args))


if __name__ == "__main__":
    main()
