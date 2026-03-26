"""
POC Phase 3: Two-Tier OCR Throughput Benchmark

Reads classification.csv to split pages into simple vs complex tiers,
then runs OCR via vLLM with tier-appropriate models and prompts.
Focus is timing/throughput metrics, not accuracy.

Usage:
    # Tier 1: simple pages via 7B AWQ (4 servers)
    python poc_3_ocr.py --run-dir results/awq__BAC-0002-1971 --tier simple

    # Tier 2: complex pages via 72B AWQ (1 server, TP=4)
    python poc_3_ocr.py --run-dir results/awq__BAC-0002-1971 --tier complex

    # Resume interrupted run
    python poc_3_ocr.py --run-dir results/awq__BAC-0002-1971 --tier simple --resume

Output:
    <run_dir>/ocr_results_simple.csv   (or ocr_results_complex.csv)
    <run_dir>/ocr_meta_simple.json     (or ocr_meta_complex.json)
"""

import argparse
import asyncio
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import aiohttp

# Allow importing prompts.py from parent directory
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from prompts import build_prompt, BASE_PROMPT, HEADER_FOOTER_BLOCK, OUTPUT_FORMAT

from poc_utils import (
    VLLM_SERVERS,
    VLLM_ENDPOINT,
    MODEL_CONFIGS,
    RESULTS_DIR,
    CLASSIFICATION_FIELDS,
    render_page_to_base64,
    ocr_image_async,
    check_server_health,
    load_checkpoint,
    save_checkpoint_entry,
    make_page_id,
    ProgressTracker,
)


# =============================================================================
# DEFAULTS PER TIER
# =============================================================================

TIER_DEFAULTS = {
    "simple": {
        "model": "Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
        "servers": [f"http://localhost:{8000 + i}" for i in range(4)],
        "max_tokens": 4096,
    },
    "complex": {
        "model": "Qwen/Qwen2.5-VL-72B-Instruct-AWQ",
        "servers": ["http://localhost:8000"],
        "max_tokens": 8192,
    },
}


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Two-tier OCR throughput benchmark"
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Run directory containing classification.csv + manifest.csv",
    )
    parser.add_argument(
        "--tier",
        choices=["simple", "complex"],
        required=True,
        help="Which tier to process (simple=7B, complex=72B)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name for vLLM (default: tier-appropriate AWQ model)",
    )
    parser.add_argument(
        "--servers",
        type=str,
        default=None,
        help="Comma-separated server URLs (default: tier-appropriate)",
    )
    parser.add_argument(
        "--concurrency-per-server",
        type=int,
        default=4,
        help="Concurrent requests per vLLM server (default: 4)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Max tokens for OCR output (default: 4096 simple, 8192 complex)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint, skipping already-processed pages",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for page rendering (default: 150)",
    )
    parser.add_argument(
        "--queue-size",
        type=int,
        default=64,
        help="Max size of rendered-page queue (default: 64)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="Log progress every N pages (default: 100)",
    )
    parser.add_argument(
        "--instance",
        type=str,
        default=None,
        help="EC2 instance type for metadata (e.g. g5.12xlarge)",
    )
    return parser.parse_args()


# =============================================================================
# DATA LOADING
# =============================================================================

def load_manifest(manifest_path: Path) -> list[dict]:
    """Load manifest CSV into list of dicts."""
    rows = []
    with open(manifest_path, "r", encoding="utf-8-sig") as f:
        # Strip leading blank lines (some CSVs have stray bytes at start)
        content = f.read().lstrip("\r\n\ufeff")
    import io
    reader = csv.DictReader(io.StringIO(content))
    for row in reader:
        if not row.get("page_num"):
            continue
        row["page_num"] = int(row["page_num"])
        row["total_pages"] = int(row["total_pages"])
        row["file_size_bytes"] = int(row["file_size_bytes"])
        rows.append(row)
    return rows


def load_classification(path: Path) -> list[dict]:
    """Load classification CSV, casting types appropriately."""
    rows = []
    timing_fields = ["render_time_sec", "classify_time_sec", "queue_wait_sec", "total_time_sec"]
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for field in timing_fields:
                try:
                    row[field] = float(row[field]) if row[field] else None
                except (ValueError, TypeError):
                    row[field] = None

            for field in CLASSIFICATION_FIELDS:
                val = row.get(field, "")
                if val in ("True", "true", "1"):
                    row[field] = True
                elif val in ("False", "false", "0"):
                    row[field] = False
                else:
                    row[field] = None

            val = row.get("is_complex", "")
            if val in ("True", "true", "1"):
                row["is_complex"] = True
            elif val in ("False", "false", "0"):
                row["is_complex"] = False
            else:
                row["is_complex"] = None

            row["page_num"] = int(row["page_num"]) if row.get("page_num") else None
            row["total_pages"] = int(row["total_pages"]) if row.get("total_pages") else None

            rows.append(row)
    return rows


# =============================================================================
# PROMPT BUILDING
# =============================================================================

def build_simple_prompt() -> str:
    """Build a single static prompt for simple pages."""
    return "\n".join([BASE_PROMPT, HEADER_FOOTER_BLOCK, OUTPUT_FORMAT])


def build_complex_prompt(classification_row: dict) -> str:
    """Build a per-page prompt from classification flags via build_prompt()."""
    cls_dict = {f: classification_row.get(f, False) for f in CLASSIFICATION_FIELDS}
    return build_prompt(cls_dict)


# =============================================================================
# CSV WRITER
# =============================================================================

OCR_FIELDNAMES = [
    "pdf_filename", "page_num", "tier", "prompt_type", "model",
    "ocr_text_length", "ocr_time_sec", "render_time_sec", "total_time_sec",
    "server", "queue_wait_sec", "error",
]


class CSVResultWriter:
    """Async-safe incremental CSV writer."""

    def __init__(self, output_path: Path, resume: bool = False):
        self.output_path = output_path
        self._lock = asyncio.Lock()

        if resume and output_path.exists():
            self._file = open(output_path, "a", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(self._file, fieldnames=OCR_FIELDNAMES)
        else:
            self._file = open(output_path, "w", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(self._file, fieldnames=OCR_FIELDNAMES)
            self._writer.writeheader()

    async def write_row(self, row: dict):
        async with self._lock:
            self._writer.writerow(row)
            self._file.flush()

    def close(self):
        self._file.close()


# =============================================================================
# PRODUCER: Render pages to base64
# =============================================================================

async def producer(
    pages: list[dict],
    queue: asyncio.Queue,
    dpi: int,
):
    """Render PDF pages to base64 and push to queue."""
    loop = asyncio.get_event_loop()

    for page_info in pages:
        pdf_path = Path(page_info["local_path"])
        page_num = page_info["page_num"]

        t0 = time.monotonic()
        try:
            image_b64 = await loop.run_in_executor(
                None, render_page_to_base64, pdf_path, page_num - 1, dpi
            )
            render_time = time.monotonic() - t0
        except Exception as e:
            render_time = time.monotonic() - t0
            await queue.put({
                "page_info": page_info,
                "image_b64": None,
                "render_time": render_time,
                "error": str(e),
            })
            continue

        await queue.put({
            "page_info": page_info,
            "image_b64": image_b64,
            "render_time": render_time,
            "error": None,
        })


# =============================================================================
# CONSUMER: Send to vLLM for OCR and collect results
# =============================================================================

async def consumer(
    queue: asyncio.Queue,
    server_url: str,
    session: aiohttp.ClientSession,
    csv_writer: CSVResultWriter,
    checkpoint_path: Path,
    progress: ProgressTracker,
    queue_max: int,
    done_event: asyncio.Event,
    tier: str,
    model_name: str,
    max_tokens: int,
    simple_prompt: str,
):
    """Pull rendered pages from queue, OCR via vLLM, write results."""
    while True:
        if done_event.is_set() and queue.empty():
            break

        try:
            item = await asyncio.wait_for(queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            if done_event.is_set():
                break
            continue

        page_info = item["page_info"]
        page_id = make_page_id(page_info["pdf_filename"], page_info["page_num"])
        t_total_start = time.monotonic()

        # Handle render errors
        if item["error"]:
            row = {
                "pdf_filename": page_info["pdf_filename"],
                "page_num": page_info["page_num"],
                "tier": tier,
                "prompt_type": tier,
                "model": model_name,
                "ocr_text_length": 0,
                "ocr_time_sec": 0,
                "render_time_sec": round(item["render_time"], 4),
                "total_time_sec": round(item["render_time"], 4),
                "server": server_url,
                "queue_wait_sec": 0,
                "error": f"render: {item['error']}",
            }
            await csv_writer.write_row(row)
            save_checkpoint_entry(checkpoint_path, page_id)
            await progress.update(error=True, queue_depth=queue.qsize(), queue_max=queue_max)
            queue.task_done()
            continue

        queue_wait = time.monotonic() - t_total_start

        # Build prompt
        if tier == "simple":
            prompt = simple_prompt
            prompt_type = "simple"
        else:
            prompt = build_complex_prompt(page_info)
            prompt_type = "complex"

        # OCR via vLLM
        ocr_text, ocr_time = await ocr_image_async(
            session, server_url, item["image_b64"],
            prompt=prompt,
            model_name=model_name,
            max_tokens=max_tokens,
        )

        total_time = item["render_time"] + queue_wait + ocr_time
        has_error = ocr_text.startswith("ERROR: ")

        row = {
            "pdf_filename": page_info["pdf_filename"],
            "page_num": page_info["page_num"],
            "tier": tier,
            "prompt_type": prompt_type,
            "model": model_name,
            "ocr_text_length": len(ocr_text) if not has_error else 0,
            "ocr_time_sec": round(ocr_time, 4),
            "render_time_sec": round(item["render_time"], 4),
            "total_time_sec": round(total_time, 4),
            "server": server_url,
            "queue_wait_sec": round(queue_wait, 4),
            "error": ocr_text if has_error else None,
        }

        await csv_writer.write_row(row)
        save_checkpoint_entry(checkpoint_path, page_id)
        await progress.update(
            error=has_error,
            queue_depth=queue.qsize(),
            queue_max=queue_max,
        )
        queue.task_done()


# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================

async def run_ocr(args):
    """Main async orchestrator: producer-consumer OCR pipeline."""

    tier = args.tier
    defaults = TIER_DEFAULTS[tier]

    # Resolve model
    model_name = args.model or defaults["model"]
    max_tokens = args.max_tokens or defaults["max_tokens"]

    # Resolve servers
    if args.servers:
        servers = [s.strip() for s in args.servers.split(",")]
        # Add http:// prefix if missing
        servers = [s if s.startswith("http") else f"http://{s}" for s in servers]
    else:
        servers = defaults["servers"]

    # Resolve run directory
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = RESULTS_DIR / run_dir.name if (RESULTS_DIR / run_dir.name).exists() else Path(args.run_dir).resolve()

    classification_path = run_dir / "classification.csv"
    manifest_path = run_dir / "manifest.csv"

    if not classification_path.exists():
        print(f"ERROR: Classification file not found: {classification_path}")
        sys.exit(1)
    if not manifest_path.exists():
        print(f"ERROR: Manifest file not found: {manifest_path}")
        sys.exit(1)

    # Load data
    print(f"Loading classification: {classification_path}")
    classification = load_classification(classification_path)
    print(f"  Total classified pages: {len(classification)}")

    print(f"Loading manifest: {manifest_path}")
    manifest = load_manifest(manifest_path)
    print(f"  Total manifest pages: {len(manifest)}")

    # Build path lookup from manifest
    path_lookup = {}
    for row in manifest:
        key = (row["pdf_filename"], row["page_num"])
        path_lookup[key] = row["local_path"]

    # Filter by tier
    if tier == "simple":
        pages = [r for r in classification if r.get("is_complex") is False]
    else:
        pages = [r for r in classification if r.get("is_complex") is True]

    print(f"  {tier.capitalize()} tier pages: {len(pages)}")

    # Enrich pages with local_path from manifest
    enriched_pages = []
    missing = 0
    for p in pages:
        key = (p["pdf_filename"], p["page_num"])
        local_path = path_lookup.get(key)
        if local_path is None:
            missing += 1
            continue
        p["local_path"] = local_path
        enriched_pages.append(p)

    if missing:
        print(f"  WARNING: {missing} pages not found in manifest (skipped)")

    pages = enriched_pages

    if not pages:
        print("No pages to process for this tier.")
        return

    # Output paths
    output_path = run_dir / f"ocr_results_{tier}.csv"
    checkpoint_file = run_dir / f".ocr_checkpoint_{tier}"
    meta_path = run_dir / f"ocr_meta_{tier}.json"

    # Filter already-completed pages if resuming
    if args.resume:
        completed = load_checkpoint(checkpoint_file)
        original_count = len(pages)
        pages = [
            p for p in pages
            if make_page_id(p["pdf_filename"], p["page_num"]) not in completed
        ]
        print(f"  Resuming: {original_count - len(pages)} already done, {len(pages)} remaining")
    else:
        if checkpoint_file.exists():
            checkpoint_file.unlink()

    if not pages:
        print("All pages already processed. Nothing to do.")
        return

    # Health check
    print(f"\nChecking vLLM server health...")
    healthy_servers = await check_server_health(servers)
    if not healthy_servers:
        print("ERROR: No healthy vLLM servers found. Start servers first.")
        sys.exit(1)
    print(f"  {len(healthy_servers)}/{len(servers)} servers healthy: {healthy_servers}")

    # Configuration
    concurrency_per_server = args.concurrency_per_server
    total_consumers = len(healthy_servers) * concurrency_per_server
    queue_size = args.queue_size

    # Build simple prompt once (only used for simple tier)
    simple_prompt = build_simple_prompt()

    print(f"\nConfiguration:")
    print(f"  Tier: {tier}")
    print(f"  Model: {model_name}")
    print(f"  Max tokens: {max_tokens}")
    print(f"  Servers: {len(healthy_servers)}")
    print(f"  Concurrency/server: {concurrency_per_server}")
    print(f"  Total concurrent requests: {total_consumers}")
    print(f"  Queue size: {queue_size}")
    print(f"  DPI: {args.dpi}")
    print(f"  Pages to OCR: {len(pages)}")
    print(f"  Output: {output_path}")
    print()

    # Setup
    queue = asyncio.Queue(maxsize=queue_size)
    progress = ProgressTracker(total=len(pages), log_interval=args.log_interval)
    csv_writer = CSVResultWriter(output_path, resume=args.resume)
    done_event = asyncio.Event()

    connector = aiohttp.TCPConnector(limit=total_consumers + 4)
    timeout = aiohttp.ClientTimeout(total=180)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        # Start consumers — round-robin assign servers
        consumer_tasks = []
        for i in range(total_consumers):
            server_url = healthy_servers[i % len(healthy_servers)]
            task = asyncio.create_task(
                consumer(
                    queue=queue,
                    server_url=server_url,
                    session=session,
                    csv_writer=csv_writer,
                    checkpoint_path=checkpoint_file,
                    progress=progress,
                    queue_max=queue_size,
                    done_event=done_event,
                    tier=tier,
                    model_name=model_name,
                    max_tokens=max_tokens,
                    simple_prompt=simple_prompt,
                )
            )
            consumer_tasks.append(task)

        # Run producer
        await producer(pages, queue, args.dpi)

        # Wait for queue to drain
        await queue.join()

        # Signal consumers to stop
        done_event.set()

        # Wait for all consumers to finish
        await asyncio.gather(*consumer_tasks)

    csv_writer.close()

    # Final summary
    summary = progress.summary()
    print(f"\nOCR complete! (tier={tier})")
    print(f"  Pages processed: {summary['total_completed']}")
    print(f"  Errors: {summary['total_errors']}")
    print(f"  Wall time: {summary['wall_time_sec']:.1f}s ({summary['wall_time_sec']/60:.1f}min)")
    print(f"  Throughput: {summary['pages_per_sec']:.1f} pages/sec")
    print(f"  Output: {output_path}")

    # Save metadata
    meta = {
        "tier": tier,
        "model": model_name,
        "instance": args.instance,
        "pages_processed": summary["total_completed"],
        "total_errors": summary["total_errors"],
        "wall_time_sec": round(summary["wall_time_sec"], 2),
        "pages_per_sec": round(summary["pages_per_sec"], 4),
        "num_servers": len(healthy_servers),
        "concurrency_per_server": concurrency_per_server,
        "total_concurrency": total_consumers,
        "max_tokens": max_tokens,
        "dpi": args.dpi,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata: {meta_path}")


def main():
    args = parse_args()
    asyncio.run(run_ocr(args))


if __name__ == "__main__":
    main()
