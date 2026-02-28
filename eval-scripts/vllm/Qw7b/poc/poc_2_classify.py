"""
POC Phase 2: Classify Pages with Timing

Reads manifest.csv, renders each page, sends to vLLM for classification,
and records per-page timing data. Uses producer-consumer pattern with
async concurrency for GPU throughput optimization.

Usage:
    python poc_2_classify.py --manifest results/manifest.csv
    python poc_2_classify.py --manifest results/manifest.csv --concurrency-per-server 8 --resume

Output: results/classification.csv
"""

import argparse
import asyncio
import csv
import json
import time
import sys
from datetime import datetime, timezone
from pathlib import Path

import aiohttp

from poc_utils import (
    VLLM_SERVERS,
    MODEL_NAME,
    render_page_to_base64,
    classify_image_async,
    flatten_classification,
    check_server_health,
    load_checkpoint,
    save_checkpoint_entry,
    make_page_id,
    ProgressTracker,
)

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
CHECKPOINT_FILE = RESULTS_DIR / ".classify_checkpoint"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Classify pages from manifest with per-page timing"
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=str(RESULTS_DIR / "manifest.csv"),
        help="Path to manifest CSV (default: results/manifest.csv)",
    )
    parser.add_argument(
        "--concurrency-per-server",
        type=int,
        default=4,
        help="Concurrent requests per vLLM server (default: 4)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint, skipping already-classified pages",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path (default: results/classification.csv)",
    )
    parser.add_argument(
        "--queue-size",
        type=int,
        default=64,
        help="Max size of rendered-page queue (default: 64)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for page rendering (default: 150)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="Log progress every N pages (default: 100)",
    )
    return parser.parse_args()


# =============================================================================
# MANIFEST LOADING
# =============================================================================

def load_manifest(manifest_path: str) -> list[dict]:
    """Load manifest CSV into list of dicts."""
    rows = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["page_num"] = int(row["page_num"])
            row["total_pages"] = int(row["total_pages"])
            row["file_size_bytes"] = int(row["file_size_bytes"])
            rows.append(row)
    return rows


# =============================================================================
# CSV WRITER
# =============================================================================

OUTPUT_FIELDNAMES = [
    "pdf_filename", "page_num", "total_pages",
    "multi_column", "has_tables", "handwritten", "has_stamps",
    "poor_quality", "has_strikethrough", "latin_script",
    "has_footnotes", "has_forms", "is_complex",
    "classify_time_sec", "render_time_sec", "total_time_sec",
    "server", "queue_wait_sec", "error", "raw_response",
]


class CSVResultWriter:
    """Thread-safe incremental CSV writer."""

    def __init__(self, output_path: Path, resume: bool = False):
        self.output_path = output_path
        self._lock = asyncio.Lock()

        if resume and output_path.exists():
            # Append mode — header already written
            self._file = open(output_path, "a", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(self._file, fieldnames=OUTPUT_FIELDNAMES)
        else:
            self._file = open(output_path, "w", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(self._file, fieldnames=OUTPUT_FIELDNAMES)
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
    """
    Render PDF pages to base64 and push to queue.
    Runs page rendering in a thread pool to avoid blocking the event loop.
    """
    loop = asyncio.get_event_loop()

    for page_info in pages:
        pdf_path = Path(page_info["local_path"])
        page_num = page_info["page_num"]

        # Render in thread pool (CPU-bound)
        t0 = time.monotonic()
        try:
            image_b64 = await loop.run_in_executor(
                None, render_page_to_base64, pdf_path, page_num - 1, dpi
            )
            render_time = time.monotonic() - t0
        except Exception as e:
            render_time = time.monotonic() - t0
            # Push error entry
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

    # Signal completion — one sentinel per consumer
    # (handled by caller after all producers finish)


# =============================================================================
# CONSUMER: Send to vLLM and collect results
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
):
    """
    Pull rendered pages from queue, classify via vLLM, write results.
    """
    while True:
        # Check if we should stop
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
                "total_pages": page_info["total_pages"],
                **{f: None for f in [
                    "multi_column", "has_tables", "handwritten", "has_stamps",
                    "poor_quality", "has_strikethrough", "latin_script",
                    "has_footnotes", "has_forms", "is_complex",
                ]},
                "classify_time_sec": 0,
                "render_time_sec": item["render_time"],
                "total_time_sec": item["render_time"],
                "server": server_url,
                "queue_wait_sec": 0,
                "error": f"render: {item['error']}",
                "raw_response": "",
            }
            await csv_writer.write_row(row)
            save_checkpoint_entry(checkpoint_path, page_id)
            await progress.update(error=True, queue_depth=queue.qsize(), queue_max=queue_max)
            queue.task_done()
            continue

        # Queue wait time = total time minus render time so far
        queue_wait = time.monotonic() - t_total_start

        # Classify via vLLM
        classification, classify_time = await classify_image_async(
            session, server_url, item["image_b64"]
        )

        total_time = item["render_time"] + queue_wait + classify_time
        flat = flatten_classification(classification)
        has_error = flat["error"] is not None

        row = {
            "pdf_filename": page_info["pdf_filename"],
            "page_num": page_info["page_num"],
            "total_pages": page_info["total_pages"],
            **{k: flat[k] for k in [
                "multi_column", "has_tables", "handwritten", "has_stamps",
                "poor_quality", "has_strikethrough", "latin_script",
                "has_footnotes", "has_forms", "is_complex",
            ]},
            "classify_time_sec": round(classify_time, 4),
            "render_time_sec": round(item["render_time"], 4),
            "total_time_sec": round(total_time, 4),
            "server": server_url,
            "queue_wait_sec": round(queue_wait, 4),
            "error": flat["error"],
            "raw_response": flat["raw_response"],
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

async def run_classification(args):
    """Main async orchestrator: producer-consumer pipeline."""

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"ERROR: Manifest not found: {manifest_path}")
        sys.exit(1)

    output_path = Path(args.output) if args.output else RESULTS_DIR / "classification.csv"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load manifest
    print(f"Loading manifest: {manifest_path}")
    manifest = load_manifest(str(manifest_path))
    print(f"  Total pages in manifest: {len(manifest)}")

    # Filter already-completed pages if resuming
    if args.resume:
        completed = load_checkpoint(CHECKPOINT_FILE)
        original_count = len(manifest)
        manifest = [
            p for p in manifest
            if make_page_id(p["pdf_filename"], p["page_num"]) not in completed
        ]
        print(f"  Resuming: {original_count - len(manifest)} already done, {len(manifest)} remaining")
    else:
        # Clear checkpoint for fresh run
        if CHECKPOINT_FILE.exists():
            CHECKPOINT_FILE.unlink()

    if not manifest:
        print("All pages already classified. Nothing to do.")
        return

    # Health check
    print("\nChecking vLLM server health...")
    healthy_servers = await check_server_health(VLLM_SERVERS)
    if not healthy_servers:
        print("ERROR: No healthy vLLM servers found. Start servers first.")
        sys.exit(1)
    print(f"  {len(healthy_servers)}/{len(VLLM_SERVERS)} servers healthy: {healthy_servers}")

    # Configuration
    concurrency_per_server = args.concurrency_per_server
    total_consumers = len(healthy_servers) * concurrency_per_server
    queue_size = args.queue_size

    print(f"\nConfiguration:")
    print(f"  Servers: {len(healthy_servers)}")
    print(f"  Concurrency/server: {concurrency_per_server}")
    print(f"  Total concurrent requests: {total_consumers}")
    print(f"  Queue size: {queue_size}")
    print(f"  DPI: {args.dpi}")
    print(f"  Pages to classify: {len(manifest)}")
    print(f"  Output: {output_path}")
    print()

    # Setup
    queue = asyncio.Queue(maxsize=queue_size)
    progress = ProgressTracker(total=len(manifest), log_interval=args.log_interval)
    csv_writer = CSVResultWriter(output_path, resume=args.resume)
    done_event = asyncio.Event()

    connector = aiohttp.TCPConnector(limit=total_consumers + 4)
    timeout = aiohttp.ClientTimeout(total=120)

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
                    checkpoint_path=CHECKPOINT_FILE,
                    progress=progress,
                    queue_max=queue_size,
                    done_event=done_event,
                )
            )
            consumer_tasks.append(task)

        # Run producer
        await producer(manifest, queue, args.dpi)

        # Wait for queue to drain
        await queue.join()

        # Signal consumers to stop
        done_event.set()

        # Wait for all consumers to finish
        await asyncio.gather(*consumer_tasks)

    csv_writer.close()

    # Final summary
    summary = progress.summary()
    print(f"\nClassification complete!")
    print(f"  Pages classified: {summary['total_completed']}")
    print(f"  Errors: {summary['total_errors']}")
    print(f"  Wall time: {summary['wall_time_sec']:.1f}s ({summary['wall_time_sec']/60:.1f}min)")
    print(f"  Throughput: {summary['pages_per_sec']:.1f} pages/sec")
    print(f"  Output: {output_path}")

    # Save metadata for the report
    meta_path = output_path.with_suffix(".meta.json")
    meta = {
        "wall_time_sec": round(summary["wall_time_sec"], 2),
        "pages_classified": summary["total_completed"],
        "total_errors": summary["total_errors"],
        "pages_per_sec": round(summary["pages_per_sec"], 4),
        "num_servers": len(healthy_servers),
        "concurrency_per_server": concurrency_per_server,
        "total_concurrency": total_consumers,
        "dpi": args.dpi,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata: {meta_path}")


def main():
    args = parse_args()
    asyncio.run(run_classification(args))


if __name__ == "__main__":
    main()
