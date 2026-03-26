#!/usr/bin/env python3
"""
Daily OCR Deployment Pipeline

Orchestrates the full daily batch on a p4d.24xlarge instance:
  1. Download PDFs from S3 input bucket
  2. Build page manifest
  3. Classify all pages (7B AWQ, 8 servers TP=1)
  4. OCR complex pages only (32B AWQ, 4 servers TP=2)
  5. Replace PDF text layers on complex pages
  6. Upload modified PDFs to S3 output bucket
  7. Move processed inputs to S3 processed/ prefix
  8. Self-shutdown

Usage:
    python deploy_pipeline.py \\
        --s3-input s3://bucket/input/ \\
        --s3-output s3://bucket/output/ \\
        --shutdown

    # Local test (no S3, no shutdown):
    python deploy_pipeline.py \\
        --local-input /path/to/pdfs \\
        --no-shutdown
"""

import argparse
import asyncio
import csv
import json
import logging
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import aiohttp

# ---------------------------------------------------------------------------
# Path setup — allow imports from parent (poc/) and sibling (benchmark/)
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
POC_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(POC_DIR))
sys.path.insert(0, str(POC_DIR / "benchmark"))

from poc_utils import (
    render_page_to_base64,
    classify_image_async,
    ocr_image_async,
    flatten_classification,
    is_complex,
    check_server_health,
    load_checkpoint,
    save_checkpoint_entry,
    make_page_id,
    ProgressTracker,
    get_classification_fields,
    CLASSIFICATION_FIELDS,
)
from prompts import get_classify_prompt, build_prompt
from benchmark_config import BenchmarkConfig, MODEL_7B, MODEL_32B
from benchmark_server import start_servers, wait_for_healthy, stop_servers

from deploy_config import DeployConfig
from pdf_text_replace import replace_text_layers_batch
from s3_sync import (
    upload_directory,
    move_processed_inputs,
    upload_completion_marker,
    upload_log_file,
)
from instance_lifecycle import shutdown_instance

log = logging.getLogger("deploy")


# =============================================================================
# CSV WRITER (reused from poc_2/poc_3 pattern)
# =============================================================================

class CSVResultWriter:
    """Thread-safe incremental CSV writer."""

    def __init__(self, output_path: Path, fieldnames: list[str]):
        self.output_path = output_path
        self._lock = asyncio.Lock()
        self._file = open(output_path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=fieldnames)
        self._writer.writeheader()

    async def write_row(self, row: dict):
        async with self._lock:
            self._writer.writerow(row)
            self._file.flush()

    def close(self):
        self._file.close()


# =============================================================================
# PHASE 1: DOWNLOAD & MANIFEST
# =============================================================================

def phase_download_and_manifest(config: DeployConfig, local_input: str | None = None):
    """Download PDFs from S3 (or use local dir) and build page manifest."""
    from poc_1_manifest import download_s3_pdfs, build_manifest

    if local_input:
        # Local mode — scan directory for PDFs
        input_dir = Path(local_input)
        pdfs = []
        for pdf_path in sorted(input_dir.glob("*.pdf")):
            pdfs.append({
                "pdf_path": pdf_path,
                "pdf_filename": pdf_path.name,
                "folder": "local",
            })
        log.info("Found %d PDFs in local dir %s", len(pdfs), input_dir)
    else:
        pdfs = download_s3_pdfs(config.s3_input_uri)

    manifest = build_manifest(pdfs)
    log.info("Manifest: %d pages across %d documents", len(manifest), len(pdfs))
    return pdfs, manifest


# =============================================================================
# PHASE 2: CLASSIFICATION (producer-consumer)
# =============================================================================

CLASSIFY_FIELDNAMES = None  # set dynamically based on profile


async def _classify_producer(pages, queue, dpi):
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


async def _classify_consumer(
    queue, server_url, session, results_list, progress,
    done_event, model_name, classify_prompt, classification_fields,
):
    """Pull rendered pages from queue, classify via vLLM, collect results."""
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

        if item["error"]:
            results_list.append({
                "pdf_filename": page_info["pdf_filename"],
                "page_num": page_info["page_num"],
                "is_complex": False,
                "error": f"render: {item['error']}",
            })
            await progress.update(error=True)
            queue.task_done()
            continue

        classification, classify_time = await classify_image_async(
            session, server_url, item["image_b64"],
            model_name=model_name, prompt=classify_prompt,
        )

        flat = flatten_classification(classification, fields=classification_fields)
        flat["pdf_filename"] = page_info["pdf_filename"]
        flat["page_num"] = page_info["page_num"]
        flat["local_path"] = page_info["local_path"]
        results_list.append(flat)
        await progress.update(error=flat["error"] is not None)
        queue.task_done()


async def phase_classify(config: DeployConfig, manifest, servers):
    """Run classification on all pages. Returns list of result dicts."""
    profile = config.classify_profile
    classify_prompt = get_classify_prompt(profile)
    classification_fields = get_classification_fields(profile)
    model_name = config.classify_server_config().model

    results = []
    queue = asyncio.Queue(maxsize=64)
    done_event = asyncio.Event()
    progress = ProgressTracker(total=len(manifest), label="Classify")

    async with aiohttp.ClientSession() as session:
        # Start consumers — one per server
        consumer_tasks = []
        for server_url in servers:
            for _ in range(config.classify_concurrency):
                task = asyncio.create_task(_classify_consumer(
                    queue, server_url, session, results, progress,
                    done_event, model_name, classify_prompt, classification_fields,
                ))
                consumer_tasks.append(task)

        # Run producer
        await _classify_producer(manifest, queue, config.dpi)
        await queue.join()
        done_event.set()
        await asyncio.gather(*consumer_tasks)

    summary = progress.summary()
    log.info(
        "Classification done: %d pages, %.1f pages/sec, %d errors",
        summary["total_completed"], summary.get("pages_per_sec", 0), summary["total_errors"],
    )
    return results


# =============================================================================
# PHASE 3: OCR COMPLEX PAGES (producer-consumer)
# =============================================================================

async def _ocr_producer(pages, queue, dpi):
    """Render complex PDF pages to base64 and push to queue."""
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


async def _ocr_consumer(
    queue, server_url, session, ocr_text_dir, progress,
    done_event, model_name, max_tokens,
):
    """Pull rendered pages, OCR via vLLM, save text files."""
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

        if item["error"]:
            await progress.update(error=True)
            queue.task_done()
            continue

        # Build per-page complex prompt from classification flags
        cls_dict = {f: page_info.get(f, False) for f in CLASSIFICATION_FIELDS}
        prompt = build_prompt(cls_dict)

        ocr_text, ocr_time = await ocr_image_async(
            session, server_url, item["image_b64"],
            prompt=prompt,
            model_name=model_name,
            max_tokens=max_tokens,
        )

        has_error = ocr_text.startswith("ERROR: ")
        if not has_error:
            # Save OCR text to file
            text_file = ocr_text_dir / f"{page_info['pdf_filename']}__page{page_info['page_num']}.txt"
            text_file.write_text(ocr_text, encoding="utf-8")

        await progress.update(error=has_error)
        queue.task_done()


async def phase_ocr_complex(config: DeployConfig, complex_pages, servers):
    """Run OCR on complex pages. Saves text files to ocr_text_dir."""
    ocr_cfg = config.ocr_server_config()
    model_name = ocr_cfg.model
    max_tokens = config.ocr_max_tokens

    queue = asyncio.Queue(maxsize=32)
    done_event = asyncio.Event()
    progress = ProgressTracker(total=len(complex_pages), label="OCR Complex")

    async with aiohttp.ClientSession() as session:
        consumer_tasks = []
        for server_url in servers:
            for _ in range(config.ocr_concurrency):
                task = asyncio.create_task(_ocr_consumer(
                    queue, server_url, session, config.ocr_text_dir, progress,
                    done_event, model_name, max_tokens,
                ))
                consumer_tasks.append(task)

        await _ocr_producer(complex_pages, queue, config.dpi)
        await queue.join()
        done_event.set()
        await asyncio.gather(*consumer_tasks)

    summary = progress.summary()
    log.info(
        "OCR done: %d pages, %.1f pages/sec, %d errors",
        summary["total_completed"], summary.get("pages_per_sec", 0), summary["total_errors"],
    )
    return summary


# =============================================================================
# MAIN PIPELINE
# =============================================================================

async def run_pipeline(config: DeployConfig, local_input: str | None = None):
    """Execute the full deployment pipeline."""
    pipeline_start = time.monotonic()
    metrics = {
        "date_partition": config.date_partition,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "unknown",
    }

    config.ensure_dirs()

    try:
        # ---- Phase 1: Download & manifest ----
        log.info("=== Phase 1: Download & Manifest ===")
        t0 = time.monotonic()
        pdfs, manifest = phase_download_and_manifest(config, local_input)
        metrics["phase1_download_sec"] = round(time.monotonic() - t0, 1)
        metrics["total_pages"] = len(manifest)
        metrics["total_documents"] = len(pdfs)

        if not manifest:
            log.info("No pages to process. Exiting.")
            metrics["status"] = "empty"
            return metrics

        # ---- Phase 2: Classification (7B) ----
        log.info("=== Phase 2: Classification (7B AWQ) ===")
        classify_cfg = config.classify_server_config()
        t0 = time.monotonic()

        log.info("Starting 7B servers (%d x TP=%d)...", classify_cfg.num_servers, classify_cfg.tp)
        procs_7b = await start_servers(classify_cfg, log_dir=config.log_dir)
        servers_7b = await wait_for_healthy(classify_cfg, timeout=300)
        log.info("%d servers healthy", len(servers_7b))

        classification = await phase_classify(config, manifest, servers_7b)
        stop_servers(procs_7b)
        await asyncio.sleep(10)  # let GPU memory release
        metrics["phase2_classify_sec"] = round(time.monotonic() - t0, 1)

        # ---- Phase 3: Route ----
        complex_pages = [p for p in classification if p.get("is_complex")]
        simple_count = len(classification) - len(complex_pages)
        metrics["pages_simple"] = simple_count
        metrics["pages_complex"] = len(complex_pages)
        log.info("Routing: %d simple, %d complex", simple_count, len(complex_pages))

        # Group by document to know which docs need processing
        doc_has_complex = set()
        for p in complex_pages:
            doc_has_complex.add(p["pdf_filename"])
        metrics["docs_with_complex"] = len(doc_has_complex)
        metrics["docs_skipped"] = len(pdfs) - len(doc_has_complex)

        # ---- Phase 4: OCR complex pages (32B) ----
        if complex_pages:
            log.info("=== Phase 4: OCR Complex (%d pages, 32B AWQ) ===", len(complex_pages))
            ocr_cfg = config.ocr_server_config()
            t0 = time.monotonic()

            log.info("Starting 32B servers (%d x TP=%d)...", ocr_cfg.num_servers, ocr_cfg.tp)
            procs_32b = await start_servers(ocr_cfg, log_dir=config.log_dir)
            servers_32b = await wait_for_healthy(ocr_cfg, timeout=300)
            log.info("%d servers healthy", len(servers_32b))

            ocr_summary = await phase_ocr_complex(config, complex_pages, servers_32b)
            stop_servers(procs_32b)
            metrics["phase4_ocr_sec"] = round(time.monotonic() - t0, 1)
            metrics["ocr_errors"] = ocr_summary.get("total_errors", 0)

            # ---- Phase 5: Replace PDF text layers ----
            log.info("=== Phase 5: Replace PDF Text Layers ===")
            t0 = time.monotonic()

            # Determine input directory (where PDFs were downloaded)
            if local_input:
                input_dir = Path(local_input)
            else:
                from poc_1_manifest import S3_STAGING_DIR
                input_dir = S3_STAGING_DIR

            replace_stats = replace_text_layers_batch(
                input_dir=input_dir,
                output_dir=config.local_output,
                ocr_text_dir=config.ocr_text_dir,
                complex_pages=complex_pages,
            )
            metrics["phase5_replace_sec"] = round(time.monotonic() - t0, 1)
            metrics["pdfs_modified"] = len(replace_stats)
        else:
            log.info("No complex pages found. Skipping OCR and text replacement.")

        # ---- Phase 6: Upload to S3 ----
        if not local_input:
            log.info("=== Phase 6: Upload & Cleanup ===")
            t0 = time.monotonic()

            if doc_has_complex:
                uploaded = upload_directory(config.local_output, config.s3_output_dated)
                metrics["files_uploaded"] = uploaded

            # Move ALL processed inputs (including skipped docs) to processed/
            all_filenames = [p["pdf_filename"] for p in pdfs]
            moved = move_processed_inputs(
                config.s3_input_uri, config.s3_processed_dated, all_filenames
            )
            metrics["files_moved"] = moved
            metrics["phase6_upload_sec"] = round(time.monotonic() - t0, 1)

        metrics["status"] = "success"

    except Exception as e:
        log.exception("Pipeline failed")
        metrics["status"] = "error"
        metrics["error"] = str(e)

    finally:
        pipeline_time = time.monotonic() - pipeline_start
        metrics["total_wall_time_sec"] = round(pipeline_time, 1)
        metrics["cost_estimate_usd"] = round(pipeline_time / 3600 * 32.77, 2)
        metrics["finished_at"] = datetime.now(timezone.utc).isoformat()

        log.info("Pipeline %s in %.0fs (est. $%.2f)",
                 metrics["status"], pipeline_time, metrics["cost_estimate_usd"])

        # Upload metrics and logs
        if not local_input:
            try:
                upload_completion_marker(config.s3_output_dated, metrics)
                # Upload log file if it exists
                log_path = config.log_dir / "pipeline.log"
                if log_path.exists():
                    upload_log_file(log_path, config.s3_logs_dated)
            except Exception as e:
                log.error("Failed to upload completion marker: %s", e)

        # Shutdown
        if config.shutdown_on_complete and not local_input:
            log.info("Shutting down instance...")
            shutdown_instance()

    return metrics


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Daily OCR deployment pipeline")
    parser.add_argument("--s3-input", default="s3://ocr-pipeline-bucket/input/",
                        help="S3 URI for input PDFs")
    parser.add_argument("--s3-output", default="s3://ocr-pipeline-bucket/output/",
                        help="S3 URI for output PDFs")
    parser.add_argument("--s3-processed", default=None,
                        help="S3 URI for processed inputs (default: same bucket /processed/)")
    parser.add_argument("--local-input", default=None,
                        help="Use local directory instead of S3 (for testing)")
    parser.add_argument("--no-shutdown", action="store_true",
                        help="Don't shutdown instance after completion")
    parser.add_argument("--date", default=None,
                        help="Date partition (default: today YYYY-MM-DD)")
    return parser.parse_args()


def setup_logging(log_dir: Path):
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "pipeline.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path),
        ],
    )


def main():
    args = parse_args()

    config = DeployConfig(
        s3_input_uri=args.s3_input,
        s3_output_uri=args.s3_output,
        shutdown_on_complete=not args.no_shutdown,
    )
    if args.s3_processed:
        config.s3_processed_uri = args.s3_processed
    if args.date:
        config.date_partition = args.date

    setup_logging(config.log_dir)
    log.info("OCR Deployment Pipeline starting")
    log.info("Config: input=%s output=%s date=%s shutdown=%s",
             config.s3_input_uri if not args.local_input else args.local_input,
             config.s3_output_uri, config.date_partition, config.shutdown_on_complete)

    metrics = asyncio.run(run_pipeline(config, local_input=args.local_input))

    # Print summary
    print(json.dumps(metrics, indent=2, default=str))
    sys.exit(0 if metrics.get("status") == "success" else 1)


if __name__ == "__main__":
    main()
