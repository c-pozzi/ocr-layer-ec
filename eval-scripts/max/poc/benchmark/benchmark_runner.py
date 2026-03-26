"""
MAX Serve Benchmark Runner — Main Orchestrator

Loops over config matrix, starts servers, runs workloads, collects metrics.
Reuses poc_utils and prompts from the vLLM POC (same OpenAI-compatible API).
"""

import argparse
import asyncio
import csv
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import aiohttp

# Add paths for imports
SCRIPT_DIR = Path(__file__).resolve().parent
VLLM_POC_DIR = SCRIPT_DIR.parent.parent.parent / "vllm" / "poc"
sys.path.insert(0, str(VLLM_POC_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from poc_utils import (
    render_page_to_base64,
    ocr_image_async,
    classify_image_async,
    get_page_count,
)
from prompts import (
    get_classify_prompt,
    build_simple_prompt_versioned,
    build_prompt_versioned,
)
from benchmark_config import (
    MAXBenchmarkConfig,
    get_7b_configs,
    get_32b_configs,
    get_all_configs,
    RESULTS_DIR,
    P4D_COST_PER_HOUR,
)
from benchmark_server import (
    start_servers,
    wait_for_healthy,
    stop_servers,
    snapshot_gpu_memory,
)

# Input data
INPUT_DIR = Path("/home/ubuntu/ocr-input")
SAMPLE_SIZE = 9000
DPI = 150


# =============================================================================
# PAGE SAMPLING
# =============================================================================

def discover_pages(input_dir: Path, max_pages: int = SAMPLE_SIZE) -> list[dict]:
    """Scan input_dir for PDFs and build a flat list of (pdf_path, page_num) entries."""
    print(f"Discovering pages in {input_dir}...")
    pdfs = sorted(input_dir.rglob("*.pdf"))
    print(f"  Found {len(pdfs)} PDFs")

    all_pages = []
    for pdf_path in pdfs:
        try:
            n_pages = get_page_count(pdf_path)
            for p in range(n_pages):
                all_pages.append({
                    "pdf_path": str(pdf_path),
                    "page_num": p,
                    "pdf_name": pdf_path.stem,
                })
        except Exception as e:
            print(f"  WARNING: Could not read {pdf_path.name}: {e}")

    print(f"  Total pages: {len(all_pages)}")

    if len(all_pages) > max_pages:
        random.seed(42)
        all_pages = random.sample(all_pages, max_pages)
        print(f"  Sampled {max_pages} pages")

    return all_pages


def pre_render_pages(pages: list[dict], dpi: int = DPI) -> list[dict]:
    """Pre-render all pages to base64."""
    print(f"Pre-rendering {len(pages)} pages at {dpi} DPI...")
    rendered = []
    errors = 0

    for i, page in enumerate(pages):
        if (i + 1) % 500 == 0:
            print(f"  Rendered {i + 1}/{len(pages)}...")
        try:
            b64 = render_page_to_base64(Path(page["pdf_path"]), page["page_num"], dpi)
            page["image_b64"] = b64
            rendered.append(page)
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  WARNING: render failed for {page['pdf_name']} p{page['page_num']}: {e}")

    print(f"  Successfully rendered: {len(rendered)} ({errors} errors)")
    return rendered


# =============================================================================
# WORKLOAD RUNNERS
# =============================================================================

async def run_classify_workload(
    pages: list[dict],
    servers: list[str],
    config: MAXBenchmarkConfig,
) -> list[dict]:
    """Run classifier workload and collect per-request metrics."""
    prompt = get_classify_prompt("lite")
    results = []
    sem = asyncio.Semaphore(len(servers) * config.concurrency_per_server)
    server_cycle = servers * (len(pages) // len(servers) + 1)

    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(limit=len(servers) * config.concurrency_per_server + 4),
        timeout=aiohttp.ClientTimeout(total=120),
    ) as session:

        async def classify_one(page: dict, server_url: str):
            async with sem:
                t0 = time.monotonic()
                result, inference_time = await classify_image_async(
                    session, server_url, page["image_b64"],
                    model_name=config.model, prompt=prompt,
                )
                total_time = time.monotonic() - t0
                has_error = "error" in result
                return {
                    "pdf_name": page["pdf_name"],
                    "page_num": page["page_num"],
                    "server": server_url,
                    "inference_time_sec": round(inference_time, 4),
                    "total_time_sec": round(total_time, 4),
                    "error": result.get("error") if has_error else None,
                }

        tasks = [
            classify_one(page, server_cycle[i])
            for i, page in enumerate(pages)
        ]
        results = await asyncio.gather(*tasks)

    return list(results)


async def run_ocr_workload(
    pages: list[dict],
    servers: list[str],
    config: MAXBenchmarkConfig,
    workload_type: str,
) -> list[dict]:
    """Run OCR workload (simple or complex) and collect per-request metrics."""
    if workload_type == "simple_ocr":
        prompt = build_simple_prompt_versioned("v1")
        max_tokens = 4096
    else:
        prompt = build_prompt_versioned(
            {"has_tables": True, "poor_quality": True}, version="v1"
        )
        max_tokens = 8192

    results = []
    sem = asyncio.Semaphore(len(servers) * config.concurrency_per_server)
    server_cycle = servers * (len(pages) // len(servers) + 1)

    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(limit=len(servers) * config.concurrency_per_server + 4),
        timeout=aiohttp.ClientTimeout(total=300),
    ) as session:

        async def ocr_one(page: dict, server_url: str):
            async with sem:
                t0 = time.monotonic()
                text, inference_time = await ocr_image_async(
                    session, server_url, page["image_b64"],
                    prompt=prompt, model_name=config.model, max_tokens=max_tokens,
                )
                total_time = time.monotonic() - t0
                has_error = text.startswith("ERROR: ")
                return {
                    "pdf_name": page["pdf_name"],
                    "page_num": page["page_num"],
                    "server": server_url,
                    "inference_time_sec": round(inference_time, 4),
                    "total_time_sec": round(total_time, 4),
                    "output_chars": len(text) if not has_error else 0,
                    "error": text if has_error else None,
                }

        tasks = [
            ocr_one(page, server_cycle[i])
            for i, page in enumerate(pages)
        ]
        results = await asyncio.gather(*tasks)

    return list(results)


async def run_warmup(servers: list[str], config: MAXBenchmarkConfig, sample_page: dict):
    """Send warmup requests to fill caches."""
    print(f"  Warming up ({config.warmup_requests} requests)...")
    prompt = get_classify_prompt("lite")

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=120),
    ) as session:
        tasks = []
        for i in range(config.warmup_requests):
            server = servers[i % len(servers)]
            tasks.append(classify_image_async(
                session, server, sample_page["image_b64"],
                model_name=config.model, prompt=prompt,
            ))
        await asyncio.gather(*tasks)
    print("  Warmup complete")


# =============================================================================
# METRICS
# =============================================================================

def compute_metrics(results: list[dict], wall_time: float) -> dict:
    """Compute throughput and latency metrics from a workload run."""
    successful = [r for r in results if r.get("error") is None]
    errors = [r for r in results if r.get("error") is not None]

    if not successful:
        return {
            "total_requests": len(results),
            "successful": 0,
            "errors": len(errors),
            "wall_time_sec": round(wall_time, 2),
            "pages_per_sec": 0,
            "pages_per_min": 0,
        }

    latencies = sorted(r["inference_time_sec"] for r in successful)
    n = len(latencies)

    return {
        "total_requests": len(results),
        "successful": n,
        "errors": len(errors),
        "wall_time_sec": round(wall_time, 2),
        "pages_per_sec": round(n / wall_time, 4) if wall_time > 0 else 0,
        "pages_per_min": round(n / wall_time * 60, 2) if wall_time > 0 else 0,
        "avg_latency_sec": round(sum(latencies) / n, 4),
        "p50_latency_sec": round(latencies[n // 2], 4),
        "p95_latency_sec": round(latencies[int(n * 0.95)], 4),
        "p99_latency_sec": round(latencies[min(int(n * 0.99), n - 1)], 4),
        "min_latency_sec": round(latencies[0], 4),
        "max_latency_sec": round(latencies[-1], 4),
    }


# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================

async def run_single_config(
    config: MAXBenchmarkConfig,
    all_pages: list[dict],
    log_dir: Path,
):
    """Run all workloads for a single config."""
    print(f"\n{'='*70}")
    print(f"CONFIG: {config.config_id}")
    print(f"  Model: {config.model}")
    print(f"  Quant={config.quantization_encoding}, GPUs/server={config.gpus_per_server}, "
          f"Servers={config.num_servers}")
    print(f"  batch_size={config.max_batch_size}, max_length={config.max_length}, "
          f"cache={config.cache_strategy}")
    if config.max_batch_context_length:
        print(f"  max_batch_context_length={config.max_batch_context_length}")
    print(f"{'='*70}")

    config.results_dir.mkdir(parents=True, exist_ok=True)

    print("\nStarting servers...")
    processes = await start_servers(config, log_dir=log_dir)

    try:
        print("Waiting for servers to be healthy...")
        healthy_servers = await wait_for_healthy(config, timeout=600)
        print(f"  {len(healthy_servers)} servers ready")

        n_pages = config.pages_per_workload
        sample = all_pages[:n_pages]

        await run_warmup(healthy_servers, config, sample[0])

        gpu_mem = snapshot_gpu_memory()

        workload_results = {}
        for workload in config.workloads:
            print(f"\n  Workload: {workload}")

            run_metrics = []
            for run_idx in range(config.num_runs):
                print(f"    Run {run_idx + 1}/{config.num_runs}...")
                t0 = time.monotonic()

                if workload == "classify":
                    results = await run_classify_workload(sample, healthy_servers, config)
                else:
                    results = await run_ocr_workload(sample, healthy_servers, config, workload)

                wall_time = time.monotonic() - t0
                metrics = compute_metrics(results, wall_time)
                run_metrics.append(metrics)

                print(f"      {metrics['pages_per_sec']:.2f} pages/s, "
                      f"p50={metrics.get('p50_latency_sec', 'N/A')}s, "
                      f"errors={metrics['errors']}")

                csv_path = config.results_dir / f"{workload}_run{run_idx}.csv"
                if results:
                    fieldnames = list(results[0].keys())
                    with open(csv_path, "w", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(results)

            workload_results[workload] = run_metrics

    except Exception as e:
        print(f"\n  ERROR: {e}")
        workload_results = {"error": str(e)}
        gpu_mem = snapshot_gpu_memory()
    finally:
        print("\nStopping servers...")
        stop_servers(processes)
        await asyncio.sleep(5)

    meta = {
        "config_id": config.config_id,
        "engine": "max",
        "model": config.model,
        "quantization_encoding": config.quantization_encoding,
        "gpus_per_server": config.gpus_per_server,
        "num_servers": config.num_servers,
        "max_batch_size": config.max_batch_size,
        "max_length": config.max_length,
        "max_batch_context_length": config.max_batch_context_length,
        "cache_strategy": config.cache_strategy,
        "concurrency_per_server": config.concurrency_per_server,
        "pages_per_workload": config.pages_per_workload,
        "num_runs": config.num_runs,
        "gpu_memory_snapshot": gpu_mem,
        "workload_results": workload_results,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    meta_path = config.results_dir / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved: {meta_path}")

    return meta


async def run_benchmark(args):
    """Main entry point."""
    if args.configs == "7b":
        configs = get_7b_configs()
    elif args.configs == "32b":
        configs = get_32b_configs()
    elif args.configs == "all":
        configs = get_all_configs()
    else:
        all_cfgs = {c.config_id: c for c in get_all_configs()}
        config_ids = [c.strip() for c in args.configs.split(",")]
        configs = []
        for cid in config_ids:
            if cid not in all_cfgs:
                print(f"ERROR: Unknown config '{cid}'. Available: {sorted(all_cfgs.keys())}")
                sys.exit(1)
            configs.append(all_cfgs[cid])

    print(f"MAX Serve Benchmark: {len(configs)} configs to test")
    print(f"Instance: p4d.24xlarge (8x A100 40GB)")

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"ERROR: Input directory not found: {input_dir}")
        sys.exit(1)

    pages = discover_pages(input_dir, max_pages=args.sample_size)
    pages = pre_render_pages(pages, dpi=args.dpi)
    if not pages:
        print("ERROR: No pages rendered successfully")
        sys.exit(1)

    random.seed(42)
    random.shuffle(pages)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log_dir = RESULTS_DIR / "server_logs"
    log_dir.mkdir(exist_ok=True)

    all_meta = []
    t_start = time.monotonic()

    for i, config in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] Running config: {config.config_id}")
        meta = await run_single_config(config, pages, log_dir)
        all_meta.append(meta)

    total_time = time.monotonic() - t_start
    print(f"\n{'='*70}")
    print(f"BENCHMARK COMPLETE")
    print(f"  Configs tested: {len(configs)}")
    print(f"  Total wall time: {total_time/60:.1f} min")
    print(f"  Results: {RESULTS_DIR}")
    print(f"{'='*70}")

    index_path = RESULTS_DIR / "benchmark_index.json"
    with open(index_path, "w") as f:
        json.dump({
            "engine": "max",
            "configs_tested": len(configs),
            "total_wall_time_sec": round(total_time, 1),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "configs": [m["config_id"] for m in all_meta],
        }, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="A100 MAX Serve Benchmark Runner")
    parser.add_argument(
        "--configs", type=str, default="all",
        help="Which configs to run: '7b', '32b', 'all', or comma-separated IDs",
    )
    parser.add_argument(
        "--input-dir", type=str, default=str(INPUT_DIR),
        help=f"Input directory with PDFs (default: {INPUT_DIR})",
    )
    parser.add_argument(
        "--sample-size", type=int, default=SAMPLE_SIZE,
        help=f"Number of pages to sample (default: {SAMPLE_SIZE})",
    )
    parser.add_argument(
        "--dpi", type=int, default=DPI,
        help=f"DPI for page rendering (default: {DPI})",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()
