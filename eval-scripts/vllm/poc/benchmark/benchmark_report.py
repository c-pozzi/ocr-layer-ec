"""
Benchmark Report Generator

Loads results from all configs, computes summary statistics, cost analysis,
and outputs ranked tables per workload.
"""

import csv
import json
import statistics
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from benchmark_config import RESULTS_DIR, P4D_COST_PER_HOUR


def load_all_results(results_dir: Path = RESULTS_DIR) -> list[dict]:
    """Load meta.json from each config subdirectory."""
    results = []
    for meta_path in sorted(results_dir.glob("*/meta.json")):
        with open(meta_path) as f:
            results.append(json.load(f))
    return results


def summarize_config(meta: dict) -> list[dict]:
    """
    Extract per-workload summary rows from a config's meta.json.
    Returns list of flat dicts suitable for CSV output.
    """
    rows = []
    config_id = meta["config_id"]
    workload_results = meta.get("workload_results", {})

    if isinstance(workload_results, dict) and "error" in workload_results:
        rows.append({
            "config_id": config_id,
            "model": meta["model"],
            "tp": meta["tp"],
            "num_servers": meta["num_servers"],
            "cuda_graphs": meta["cuda_graphs"],
            "gpu_mem_util": meta["gpu_mem_util"],
            "max_num_seqs": meta["max_num_seqs"],
            "workload": "ERROR",
            "error": workload_results["error"],
        })
        return rows

    for workload, run_metrics in workload_results.items():
        if not run_metrics:
            continue

        # Aggregate across runs
        throughputs = [m["pages_per_sec"] for m in run_metrics if m.get("pages_per_sec", 0) > 0]
        p50s = [m["p50_latency_sec"] for m in run_metrics if "p50_latency_sec" in m]
        p95s = [m["p95_latency_sec"] for m in run_metrics if "p95_latency_sec" in m]
        p99s = [m["p99_latency_sec"] for m in run_metrics if "p99_latency_sec" in m]
        error_rates = [
            m["errors"] / m["total_requests"] * 100
            for m in run_metrics if m["total_requests"] > 0
        ]

        if not throughputs:
            continue

        mean_throughput = statistics.mean(throughputs)
        std_throughput = statistics.stdev(throughputs) if len(throughputs) > 1 else 0

        # Cost: p4d.24xlarge per hour / pages per hour
        pages_per_hour = mean_throughput * 3600
        cost_per_1000 = (P4D_COST_PER_HOUR / pages_per_hour * 1000) if pages_per_hour > 0 else float("inf")

        # GPU memory
        gpu_mem = meta.get("gpu_memory_snapshot", {})
        gpu_used_mb = gpu_mem.get("total_used_mb", 0)
        gpu_util_pct = gpu_mem.get("utilization_pct", 0)

        row = {
            "config_id": config_id,
            "model": meta["model"].split("/")[-1],
            "tp": meta["tp"],
            "num_servers": meta["num_servers"],
            "cuda_graphs": meta["cuda_graphs"],
            "gpu_mem_util": meta["gpu_mem_util"],
            "max_num_seqs": meta["max_num_seqs"],
            "max_num_batched_tokens": meta.get("max_num_batched_tokens"),
            "workload": workload,
            "mean_pages_per_sec": round(mean_throughput, 4),
            "std_pages_per_sec": round(std_throughput, 4),
            "mean_pages_per_min": round(mean_throughput * 60, 2),
            "mean_p50_latency_sec": round(statistics.mean(p50s), 4) if p50s else None,
            "mean_p95_latency_sec": round(statistics.mean(p95s), 4) if p95s else None,
            "mean_p99_latency_sec": round(statistics.mean(p99s), 4) if p99s else None,
            "mean_error_rate_pct": round(statistics.mean(error_rates), 2) if error_rates else 0,
            "cost_per_1000_pages": round(cost_per_1000, 4),
            "gpu_used_mb": gpu_used_mb,
            "gpu_util_pct": gpu_util_pct,
            "num_runs": len(run_metrics),
        }
        rows.append(row)

    return rows


def generate_report(results_dir: Path = RESULTS_DIR):
    """Generate summary CSV and print ranked tables."""
    results = load_all_results(results_dir)
    if not results:
        print(f"No results found in {results_dir}")
        return

    print(f"Loaded results from {len(results)} configs")

    # Summarize all configs
    all_rows = []
    for meta in results:
        all_rows.extend(summarize_config(meta))

    if not all_rows:
        print("No workload results to report")
        return

    # Write summary CSV
    summary_path = results_dir / "summary.csv"
    fieldnames = [k for k in all_rows[0].keys() if k != "error"]
    # Include error column only if present
    if any("error" in r for r in all_rows):
        fieldnames.append("error")

    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Wrote summary: {summary_path}")

    # Print ranked tables per workload
    workloads = sorted(set(r["workload"] for r in all_rows if r["workload"] != "ERROR"))

    for workload in workloads:
        wl_rows = [r for r in all_rows if r["workload"] == workload]
        wl_rows.sort(key=lambda r: r.get("mean_pages_per_sec", 0), reverse=True)

        print(f"\n{'='*90}")
        print(f"  WORKLOAD: {workload} — Ranked by throughput")
        print(f"{'='*90}")
        print(f"{'Rank':<5} {'Config':<30} {'pages/s':<10} {'pages/min':<11} "
              f"{'p50(s)':<8} {'p95(s)':<8} {'$/1K pg':<9} {'Err%':<6}")
        print("-" * 90)

        for i, r in enumerate(wl_rows[:10], 1):
            print(
                f"{i:<5} {r['config_id']:<30} "
                f"{r.get('mean_pages_per_sec', 0):<10.3f} "
                f"{r.get('mean_pages_per_min', 0):<11.1f} "
                f"{r.get('mean_p50_latency_sec', 0) or 0:<8.3f} "
                f"{r.get('mean_p95_latency_sec', 0) or 0:<8.3f} "
                f"${r.get('cost_per_1000_pages', 0):<8.3f} "
                f"{r.get('mean_error_rate_pct', 0):<6.1f}"
            )

    # Baseline comparison
    print(f"\n{'='*90}")
    print("  BASELINE COMPARISON (g5.12xlarge 4xA10G)")
    print(f"{'='*90}")

    baselines = {
        "classify": None,  # No baseline for classifier
        "simple_ocr": 0.36,
        "complex_ocr": 0.10,
    }

    for workload in workloads:
        baseline = baselines.get(workload)
        if baseline is None:
            continue
        wl_rows = [r for r in all_rows if r["workload"] == workload]
        if not wl_rows:
            continue
        best = max(wl_rows, key=lambda r: r.get("mean_pages_per_sec", 0))
        best_tps = best.get("mean_pages_per_sec", 0)
        speedup = best_tps / baseline if baseline > 0 else 0
        print(f"  {workload}: best={best_tps:.3f} pages/s ({best['config_id']}) "
              f"vs baseline={baseline} pages/s → {speedup:.1f}x speedup")

    # Error configs
    error_rows = [r for r in all_rows if r.get("workload") == "ERROR"]
    if error_rows:
        print(f"\n  FAILED CONFIGS:")
        for r in error_rows:
            print(f"    {r['config_id']}: {r.get('error', 'unknown')}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate benchmark report")
    parser.add_argument(
        "--results-dir", type=str, default=str(RESULTS_DIR),
        help=f"Results directory (default: {RESULTS_DIR})",
    )
    args = parser.parse_args()
    generate_report(Path(args.results_dir))
