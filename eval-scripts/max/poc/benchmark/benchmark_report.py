"""
MAX Serve Benchmark Report Generator

Loads results, computes summary statistics, cost analysis,
and outputs ranked tables per workload with vLLM baseline comparison.
"""

import csv
import json
import statistics
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from benchmark_config import RESULTS_DIR, P4D_COST_PER_HOUR

# vLLM results for cross-engine comparison
VLLM_RESULTS_DIR = SCRIPT_DIR.parent.parent.parent / "vllm" / "poc" / "results" / "benchmark_a100"


def load_all_results(results_dir: Path = RESULTS_DIR) -> list[dict]:
    """Load meta.json from each config subdirectory."""
    results = []
    for meta_path in sorted(results_dir.glob("*/meta.json")):
        with open(meta_path) as f:
            results.append(json.load(f))
    return results


def summarize_config(meta: dict) -> list[dict]:
    """Extract per-workload summary rows from a config's meta.json."""
    rows = []
    config_id = meta["config_id"]
    workload_results = meta.get("workload_results", {})

    if isinstance(workload_results, dict) and "error" in workload_results:
        rows.append({
            "config_id": config_id,
            "engine": meta.get("engine", "max"),
            "model": meta["model"],
            "quantization": meta.get("quantization_encoding", ""),
            "gpus_per_server": meta.get("gpus_per_server", meta.get("tp", "")),
            "num_servers": meta.get("num_servers", ""),
            "max_batch_size": meta.get("max_batch_size", meta.get("max_num_seqs", "")),
            "cache_strategy": meta.get("cache_strategy", ""),
            "workload": "ERROR",
            "error": workload_results["error"],
        })
        return rows

    for workload, run_metrics in workload_results.items():
        if not run_metrics:
            continue

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

        pages_per_hour = mean_throughput * 3600
        cost_per_1000 = (P4D_COST_PER_HOUR / pages_per_hour * 1000) if pages_per_hour > 0 else float("inf")

        gpu_mem = meta.get("gpu_memory_snapshot", {})

        row = {
            "config_id": config_id,
            "engine": meta.get("engine", "max"),
            "model": meta["model"].split("/")[-1],
            "quantization": meta.get("quantization_encoding", meta.get("quantization", "")),
            "gpus_per_server": meta.get("gpus_per_server", meta.get("tp", "")),
            "num_servers": meta.get("num_servers", ""),
            "max_batch_size": meta.get("max_batch_size", meta.get("max_num_seqs", "")),
            "cache_strategy": meta.get("cache_strategy", ""),
            "workload": workload,
            "mean_pages_per_sec": round(mean_throughput, 4),
            "std_pages_per_sec": round(std_throughput, 4),
            "mean_pages_per_min": round(mean_throughput * 60, 2),
            "mean_p50_latency_sec": round(statistics.mean(p50s), 4) if p50s else None,
            "mean_p95_latency_sec": round(statistics.mean(p95s), 4) if p95s else None,
            "mean_p99_latency_sec": round(statistics.mean(p99s), 4) if p99s else None,
            "mean_error_rate_pct": round(statistics.mean(error_rates), 2) if error_rates else 0,
            "cost_per_1000_pages": round(cost_per_1000, 4),
            "gpu_used_mb": gpu_mem.get("total_used_mb", 0),
            "gpu_util_pct": gpu_mem.get("utilization_pct", 0),
            "num_runs": len(run_metrics),
        }
        rows.append(row)

    return rows


def load_vllm_baselines() -> dict:
    """Load vLLM A100 benchmark results for comparison."""
    vllm_summary = VLLM_RESULTS_DIR / "summary.csv"
    if not vllm_summary.exists():
        return {}

    baselines = {}  # workload -> best config dict
    try:
        with open(vllm_summary) as f:
            reader = csv.DictReader(f)
            for row in reader:
                workload = row.get("workload", "")
                tps = float(row.get("mean_pages_per_sec", 0))
                if workload not in baselines or tps > baselines[workload]["tps"]:
                    baselines[workload] = {
                        "config_id": row.get("config_id", ""),
                        "tps": tps,
                        "p50": float(row.get("mean_p50_latency_sec", 0) or 0),
                        "cost": float(row.get("cost_per_1000_pages", 0) or 0),
                    }
    except Exception:
        pass
    return baselines


def generate_report(results_dir: Path = RESULTS_DIR):
    """Generate summary CSV and print ranked tables."""
    results = load_all_results(results_dir)
    if not results:
        print(f"No results found in {results_dir}")
        return

    print(f"Loaded results from {len(results)} configs")

    all_rows = []
    for meta in results:
        all_rows.extend(summarize_config(meta))

    if not all_rows:
        print("No workload results to report")
        return

    # Write summary CSV
    summary_path = results_dir / "summary.csv"
    fieldnames = [k for k in all_rows[0].keys() if k != "error"]
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

        print(f"\n{'='*100}")
        print(f"  WORKLOAD: {workload} — Ranked by throughput (MAX serve)")
        print(f"{'='*100}")
        print(f"{'Rank':<5} {'Config':<30} {'Quant':<8} {'pages/s':<10} {'pages/min':<11} "
              f"{'p50(s)':<8} {'p95(s)':<8} {'$/1K pg':<9} {'Err%':<6}")
        print("-" * 100)

        for i, r in enumerate(wl_rows[:10], 1):
            print(
                f"{i:<5} {r['config_id']:<30} "
                f"{r.get('quantization', ''):<8} "
                f"{r.get('mean_pages_per_sec', 0):<10.3f} "
                f"{r.get('mean_pages_per_min', 0):<11.1f} "
                f"{r.get('mean_p50_latency_sec', 0) or 0:<8.3f} "
                f"{r.get('mean_p95_latency_sec', 0) or 0:<8.3f} "
                f"${r.get('cost_per_1000_pages', 0):<8.3f} "
                f"{r.get('mean_error_rate_pct', 0):<6.1f}"
            )

    # vLLM A100 comparison
    vllm_baselines = load_vllm_baselines()

    print(f"\n{'='*100}")
    print("  vLLM A100 COMPARISON (same instance)")
    print(f"{'='*100}")

    for workload in workloads:
        wl_rows = [r for r in all_rows if r["workload"] == workload]
        if not wl_rows:
            continue
        best_max = max(wl_rows, key=lambda r: r.get("mean_pages_per_sec", 0))
        best_max_tps = best_max.get("mean_pages_per_sec", 0)

        if workload in vllm_baselines:
            vllm = vllm_baselines[workload]
            ratio = best_max_tps / vllm["tps"] if vllm["tps"] > 0 else 0
            print(f"  {workload}:")
            print(f"    MAX best:  {best_max_tps:.3f} pages/s ({best_max['config_id']})")
            print(f"    vLLM best: {vllm['tps']:.3f} pages/s ({vllm['config_id']})")
            print(f"    Ratio:     {ratio:.2f}x {'(MAX faster)' if ratio > 1 else '(vLLM faster)'}")
        else:
            print(f"  {workload}: MAX best={best_max_tps:.3f} pages/s ({best_max['config_id']}) — no vLLM baseline")

    # g5 baselines
    g5_baselines = {"simple_ocr": 0.36, "complex_ocr": 0.10}
    print(f"\n{'='*100}")
    print("  g5.12xlarge BASELINE COMPARISON (4xA10G vLLM)")
    print(f"{'='*100}")
    for workload in workloads:
        baseline = g5_baselines.get(workload)
        if baseline is None:
            continue
        wl_rows = [r for r in all_rows if r["workload"] == workload]
        if not wl_rows:
            continue
        best = max(wl_rows, key=lambda r: r.get("mean_pages_per_sec", 0))
        speedup = best.get("mean_pages_per_sec", 0) / baseline if baseline > 0 else 0
        print(f"  {workload}: best={best.get('mean_pages_per_sec', 0):.3f} pages/s "
              f"({best['config_id']}) vs baseline={baseline} → {speedup:.1f}x speedup")

    # Error configs
    error_rows = [r for r in all_rows if r.get("workload") == "ERROR"]
    if error_rows:
        print(f"\n  FAILED CONFIGS:")
        for r in error_rows:
            print(f"    {r['config_id']}: {r.get('error', 'unknown')}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate MAX benchmark report")
    parser.add_argument(
        "--results-dir", type=str, default=str(RESULTS_DIR),
        help=f"Results directory (default: {RESULTS_DIR})",
    )
    args = parser.parse_args()
    generate_report(Path(args.results_dir))
