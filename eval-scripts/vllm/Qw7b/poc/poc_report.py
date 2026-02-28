"""
POC Phase 3: Classification Report

Reads classification.csv and generates comprehensive statistics:
- Classification distribution (simple vs complex, per-feature)
- Per-page timing statistics (render, classify, queue, total)
- Throughput analysis (pages/sec, per-GPU)
- Error summary
- Feature co-occurrence matrix

Usage:
    python poc_report.py
    python poc_report.py --classification results/classification.csv
    python poc_report.py --classification results/classification.csv --save

Output: printed to stdout + optionally saved to results/poc_report.txt
"""

import argparse
import csv
import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"

CLASSIFICATION_FIELDS = [
    "multi_column", "has_tables", "handwritten", "has_stamps",
    "poor_quality", "has_strikethrough", "latin_script",
    "has_footnotes", "has_forms",
]

TIMING_FIELDS = [
    "render_time_sec", "classify_time_sec", "queue_wait_sec", "total_time_sec",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate POC classification report")
    parser.add_argument(
        "--classification",
        type=str,
        default=str(RESULTS_DIR / "classification.csv"),
        help="Path to classification CSV (default: results/classification.csv)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save report to results/poc_report.txt",
    )
    return parser.parse_args()


# =============================================================================
# DATA LOADING
# =============================================================================

def load_classification(path: str) -> list[dict]:
    """Load classification CSV, casting types appropriately."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Cast numeric fields
            for field in TIMING_FIELDS:
                try:
                    row[field] = float(row[field]) if row[field] else None
                except (ValueError, TypeError):
                    row[field] = None

            # Cast boolean fields
            for field in CLASSIFICATION_FIELDS:
                val = row.get(field, "")
                if val in ("True", "true", "1"):
                    row[field] = True
                elif val in ("False", "false", "0"):
                    row[field] = False
                else:
                    row[field] = None

            # Cast is_complex
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
# STATISTICS HELPERS
# =============================================================================

def percentile(sorted_values: list[float], p: float) -> float:
    """Calculate percentile from sorted list."""
    if not sorted_values:
        return 0.0
    k = (len(sorted_values) - 1) * (p / 100.0)
    f = int(k)
    c = f + 1
    if c >= len(sorted_values):
        return sorted_values[-1]
    return sorted_values[f] + (k - f) * (sorted_values[c] - sorted_values[f])


def compute_stats(values: list[float]) -> dict:
    """Compute summary statistics for a list of values."""
    if not values:
        return {"count": 0, "mean": 0, "median": 0, "p95": 0, "p99": 0, "min": 0, "max": 0}
    sv = sorted(values)
    return {
        "count": len(sv),
        "mean": sum(sv) / len(sv),
        "median": percentile(sv, 50),
        "p95": percentile(sv, 95),
        "p99": percentile(sv, 99),
        "min": sv[0],
        "max": sv[-1],
    }


# =============================================================================
# REPORT SECTIONS
# =============================================================================

def section_distribution(rows: list[dict]) -> str:
    """1. Classification Distribution."""
    lines = []
    lines.append("=" * 70)
    lines.append("1. CLASSIFICATION DISTRIBUTION")
    lines.append("=" * 70)

    total = len(rows)
    valid = [r for r in rows if r.get("error") is None or r["error"] in ("", "None", None)]
    errors = total - len(valid)

    lines.append(f"Total pages:      {total}")
    lines.append(f"Successfully classified: {len(valid)}")
    lines.append(f"Errors:           {errors}")
    lines.append("")

    # Simple vs Complex
    complex_pages = [r for r in valid if r.get("is_complex") is True]
    simple_pages = [r for r in valid if r.get("is_complex") is False]

    lines.append(f"Simple (Tier 1):  {len(simple_pages):>6}  ({100*len(simple_pages)/len(valid):.1f}%)" if valid else "")
    lines.append(f"Complex (Tier 2): {len(complex_pages):>6}  ({100*len(complex_pages)/len(valid):.1f}%)" if valid else "")
    lines.append("")

    # Per-feature breakdown
    lines.append("Per-feature breakdown:")
    lines.append(f"  {'Feature':<22} {'Count':>6}  {'%':>6}")
    lines.append(f"  {'-'*22} {'-'*6}  {'-'*6}")
    for field in CLASSIFICATION_FIELDS:
        count = sum(1 for r in valid if r.get(field) is True)
        pct = 100 * count / len(valid) if valid else 0
        lines.append(f"  {field:<22} {count:>6}  {pct:>5.1f}%")

    lines.append("")

    # Feature co-occurrence
    lines.append("Feature co-occurrence (top 10 pairs):")
    pair_counts = Counter()
    for r in valid:
        active = [f for f in CLASSIFICATION_FIELDS if r.get(f) is True]
        for a, b in combinations(active, 2):
            pair_counts[(a, b)] += 1

    for (a, b), count in pair_counts.most_common(10):
        pct = 100 * count / len(valid) if valid else 0
        lines.append(f"  {a} + {b}: {count} ({pct:.1f}%)")

    return "\n".join(lines)


def section_timing(rows: list[dict]) -> str:
    """2. Per-Page Timing Statistics."""
    lines = []
    lines.append("")
    lines.append("=" * 70)
    lines.append("2. PER-PAGE TIMING STATISTICS")
    lines.append("=" * 70)

    # Filter to successful rows with timing data
    valid = [r for r in rows if r.get("error") in ("", "None", None) and r.get("classify_time_sec") is not None]

    for field in TIMING_FIELDS:
        label = field.replace("_sec", "").replace("_", " ").title()
        values = [r[field] for r in valid if r[field] is not None]
        stats = compute_stats(values)

        lines.append(f"\n  {label} (seconds):")
        lines.append(f"    Count:  {stats['count']}")
        lines.append(f"    Mean:   {stats['mean']:.4f}")
        lines.append(f"    Median: {stats['median']:.4f}")
        lines.append(f"    P95:    {stats['p95']:.4f}")
        lines.append(f"    P99:    {stats['p99']:.4f}")
        lines.append(f"    Min:    {stats['min']:.4f}")
        lines.append(f"    Max:    {stats['max']:.4f}")

    return "\n".join(lines)


def section_throughput(rows: list[dict], meta: dict | None = None) -> str:
    """3. Throughput Analysis."""
    lines = []
    lines.append("")
    lines.append("=" * 70)
    lines.append("3. THROUGHPUT")
    lines.append("=" * 70)

    valid = [r for r in rows if r.get("total_time_sec") is not None]
    if not valid:
        lines.append("  No valid timing data.")
        return "\n".join(lines)

    total_classify_time = sum(r["classify_time_sec"] for r in valid if r["classify_time_sec"])

    servers = set(r.get("server", "") for r in valid if r.get("server"))
    num_gpus = len(servers) if servers else 4

    # Per-page averages
    avg_total = sum(r["total_time_sec"] for r in valid) / len(valid)
    avg_classify = total_classify_time / len(valid) if valid else 0

    # Use actual wall time from metadata if available, otherwise estimate
    if meta and meta.get("wall_time_sec"):
        wall_time = meta["wall_time_sec"]
        concurrency = meta.get("total_concurrency", num_gpus)
        wall_time_source = "actual"
    else:
        concurrency = num_gpus
        wall_time = total_classify_time / concurrency if concurrency > 0 else total_classify_time
        wall_time_source = "estimated"

    overall_rate = len(valid) / wall_time if wall_time > 0 else 0
    per_gpu_rate = overall_rate / num_gpus if num_gpus > 0 else 0

    lines.append(f"  Pages classified:         {len(valid)}")
    lines.append(f"  GPUs used:                {num_gpus}")
    lines.append(f"  Concurrency:              {concurrency}")
    lines.append(f"  Avg classify time/page:   {avg_classify:.4f}s")
    lines.append(f"  Avg total time/page:      {avg_total:.4f}s")
    lines.append(f"  Total GPU-seconds:        {total_classify_time:.1f}s")
    lines.append(f"  Wall time ({wall_time_source:>9}):   {wall_time:.1f}s ({wall_time/60:.1f}min)")
    lines.append(f"  Overall throughput:       {overall_rate:.1f} pages/sec")
    lines.append(f"  Per-GPU throughput:       {per_gpu_rate:.1f} pages/sec/GPU")

    # Extrapolation
    for target in [9000, 50000, 600000]:
        est_time = target / overall_rate if overall_rate > 0 else 0
        lines.append(f"  Extrapolation ({target:>6} pages): {est_time:.0f}s ({est_time/60:.1f}min)")

    return "\n".join(lines)


def section_errors(rows: list[dict]) -> str:
    """4. Error Summary."""
    lines = []
    lines.append("")
    lines.append("=" * 70)
    lines.append("4. ERRORS")
    lines.append("=" * 70)

    error_rows = [r for r in rows if r.get("error") and r["error"] not in ("", "None")]
    lines.append(f"  Total errors: {len(error_rows)} / {len(rows)} ({100*len(error_rows)/len(rows):.2f}%)" if rows else "  No data.")

    if error_rows:
        # Group by error type
        error_types = Counter()
        for r in error_rows:
            error_msg = str(r["error"])
            # Truncate long error messages for grouping
            if len(error_msg) > 80:
                error_msg = error_msg[:80] + "..."
            error_types[error_msg] += 1

        lines.append("")
        lines.append("  Error types:")
        for err, count in error_types.most_common(10):
            lines.append(f"    [{count:>4}x] {err}")

        # Parse failures specifically
        parse_failures = [r for r in error_rows if "JSON parse" in str(r.get("error", ""))]
        if parse_failures:
            lines.append(f"\n  JSON parse failures: {len(parse_failures)}")
    else:
        lines.append("  No errors encountered.")

    return "\n".join(lines)


def section_per_pdf_summary(rows: list[dict]) -> str:
    """5. Per-PDF summary (top documents by complexity)."""
    lines = []
    lines.append("")
    lines.append("=" * 70)
    lines.append("5. PER-PDF SUMMARY (top 20 by complexity ratio)")
    lines.append("=" * 70)

    # Group by PDF
    pdf_groups = {}
    for r in rows:
        fname = r["pdf_filename"]
        if fname not in pdf_groups:
            pdf_groups[fname] = {"total": 0, "complex": 0, "simple": 0, "errors": 0}
        pdf_groups[fname]["total"] += 1
        if r.get("is_complex") is True:
            pdf_groups[fname]["complex"] += 1
        elif r.get("is_complex") is False:
            pdf_groups[fname]["simple"] += 1
        if r.get("error") and r["error"] not in ("", "None"):
            pdf_groups[fname]["errors"] += 1

    # Sort by complexity ratio descending
    sorted_pdfs = sorted(
        pdf_groups.items(),
        key=lambda x: x[1]["complex"] / max(x[1]["total"], 1),
        reverse=True,
    )

    lines.append(f"  {'PDF':<45} {'Pages':>5} {'Complex':>7} {'%':>6}")
    lines.append(f"  {'-'*45} {'-'*5} {'-'*7} {'-'*6}")
    for fname, stats in sorted_pdfs[:20]:
        pct = 100 * stats["complex"] / max(stats["total"], 1)
        display_name = fname if len(fname) <= 45 else fname[:42] + "..."
        lines.append(f"  {display_name:<45} {stats['total']:>5} {stats['complex']:>7} {pct:>5.1f}%")

    lines.append(f"\n  Total PDFs: {len(pdf_groups)}")

    return "\n".join(lines)


# =============================================================================
# MAIN
# =============================================================================

def load_metadata(classification_path: Path) -> dict | None:
    """Load metadata JSON sidecar if it exists."""
    meta_path = classification_path.with_suffix(".meta.json")
    if meta_path.exists():
        with open(meta_path, "r") as f:
            return json.load(f)
    return None


def generate_report(rows: list[dict], meta: dict | None = None) -> str:
    """Generate full report string."""
    sections = [
        "",
        "POC CLASSIFICATION REPORT",
        "=" * 70,
        "",
        section_distribution(rows),
        section_timing(rows),
        section_throughput(rows, meta),
        section_errors(rows),
        section_per_pdf_summary(rows),
        "",
        "=" * 70,
        "END OF REPORT",
        "=" * 70,
    ]
    return "\n".join(sections)


def main():
    args = parse_args()

    classification_path = Path(args.classification)
    if not classification_path.exists():
        print(f"ERROR: Classification file not found: {classification_path}")
        sys.exit(1)

    print(f"Loading: {classification_path}")
    rows = load_classification(str(classification_path))
    meta = load_metadata(classification_path)
    if meta:
        print(f"Loaded metadata: wall_time={meta['wall_time_sec']}s, concurrency={meta.get('total_concurrency')}")
    print(f"Loaded {len(rows)} rows\n")

    report = generate_report(rows, meta)
    print(report)

    if args.save:
        report_path = RESULTS_DIR / "poc_report.txt"
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
