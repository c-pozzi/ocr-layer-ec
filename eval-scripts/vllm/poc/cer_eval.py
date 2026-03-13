#!/usr/bin/env python3
"""
CER / WER evaluation with table-aware text normalization.

Compares OCR output against ground truth for complex pages, producing
per-document CER and WER in a CSV. Also used as a library by compare_app.py.

Usage:
    # Evaluate complex v2 results (default)
    python cer_eval.py

    # Evaluate a specific OCR directory
    python cer_eval.py --ocr-dir results/20260216/ocr_complex_v1

    # Include simple pages too
    python cer_eval.py --ocr-dir results/20260216/ocr_simple_v1

    # Also compute legacy PDF OCR metrics
    python cer_eval.py --include-pdf
"""

import argparse
import csv
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

from jiwer import cer, wer

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = Path(
    "/home/ubuntu/ocr-evaluation-samples/20260216_OCR_files/20260216_OCR_files"
)
DEFAULT_OCR_DIR = SCRIPT_DIR / "results" / "20260216" / "ocr_complex_v2"


# ---------------------------------------------------------------------------
# Stage 1: Strip markup / annotation artifacts
# ---------------------------------------------------------------------------

def strip_markup(text: str) -> str:
    """Remove OCR pipeline annotation tags, code fences, and HTML tags."""
    # Remove ```markdown ... ``` code fences
    text = re.sub(r"^```(?:markdown)?\s*\n?", "", text, flags=re.MULTILINE)
    text = re.sub(r"^```\s*$", "", text, flags=re.MULTILINE)
    # Remove [[TAG]] and [[/TAG]] annotation markers (HEADER, FOOTER, C1, ILLEGIBLE, H, etc.)
    text = re.sub(r"\[\[/?[A-Za-z0-9_:]+\]\]", "", text)
    # Remove <br> tags → space
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
    return text


# ---------------------------------------------------------------------------
# Stage 2: Table-aware cell extraction
# ---------------------------------------------------------------------------

_TABLE_SEP_RE = re.compile(
    r"^\s*\|[\s:+\-|]+\|\s*$"  # separator rows like |---|---|
)
_TABLE_ROW_RE = re.compile(
    r"^\s*\|(.+)\|\s*$"  # any row with pipes: | x | y |
)


def _extract_table_cells(row: str) -> list[str]:
    """Split a markdown table row into trimmed cell values."""
    # Strip leading/trailing pipes and split on inner pipes
    inner = row.strip().strip("|")
    return [cell.strip() for cell in inner.split("|")]


def extract_table_text(text: str) -> str:
    """Parse markdown tables into flat cell-value text; pass non-table lines through.

    Tables are parsed row by row. Separator rows (|---|---|) are dropped.
    Cell values are joined with a single space. Empty cells are skipped
    so that column-count differences between GT and OCR don't inflate CER.
    """
    lines = text.split("\n")
    out_lines: list[str] = []

    for line in lines:
        if _TABLE_SEP_RE.match(line):
            # Separator row — skip entirely
            continue
        m = _TABLE_ROW_RE.match(line)
        if m:
            cells = _extract_table_cells(line)
            # Keep only non-empty cells
            cell_text = " ".join(c for c in cells if c)
            if cell_text:
                out_lines.append(cell_text)
        else:
            out_lines.append(line)

    return "\n".join(out_lines)


# ---------------------------------------------------------------------------
# Stage 3: Text normalization
# ---------------------------------------------------------------------------

# Map various Unicode dashes to plain hyphen
_DASH_RE = re.compile(r"[\u2010\u2011\u2012\u2013\u2014\u2015\u2212\uFE58\uFE63\uFF0D]")


def normalize_text_core(text: str) -> str:
    """Lowercase, Unicode NFKC, normalize dashes, collapse whitespace."""
    # Unicode NFKC (normalizes ligatures, compatibility chars, etc.)
    text = unicodedata.normalize("NFKC", text)
    # Lowercase
    text = text.lower()
    # Normalize dashes
    text = _DASH_RE.sub("-", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


# ---------------------------------------------------------------------------
# Full normalization pipeline
# ---------------------------------------------------------------------------

def normalize_full(text: str) -> str:
    """Full normalization pipeline: markup → tables → text."""
    text = strip_markup(text)
    text = extract_table_text(text)
    text = normalize_text_core(text)
    return text


def normalize_light(text: str) -> str:
    """Light normalization (no table extraction) for comparison."""
    text = strip_markup(text)
    text = normalize_text_core(text)
    return text


# ---------------------------------------------------------------------------
# PDF text extraction (for optional legacy comparison)
# ---------------------------------------------------------------------------

def extract_pdf_text(pdf_path: Path) -> str:
    """Extract text from PDF using pdftotext -layout."""
    try:
        result = subprocess.run(
            ["pdftotext", "-layout", str(pdf_path), "-"],
            capture_output=True, text=True, timeout=30,
        )
        return result.stdout if result.returncode == 0 else ""
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(reference: str, hypothesis: str) -> dict:
    """Compute CER and WER between reference and hypothesis strings."""
    if not reference:
        return {"cer": 1.0 if hypothesis else 0.0, "wer": 1.0 if hypothesis else 0.0}
    return {
        "cer": cer(reference, hypothesis),
        "wer": wer(reference, hypothesis),
    }


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate_directory(
    ocr_dir: Path,
    input_dir: Path,
    skip_pdf: bool = False,
) -> list[dict]:
    """Evaluate all OCR files in a directory against ground truth.

    By default, also evaluates the legacy PDF OCR text layer (extracted
    via ``pdftotext -layout``) for comparison.  Pass *skip_pdf=True* to
    disable this.

    Returns a list of dicts with per-document metrics.
    """
    ocr_files = sorted(ocr_dir.glob("*.txt"))
    results = []

    for ocr_file in ocr_files:
        doc_id = ocr_file.stem
        gt_path = input_dir / f"{doc_id}_ground_truth.txt"

        if not gt_path.exists():
            print(f"  SKIP {doc_id}: no ground truth")
            continue

        ocr_raw = ocr_file.read_text(encoding="utf-8")
        gt_raw = gt_path.read_text(encoding="utf-8")

        # Full normalization (table-aware)
        ocr_norm = normalize_full(ocr_raw)
        gt_norm = normalize_full(gt_raw)
        metrics = compute_metrics(gt_norm, ocr_norm)

        row = {
            "doc_id": doc_id,
            "gt_chars": len(gt_norm),
            "ocr_chars": len(ocr_norm),
            "cer": round(metrics["cer"], 4),
            "wer": round(metrics["wer"], 4),
        }

        # Legacy PDF OCR text layer comparison (default: enabled)
        if not skip_pdf:
            pdf_path = input_dir / f"{doc_id}.pdf"
            if pdf_path.exists():
                pdf_raw = extract_pdf_text(pdf_path)
                pdf_norm = normalize_full(pdf_raw)
                pdf_metrics = compute_metrics(gt_norm, pdf_norm)
                row["pdf_chars"] = len(pdf_norm)
                row["pdf_cer"] = round(pdf_metrics["cer"], 4)
                row["pdf_wer"] = round(pdf_metrics["wer"], 4)
                row["cer_improvement"] = round(row["pdf_cer"] - row["cer"], 4)
                row["wer_improvement"] = round(row["pdf_wer"] - row["wer"], 4)

        results.append(row)

    return results


def write_csv(results: list[dict], output_path: Path):
    """Write evaluation results to CSV."""
    if not results:
        print("No results to write.")
        return

    fieldnames = list(results[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def print_summary(results: list[dict]):
    """Print summary statistics to stdout."""
    if not results:
        print("No results.")
        return

    cers = [r["cer"] for r in results]
    wers = [r["wer"] for r in results]

    import statistics
    print(f"\n{'='*60}")
    print(f"  Documents evaluated: {len(results)}")
    print(f"  CER  — mean: {statistics.mean(cers):.4f}  median: {statistics.median(cers):.4f}  "
          f"min: {min(cers):.4f}  max: {max(cers):.4f}")
    print(f"  WER  — mean: {statistics.mean(wers):.4f}  median: {statistics.median(wers):.4f}  "
          f"min: {min(wers):.4f}  max: {max(wers):.4f}")

    has_pdf = "pdf_cer" in results[0]
    if has_pdf:
        pdf_cers = [r["pdf_cer"] for r in results if "pdf_cer" in r]
        pdf_wers = [r["pdf_wer"] for r in results if "pdf_wer" in r]
        cer_impr = [r["cer_improvement"] for r in results if "cer_improvement" in r]
        wer_impr = [r["wer_improvement"] for r in results if "wer_improvement" in r]
        print(f"  PDF CER — mean: {statistics.mean(pdf_cers):.4f}  median: {statistics.median(pdf_cers):.4f}  "
              f"min: {min(pdf_cers):.4f}  max: {max(pdf_cers):.4f}")
        print(f"  PDF WER — mean: {statistics.mean(pdf_wers):.4f}  median: {statistics.median(pdf_wers):.4f}  "
              f"min: {min(pdf_wers):.4f}  max: {max(pdf_wers):.4f}")
        print(f"  CER improvement (PDF-AI) — mean: {statistics.mean(cer_impr):+.4f}  median: {statistics.median(cer_impr):+.4f}")
        print(f"  WER improvement (PDF-AI) — mean: {statistics.mean(wer_impr):+.4f}  median: {statistics.median(wer_impr):+.4f}")

    print(f"{'='*60}\n")

    # Per-document table
    hdr = f"  {'doc_id':<40} {'CER':>8} {'WER':>8}"
    sep = f"  {'-'*40} {'-'*8} {'-'*8}"
    if has_pdf:
        hdr += f" {'PDF_CER':>8} {'PDF_WER':>8} {'CER_imp':>8} {'WER_imp':>8}"
        sep += f" {'-'*8} {'-'*8} {'-'*8} {'-'*8}"
    print(hdr)
    print(sep)

    for r in sorted(results, key=lambda x: x["cer"], reverse=True):
        line = f"  {r['doc_id']:<40} {r['cer']:>8.4f} {r['wer']:>8.4f}"
        if has_pdf and "pdf_cer" in r:
            line += (f" {r['pdf_cer']:>8.4f} {r['pdf_wer']:>8.4f}"
                     f" {r.get('cer_improvement', 0):>+8.4f} {r.get('wer_improvement', 0):>+8.4f}")
        print(line)


def main():
    parser = argparse.ArgumentParser(description="CER/WER evaluation with table-aware normalization")
    parser.add_argument(
        "--ocr-dir", type=Path, default=DEFAULT_OCR_DIR,
        help="Directory with OCR output .txt files",
    )
    parser.add_argument(
        "--input-dir", type=Path, default=DEFAULT_INPUT_DIR,
        help="Directory with ground truth and PDF files",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output CSV path (default: <ocr-dir>/cer_wer_results.csv)",
    )
    parser.add_argument(
        "--skip-pdf", action="store_true",
        help="Skip legacy PDF OCR text layer evaluation",
    )
    args = parser.parse_args()

    output_path = args.output or (args.ocr_dir / "cer_wer_results.csv")

    print(f"OCR dir:   {args.ocr_dir}")
    print(f"Input dir: {args.input_dir}")
    print(f"Output:    {output_path}")
    if not args.skip_pdf:
        print(f"PDF OCR:   enabled (pdftotext -layout)")
    print()

    results = evaluate_directory(args.ocr_dir, args.input_dir, args.skip_pdf)
    write_csv(results, output_path)
    print_summary(results)
    print(f"Results written to {output_path}")


if __name__ == "__main__":
    main()
