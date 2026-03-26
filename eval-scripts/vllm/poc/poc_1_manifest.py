"""
POC Phase 1: Build Page Manifest

Scans local PDF directories (or downloads from S3) and creates a manifest
with one row per page. Supports --max-pages to cap total page count.

Usage:
    python poc_1_manifest.py --input ~/ocr-input/BAC-0002-1971 --model awq --max-pages 9000

Output: results/<model>__<input_basename>/manifest.csv
"""

import argparse
import csv
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import fitz  # PyMuPDF

from poc_utils import MODEL_CONFIGS, RESULTS_DIR, make_run_dir

# S3 staging directory (fast NVMe)
S3_STAGING_DIR = Path("/opt/dlami/nvme/poc_staging")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build page manifest from local or S3 PDFs"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Local directory path or S3 URI (s3://bucket/prefix)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Stop adding PDFs once cumulative page count reaches this value",
    )
    parser.add_argument(
        "--model",
        choices=list(MODEL_CONFIGS.keys()),
        default="awq",
        help="Model variant (default: awq)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path (default: results/<model>__<input_basename>/manifest.csv)",
    )
    return parser.parse_args()


# =============================================================================
# LOCAL PDF SCANNING
# =============================================================================

def scan_local_pdfs(input_dir: Path) -> list[dict]:
    """
    Scan local directory for PDFs. Looks in subdirectories (folder-per-document).

    Returns list of dicts with pdf info, sorted by folder then filename.
    """
    input_dir = Path(input_dir).resolve()
    pdfs = []

    # Check for PDFs directly in input_dir
    for pdf_path in sorted(input_dir.glob("*.pdf")):
        pdfs.append({
            "pdf_path": pdf_path,
            "pdf_filename": pdf_path.name,
            "folder": input_dir.name,
        })

    # Check subdirectories
    for subdir in sorted(input_dir.iterdir()):
        if subdir.is_dir():
            for pdf_path in sorted(subdir.glob("*.pdf")):
                pdfs.append({
                    "pdf_path": pdf_path,
                    "pdf_filename": pdf_path.name,
                    "folder": subdir.name,
                })

    return pdfs


# =============================================================================
# S3 PDF DOWNLOADING
# =============================================================================

def download_s3_pdfs(s3_uri: str, max_pages: int | None = None) -> list[dict]:
    """
    Download PDFs from S3 to local staging directory.

    Uses boto3 with ThreadPoolExecutor for concurrent downloads.
    """
    try:
        import boto3
    except ImportError:
        print("ERROR: boto3 required for S3 access. Install with: pip install boto3")
        sys.exit(1)

    # Parse s3://bucket/prefix
    parts = s3_uri.replace("s3://", "").split("/", 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""

    S3_STAGING_DIR.mkdir(parents=True, exist_ok=True)

    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")

    # List all PDF objects
    pdf_keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].lower().endswith(".pdf"):
                pdf_keys.append(obj["Key"])

    print(f"Found {len(pdf_keys)} PDFs in s3://{bucket}/{prefix}")

    def download_one(key: str) -> Path:
        local_path = S3_STAGING_DIR / Path(key).name
        if not local_path.exists():
            s3.download_file(bucket, key, str(local_path))
        return local_path

    # Concurrent download
    downloaded = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(download_one, k): k for k in pdf_keys}
        for future in as_completed(futures):
            key = futures[future]
            try:
                local_path = future.result()
                downloaded.append({
                    "pdf_path": local_path,
                    "pdf_filename": local_path.name,
                    "folder": "s3",
                })
            except Exception as e:
                print(f"  WARN: Failed to download {key}: {e}")

    return sorted(downloaded, key=lambda d: d["pdf_filename"])


# =============================================================================
# MANIFEST BUILDER
# =============================================================================

def build_manifest(pdfs: list[dict], max_pages: int | None = None) -> list[dict]:
    """
    Build page-level manifest from list of PDF info dicts.

    Each PDF is opened with fitz for fast page counting (no rendering).
    Stops adding PDFs once cumulative pages reach max_pages.
    """
    manifest = []
    total_pages = 0
    pdfs_included = 0

    for pdf_info in pdfs:
        pdf_path = pdf_info["pdf_path"]

        try:
            doc = fitz.open(pdf_path)
            page_count = len(doc)
            file_size = pdf_path.stat().st_size
            doc.close()
        except Exception as e:
            print(f"  WARN: Cannot open {pdf_path.name}: {e}")
            continue

        if page_count == 0:
            continue

        # Check if adding this PDF would exceed max_pages
        if max_pages is not None and total_pages + page_count > max_pages:
            # Add partial — only enough pages to reach max_pages
            remaining = max_pages - total_pages
            if remaining <= 0:
                break
            for page_num in range(1, remaining + 1):
                manifest.append({
                    "pdf_filename": pdf_info["pdf_filename"],
                    "folder": pdf_info["folder"],
                    "local_path": str(pdf_info["pdf_path"]),
                    "page_num": page_num,
                    "total_pages": page_count,
                    "file_size_bytes": file_size,
                })
            total_pages += remaining
            pdfs_included += 1
            break

        # Add all pages from this PDF
        for page_num in range(1, page_count + 1):
            manifest.append({
                "pdf_filename": pdf_info["pdf_filename"],
                "folder": pdf_info["folder"],
                "local_path": str(pdf_info["pdf_path"]),
                "page_num": page_num,
                "total_pages": page_count,
                "file_size_bytes": file_size,
            })

        total_pages += page_count
        pdfs_included += 1

        if max_pages is not None and total_pages >= max_pages:
            break

    return manifest, pdfs_included


def main():
    args = parse_args()
    input_str = args.input

    # Determine source
    if input_str.startswith("s3://"):
        print(f"Scanning S3: {input_str}")
        pdfs = download_s3_pdfs(input_str, args.max_pages)
    else:
        input_dir = Path(input_str).resolve()
        if not input_dir.exists():
            print(f"ERROR: Input directory does not exist: {input_dir}")
            sys.exit(1)
        print(f"Scanning local: {input_dir}")
        pdfs = scan_local_pdfs(input_dir)

    if not pdfs:
        print("No PDFs found.")
        sys.exit(1)

    print(f"Found {len(pdfs)} PDF files")
    if args.max_pages:
        print(f"Max pages cap: {args.max_pages}")

    # Resolve output path
    if args.output:
        output_path = Path(args.output)
    else:
        run_dir = make_run_dir(args.model, input_str)
        run_dir.mkdir(parents=True, exist_ok=True)
        output_path = run_dir / "manifest.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build manifest
    print("Building page manifest (counting pages)...")
    manifest, pdfs_included = build_manifest(pdfs, args.max_pages)

    if not manifest:
        print("No pages to include.")
        sys.exit(1)

    # Write CSV
    fieldnames = [
        "pdf_filename", "folder", "local_path",
        "page_num", "total_pages", "file_size_bytes",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest)

    # Summary
    unique_pdfs = len(set(r["pdf_filename"] for r in manifest))
    unique_folders = len(set(r["folder"] for r in manifest))
    total_size_mb = sum(
        r["file_size_bytes"]
        for r in {r["pdf_filename"]: r for r in manifest}.values()
    ) / (1024 * 1024)

    print(f"\nManifest written to: {output_path}")
    print(f"  Total pages: {len(manifest)}")
    print(f"  PDFs included: {unique_pdfs}")
    print(f"  Folders: {unique_folders}")
    print(f"  Total PDF size: {total_size_mb:.1f} MB")


if __name__ == "__main__":
    main()
