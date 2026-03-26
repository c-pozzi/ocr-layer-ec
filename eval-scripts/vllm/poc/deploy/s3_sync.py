#!/usr/bin/env python3
"""
S3 upload and move helpers for the deployment pipeline.

- Upload modified PDFs to the output bucket
- Move processed inputs to the processed/ prefix
- Upload pipeline completion marker
"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

log = logging.getLogger(__name__)


def _get_s3_client():
    import boto3
    return boto3.client("s3")


def _parse_s3_uri(s3_uri: str) -> tuple[str, str]:
    """Parse s3://bucket/prefix into (bucket, prefix)."""
    parts = s3_uri.replace("s3://", "").split("/", 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    return bucket, prefix


def upload_directory(local_dir: Path, s3_uri: str, max_workers: int = 8) -> int:
    """
    Upload all files in local_dir to s3_uri, preserving filenames.

    Returns number of files uploaded.
    """
    s3 = _get_s3_client()
    bucket, prefix = _parse_s3_uri(s3_uri)

    files = [f for f in local_dir.iterdir() if f.is_file()]
    if not files:
        log.info("No files to upload from %s", local_dir)
        return 0

    uploaded = 0

    def upload_one(local_path: Path) -> str:
        key = f"{prefix}{local_path.name}"
        s3.upload_file(str(local_path), bucket, key)
        return key

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(upload_one, f): f for f in files}
        for future in as_completed(futures):
            f = futures[future]
            try:
                key = future.result()
                uploaded += 1
            except Exception as e:
                log.error("Failed to upload %s: %s", f.name, e)

    log.info("Uploaded %d/%d files to %s", uploaded, len(files), s3_uri)
    return uploaded


def move_processed_inputs(
    s3_input_uri: str,
    s3_processed_uri: str,
    pdf_filenames: list[str],
    max_workers: int = 8,
) -> int:
    """
    Move (copy + delete) input PDFs to the processed prefix.

    Args:
        s3_input_uri: Source prefix (e.g., s3://bucket/input/).
        s3_processed_uri: Destination prefix (e.g., s3://bucket/processed/2026-03-26/).
        pdf_filenames: List of PDF filenames to move.

    Returns number of files moved.
    """
    s3 = _get_s3_client()
    src_bucket, src_prefix = _parse_s3_uri(s3_input_uri)
    dst_bucket, dst_prefix = _parse_s3_uri(s3_processed_uri)

    moved = 0

    def move_one(filename: str) -> str:
        src_key = f"{src_prefix}{filename}"
        dst_key = f"{dst_prefix}{filename}"
        # Copy to processed
        s3.copy_object(
            CopySource={"Bucket": src_bucket, "Key": src_key},
            Bucket=dst_bucket,
            Key=dst_key,
        )
        # Delete from input
        s3.delete_object(Bucket=src_bucket, Key=src_key)
        return filename

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(move_one, f): f for f in pdf_filenames}
        for future in as_completed(futures):
            f = futures[future]
            try:
                future.result()
                moved += 1
            except Exception as e:
                log.error("Failed to move %s: %s", f, e)

    log.info("Moved %d/%d files to %s", moved, len(pdf_filenames), s3_processed_uri)
    return moved


def upload_completion_marker(s3_uri: str, metrics: dict) -> None:
    """Upload _pipeline_complete.json to the output prefix."""
    s3 = _get_s3_client()
    bucket, prefix = _parse_s3_uri(s3_uri)
    key = f"{prefix}_pipeline_complete.json"
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(metrics, indent=2, default=str),
        ContentType="application/json",
    )
    log.info("Completion marker uploaded: s3://%s/%s", bucket, key)


def upload_log_file(log_path: Path, s3_uri: str) -> None:
    """Upload a single log file to S3."""
    if not log_path.exists():
        return
    s3 = _get_s3_client()
    bucket, prefix = _parse_s3_uri(s3_uri)
    key = f"{prefix}{log_path.name}"
    s3.upload_file(str(log_path), bucket, key)
    log.info("Log uploaded: s3://%s/%s", bucket, key)
