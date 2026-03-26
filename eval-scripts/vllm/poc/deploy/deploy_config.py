#!/usr/bin/env python3
"""
Deployment pipeline configuration.
"""

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
import sys

# Allow imports from sibling packages
SCRIPT_DIR = Path(__file__).resolve().parent
POC_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(POC_DIR))
sys.path.insert(0, str(POC_DIR / "benchmark"))

from benchmark_config import BenchmarkConfig, MODEL_7B, MODEL_32B, P4D_COST_PER_HOUR  # noqa: E402


@dataclass
class DeployConfig:
    """Configuration for the daily OCR deployment pipeline."""

    # S3 locations
    s3_input_uri: str = "s3://ocr-pipeline-bucket/input/"
    s3_output_uri: str = "s3://ocr-pipeline-bucket/output/"
    s3_processed_uri: str = "s3://ocr-pipeline-bucket/processed/"
    s3_logs_uri: str = "s3://ocr-pipeline-bucket/logs/"

    # Local paths (NVMe for speed)
    local_staging: Path = Path("/opt/dlami/nvme/pipeline_staging")
    local_output: Path = Path("/opt/dlami/nvme/pipeline_output")
    ocr_text_dir: Path = Path("/opt/dlami/nvme/pipeline_ocr_text")
    log_dir: Path = Path("/opt/dlami/nvme/pipeline_logs")

    # Pipeline behaviour
    dpi: int = 150
    shutdown_on_complete: bool = True
    date_partition: str = ""

    # Classification (7B AWQ, TP=1 → 8 servers)
    classify_concurrency: int = 8  # per server
    classify_profile: str = "lite"

    # Complex OCR (32B AWQ, TP=2 → 4 servers)
    ocr_concurrency: int = 4  # per server
    ocr_max_tokens: int = 8192

    def __post_init__(self):
        if not self.date_partition:
            self.date_partition = date.today().isoformat()

    @property
    def s3_output_dated(self) -> str:
        return f"{self.s3_output_uri.rstrip('/')}/{self.date_partition}/"

    @property
    def s3_processed_dated(self) -> str:
        return f"{self.s3_processed_uri.rstrip('/')}/{self.date_partition}/"

    @property
    def s3_logs_dated(self) -> str:
        return f"{self.s3_logs_uri.rstrip('/')}/{self.date_partition}/"

    def classify_server_config(self) -> BenchmarkConfig:
        """BenchmarkConfig for 7B classification servers."""
        return BenchmarkConfig(
            config_id="deploy_7b_classify",
            model=MODEL_7B,
            tp=1,
            cuda_graphs=True,
            gpu_mem_util=0.95,
            max_num_seqs=128,
            max_model_len=8192,
            concurrency_per_server=self.classify_concurrency,
        )

    def ocr_server_config(self) -> BenchmarkConfig:
        """BenchmarkConfig for 32B complex OCR servers."""
        return BenchmarkConfig(
            config_id="deploy_32b_ocr",
            model=MODEL_32B,
            tp=2,
            cuda_graphs=True,
            gpu_mem_util=0.95,
            max_num_seqs=64,
            max_model_len=16384,
            concurrency_per_server=self.ocr_concurrency,
        )

    def ensure_dirs(self):
        """Create local directories."""
        for d in (self.local_staging, self.local_output, self.ocr_text_dir, self.log_dir):
            d.mkdir(parents=True, exist_ok=True)
