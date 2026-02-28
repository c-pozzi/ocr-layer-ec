"""
POC Shared Utilities

Common functions for the classification POC pipeline:
- PDF processing (page rendering to base64)
- vLLM client (async classify requests)
- JSON response parsing
- Classification flattening and complexity routing
- Checkpoint/resume support
- Progress logging
"""

import asyncio
import aiohttp
import base64
import json
import time
import sys
from pathlib import Path

import fitz  # PyMuPDF

# Allow importing prompts.py from parent directory
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))

from prompts import get_classify_prompt

# =============================================================================
# CONSTANTS
# =============================================================================

CLASSIFICATION_FIELDS = [
    "multi_column", "has_tables", "handwritten", "has_stamps",
    "poor_quality", "has_strikethrough", "latin_script",
    "has_footnotes", "has_forms",
]

VLLM_SERVERS = [f"http://localhost:{8000 + i}" for i in range(4)]
VLLM_ENDPOINT = "/v1/chat/completions"

MODEL_CONFIGS = {
    "bf16": "Qwen/Qwen2.5-VL-7B-Instruct",
    "awq": "Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
}

RESULTS_DIR = SCRIPT_DIR / "results"


def make_run_dir(model_key: str, input_path: str | Path) -> Path:
    """
    Build per-run results directory: results/<model_key>__<folder_name>

    Args:
        model_key: Key from MODEL_CONFIGS (e.g. "awq", "bf16")
        input_path: Path to input folder (basename is used)

    Returns:
        Path to the run directory (not yet created)
    """
    folder_name = Path(input_path).name
    return RESULTS_DIR / f"{model_key}__{folder_name}"

CLASSIFY_PROMPT = get_classify_prompt()

# =============================================================================
# PDF PROCESSING
# =============================================================================

def get_page_count(pdf_path: Path) -> int:
    """Get page count without rendering (fast)."""
    doc = fitz.open(pdf_path)
    count = len(doc)
    doc.close()
    return count


def render_page_to_base64(pdf_path: Path, page_num: int, dpi: int = 150) -> str:
    """
    Render a single PDF page to base64-encoded PNG.

    Args:
        pdf_path: Path to PDF file
        page_num: 0-indexed page number
        dpi: Resolution for rendering

    Returns:
        Base64-encoded PNG string
    """
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    img_bytes = pix.tobytes("png")
    doc.close()
    return base64.standard_b64encode(img_bytes).decode("utf-8")


# =============================================================================
# VLLM CLIENT
# =============================================================================

async def check_server_health(servers: list[str], timeout: float = 10.0) -> list[str]:
    """
    Ping /health on each server. Returns list of healthy server URLs.
    Raises RuntimeError if no servers are healthy.
    """
    healthy = []
    async with aiohttp.ClientSession() as session:
        for url in servers:
            try:
                async with session.get(
                    f"{url}/health",
                    timeout=aiohttp.ClientTimeout(total=timeout),
                ) as resp:
                    if resp.status == 200:
                        healthy.append(url)
                    else:
                        print(f"  WARN: {url} returned status {resp.status}")
            except Exception as e:
                print(f"  WARN: {url} unreachable: {e}")
    return healthy


async def classify_image_async(
    session: aiohttp.ClientSession,
    server_url: str,
    image_b64: str,
    model_name: str = MODEL_CONFIGS["awq"],
    request_timeout: float = 60.0,
) -> tuple[dict, float]:
    """
    Send a single image to vLLM for classification.

    Returns:
        (parsed_result_dict, inference_time_sec)
    """
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                    {"type": "text", "text": CLASSIFY_PROMPT},
                ],
            }
        ],
        "max_tokens": 256,
        "temperature": 0.0,
    }

    t0 = time.monotonic()
    try:
        async with session.post(
            f"{server_url}{VLLM_ENDPOINT}",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=request_timeout),
        ) as response:
            elapsed = time.monotonic() - t0
            if response.status != 200:
                error_text = await response.text()
                return {"error": f"HTTP {response.status}: {error_text}"}, elapsed

            result = await response.json()
            content = result["choices"][0]["message"]["content"]
            parsed = parse_classification_json(content)
            return parsed, elapsed

    except asyncio.TimeoutError:
        return {"error": "Request timeout"}, time.monotonic() - t0
    except Exception as e:
        return {"error": str(e)}, time.monotonic() - t0


# =============================================================================
# JSON PARSING
# =============================================================================

def parse_classification_json(content: str) -> dict:
    """
    Parse classification JSON from model response.
    Handles markdown code blocks wrapping.
    """
    raw = content
    try:
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        return json.loads(content.strip())
    except json.JSONDecodeError as e:
        return {"error": f"JSON parse error: {e}", "raw_response": raw}


# =============================================================================
# CLASSIFICATION HELPERS
# =============================================================================

def flatten_classification(classification: dict) -> dict:
    """Flatten classification dict for CSV output, with error handling."""
    if "error" in classification:
        result = {f: None for f in CLASSIFICATION_FIELDS}
        result["is_complex"] = None
        result["error"] = classification.get("error")
        result["raw_response"] = classification.get("raw_response", "")
        return result

    result = {f: classification.get(f) for f in CLASSIFICATION_FIELDS}
    result["is_complex"] = is_complex(classification)
    result["error"] = None
    result["raw_response"] = ""
    return result


def is_complex(cls: dict) -> bool:
    """
    Determine if a page is complex (Tier 2) based on classification.

    Complex if any of:
    - has_tables
    - handwritten
    - poor_quality
    - multi_column AND has_tables together
    """
    return (
        any(cls.get(f, False) for f in ["has_tables", "handwritten", "poor_quality"])
        or (cls.get("multi_column", False) and cls.get("has_tables", False))
    )


# =============================================================================
# CHECKPOINT / RESUME
# =============================================================================

def load_checkpoint(checkpoint_path: Path) -> set:
    """Load set of completed page IDs from checkpoint file."""
    completed = set()
    if checkpoint_path.exists():
        with open(checkpoint_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    completed.add(line)
    return completed


def save_checkpoint_entry(checkpoint_path: Path, page_id: str):
    """Append a completed page ID to checkpoint file."""
    with open(checkpoint_path, "a") as f:
        f.write(page_id + "\n")


def make_page_id(pdf_filename: str, page_num: int) -> str:
    """Create unique page identifier for checkpoint tracking."""
    return f"{pdf_filename}::{page_num}"


# =============================================================================
# PROGRESS LOGGING
# =============================================================================

class ProgressTracker:
    """Track and log classification progress."""

    def __init__(self, total: int, log_interval: int = 100):
        self.total = total
        self.log_interval = log_interval
        self.completed = 0
        self.errors = 0
        self.start_time = time.monotonic()
        self._lock = asyncio.Lock()

    async def update(self, error: bool = False, queue_depth: int = 0, queue_max: int = 0):
        async with self._lock:
            self.completed += 1
            if error:
                self.errors += 1

            if self.completed % self.log_interval == 0 or self.completed == self.total:
                elapsed = time.monotonic() - self.start_time
                rate = self.completed / elapsed if elapsed > 0 else 0
                remaining = (self.total - self.completed) / rate / 60 if rate > 0 else 0
                pct = 100.0 * self.completed / self.total
                print(
                    f"Progress: {self.completed}/{self.total} ({pct:.1f}%) | "
                    f"{rate:.1f} pages/s | ETA: {remaining:.1f}min | "
                    f"Errors: {self.errors} | Queue: {queue_depth}/{queue_max}"
                )

    def summary(self) -> dict:
        elapsed = time.monotonic() - self.start_time
        rate = self.completed / elapsed if elapsed > 0 else 0
        return {
            "total_completed": self.completed,
            "total_errors": self.errors,
            "wall_time_sec": elapsed,
            "pages_per_sec": rate,
        }
