"""
OCR Evaluation with vLLM - FULL RESOLUTION TIF TEST

This script tests TIF files at their native resolution (no downscaling).
Requires larger GPU memory and vLLM context.

Usage:
1. Start vLLM servers with LARGER CONTEXT (one per GPU):
   
   CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen2.5-VL-7B-Instruct --host 0.0.0.0 --port 8000 --dtype bfloat16 --max-model-len 32768 &
   CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen2.5-VL-7B-Instruct --host 0.0.0.0 --port 8001 --dtype bfloat16 --max-model-len 32768 &
   # ... etc for more GPUs
   
   NOTE: 32K context requires ~40GB+ VRAM per GPU (A100/H100)

2. Run this script:
   python ocr_eval_vllm_fullres.py /path/to/eval_samples/
"""

import asyncio
import aiohttp
import base64
import time
import sys
import csv
import statistics
from pathlib import Path
from io import BytesIO
from pdf2image import convert_from_path
from PIL import Image
import re

# PDF text extraction
try:
    import pdfplumber
    USE_PDFPLUMBER = True
except ImportError:
    USE_PDFPLUMBER = False

try:
    import fitz  # PyMuPDF
    USE_PYMUPDF = True
except ImportError:
    USE_PYMUPDF = False

# Try to import CER calculation library
try:
    from jiwer import cer
    USE_JIWER = True
except ImportError:
    import Levenshtein
    USE_JIWER = False
    def cer(reference, hypothesis):
        """Calculate Character Error Rate using Levenshtein distance."""
        if len(reference) == 0:
            return 1.0 if len(hypothesis) > 0 else 0.0
        distance = Levenshtein.distance(reference, hypothesis)
        return distance / len(reference)


def extract_pdf_text_layer(pdf_path: Path) -> str:
    """Extract embedded text layer from PDF using available library."""
    
    # Try PyMuPDF first (faster)
    if USE_PYMUPDF:
        try:
            doc = fitz.open(str(pdf_path))
            text_parts = []
            for page in doc:
                text_parts.append(page.get_text())
            doc.close()
            return "\n".join(text_parts)
        except Exception as e:
            print(f"  PyMuPDF failed: {e}")
    
    # Fall back to pdfplumber
    if USE_PDFPLUMBER:
        try:
            with pdfplumber.open(str(pdf_path)) as pdf:
                text_parts = []
                for page in pdf.pages:
                    text = page.extract_text() or ""
                    text_parts.append(text)
                return "\n".join(text_parts)
        except Exception as e:
            print(f"  pdfplumber failed: {e}")
    
    return ""

# vLLM server configuration
VLLM_URLS = [
    "http://localhost:8000/v1/chat/completions",
    "http://localhost:8001/v1/chat/completions",
    "http://localhost:8002/v1/chat/completions",
    "http://localhost:8003/v1/chat/completions",
]
MAX_CONCURRENT = 16  # Reduce if memory issues

PROMPT = """Extract all text from this document image,
Do not skip any visible text, even if incomplete or degraded.
For tables, output them in markdown format using | for columns.
For regular text, preserve the original layout.
Do NOT use HTML. Output plain text and markdown only.
Transcribe verbatim in polytonic Greek, preserving all diacritics (
breathings, circumflexes, graves, iota subscripts)."""


def find_document_pairs(directory: Path) -> list:
    """
    Find all PDF/TIF files and their corresponding ground truth files.
    PREFERS TIF over PDF for image input.
    Returns list of tuples: (image_path, ground_truth_path, doc_id)
    """
    pairs = []
    
    # Look for ground truth files first
    gt_files = list(directory.glob("*_ground_truth.txt"))
    
    for gt_path in gt_files:
        # Extract document ID (everything before _ground_truth.txt)
        doc_id = gt_path.stem.replace("_ground_truth", "")
        
        # Look for corresponding TIF or PDF - PREFER TIF
        pdf_path = directory / f"{doc_id}.pdf"
        tif_path = directory / f"{doc_id}.tif"
        
        if tif_path.exists():
            pairs.append((tif_path, gt_path, doc_id))
        elif pdf_path.exists():
            pairs.append((pdf_path, gt_path, doc_id))
        else:
            print(f"Warning: No image file found for {doc_id}")
    
    return sorted(pairs, key=lambda x: x[2])


def load_image(image_path: Path, dpi: int = 300) -> list:
    """
    Load image(s) from PDF or TIF file.
    
    FULL RESOLUTION MODE:
    - TIF: Loaded at native resolution (NO resizing)
    - PDF: Rendered at specified DPI (default 300 to match TIF)
    """
    suffix = image_path.suffix.lower()
    
    if suffix == ".pdf":
        # Render PDF at 300 DPI to match TIF resolution
        images = convert_from_path(str(image_path), dpi=dpi)
    elif suffix in [".tif", ".tiff"]:
        img = Image.open(image_path)
        images = []
        try:
            while True:
                frame = img.copy()
                # NO RESIZING - use full resolution
                images.append(frame)
                img.seek(img.tell() + 1)
        except EOFError:
            pass
        if not images:
            images = [Image.open(image_path)]
    else:
        images = [Image.open(image_path)]
    
    return images


def image_to_base64(image) -> str:
    """Convert PIL image to base64 string. Always converts to RGB."""
    # ALWAYS convert to RGB for consistency
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def load_ground_truth(gt_path: Path) -> str:
    """Load ground truth text from file."""
    with open(gt_path, "r", encoding="utf-8", errors="replace") as f:
        return f.read().strip()


def normalize_text(text: str) -> str:
    # Remove table formatting
    text = re.sub(r'[|+\-:]{2,}', ' ', text)  # Remove |, +++, ---, :::
    text = re.sub(r'\|', ' ', text)           # Remove remaining pipes
    text = re.sub(r'<br>', ' ', text)         # Remove <br> tags
    # Existing normalizations
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)          # Collapse whitespace
    text = text.strip()
    return text


async def process_single_image(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    image_b64: str,
    server_idx: int,
) -> tuple:
    """Send a single image to vLLM for OCR. Returns (text, inference_time, error)."""
    async with semaphore:
        url = VLLM_URLS[server_idx % len(VLLM_URLS)]
        
        payload = {
            "model": "Qwen/Qwen2.5-VL-7B-Instruct",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_b64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": PROMPT
                        }
                    ]
                }
            ],
            "max_tokens": 8000,
            "temperature": 0,
        }
        
        try:
            start = time.perf_counter()
            async with session.post(url, json=payload) as response:
                result = await response.json()
            inference_time = time.perf_counter() - start
            
            if "choices" in result:
                text = result["choices"][0]["message"]["content"]
                return text, inference_time, None
            else:
                error_msg = result.get("error", {}).get("message", "Unknown error")
                return "", inference_time, error_msg
        except Exception as e:
            return "", 0.0, str(e)


async def process_all_documents(pairs: list, dpi: int = 300) -> tuple:
    """Process all documents in parallel. Returns (results, timing_breakdown)."""
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    timeout = aiohttp.ClientTimeout(total=600)
    
    timing = {
        "pdf_extraction": 0.0,
        "image_loading": 0.0,
        "base64_encoding": 0.0,
        "model_inference": 0.0,
        "cer_calculation": 0.0,
    }
    
    # First, extract PDF text layers (sequential, fast)
    print("Extracting PDF text layers...")
    pdf_extract_start = time.perf_counter()
    pdf_texts = {}
    for image_path, gt_path, doc_id in pairs:
        pdf_path = image_path.with_suffix(".pdf")
        if pdf_path.exists():
            pdf_texts[doc_id] = extract_pdf_text_layer(pdf_path)
            print(f"  {doc_id}: extracted from PDF")
        else:
            pdf_texts[doc_id] = ""
    timing["pdf_extraction"] = time.perf_counter() - pdf_extract_start
    print(f"  PDF extraction: {timing['pdf_extraction']:.2f}s")
    
    # Load all images and ground truths
    print("\nLoading images at FULL RESOLUTION...")
    load_start = time.perf_counter()
    all_tasks = []
    doc_info = {}
    
    for image_path, gt_path, doc_id in pairs:
        try:
            images = load_image(image_path, dpi=dpi)
            ground_truth = load_ground_truth(gt_path)
            pdf_text = pdf_texts.get(doc_id, "")
            
            # Log image dimensions
            if images:
                first_img = images[0]
                print(f"  {doc_id}: {len(images)} page(s) from {image_path.suffix}, size: {first_img.size} ({first_img.size[0] * first_img.size[1] / 1e6:.1f} MP)")
            
            doc_info[doc_id] = {
                "num_pages": len(images),
                "ground_truth": ground_truth,
                "pdf_text": pdf_text,
                "image_source": image_path.suffix,
                "image_size": images[0].size if images else (0, 0)
            }
            
            for page_idx, image in enumerate(images):
                all_tasks.append((doc_id, page_idx, image))
                
        except Exception as e:
            print(f"  Error loading {doc_id}: {e}")
            doc_info[doc_id] = {
                "num_pages": 0,
                "ground_truth": "",
                "pdf_text": "",
                "error": str(e)
            }
    timing["image_loading"] = time.perf_counter() - load_start
    print(f"  Image loading: {timing['image_loading']:.2f}s")
    
    # Pre-encode all images to base64
    print("\nEncoding images to base64 (RGB conversion)...")
    encode_start = time.perf_counter()
    encoded_tasks = []
    for doc_id, page_idx, image in all_tasks:
        image_b64 = image_to_base64(image)
        encoded_tasks.append((doc_id, page_idx, image_b64))
        # Log base64 size
        b64_mb = len(image_b64) / (1024 * 1024)
        print(f"  {doc_id} page {page_idx + 1}: {b64_mb:.1f} MB base64")
    timing["base64_encoding"] = time.perf_counter() - encode_start
    print(f"  Base64 encoding: {timing['base64_encoding']:.2f}s")
    
    print(f"\nProcessing {len(encoded_tasks)} pages from {len(pairs)} documents...")
    print("=" * 70)
    
    # Process all pages in parallel
    inference_start = time.perf_counter()
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = []
        for idx, (doc_id, page_idx, image_b64) in enumerate(encoded_tasks):
            task = process_single_image(session, semaphore, image_b64, idx)
            tasks.append((doc_id, page_idx, task))
        
        raw_results = await asyncio.gather(*[t[2] for t in tasks])
    timing["model_inference"] = time.perf_counter() - inference_start
    
    # Reassemble results by document
    doc_pages = {}
    total_inference_time = 0.0
    for (doc_id, page_idx, _), (text, inference_time, error) in zip(tasks, raw_results):
        if doc_id not in doc_pages:
            doc_pages[doc_id] = []
        doc_pages[doc_id].append((page_idx, text, inference_time, error))
        total_inference_time += inference_time
        if error:
            print(f"  {doc_id} page {page_idx + 1}: ERROR - {error}")
        else:
            print(f"  {doc_id} page {page_idx + 1}: {inference_time:.2f}s (inference)")
    
    # Calculate CER for each document
    print("\nCalculating CER...")
    cer_start = time.perf_counter()
    results = []
    for image_path, gt_path, doc_id in pairs:
        info = doc_info[doc_id]
        
        if "error" in info:
            results.append({
                "doc_id": doc_id,
                "num_pages": 0,
                "qwen_ocr_text": "",
                "pdf_ocr_text": "",
                "ground_truth": "",
                "qwen_cer": 1.0,
                "pdf_ocr_cer": None,
                "inference_time": 0,
                "image_source": info.get("image_source", ""),
                "image_size": (0, 0),
                "error": info["error"]
            })
            continue
        
        # Sort pages and combine text
        pages = sorted(doc_pages.get(doc_id, []), key=lambda x: x[0])
        qwen_ocr_text = "\n".join([p[1] for p in pages])
        doc_inference_time = sum([p[2] for p in pages])
        
        # Calculate CERs
        gt_normalized = normalize_text(info["ground_truth"])
        qwen_normalized = normalize_text(qwen_ocr_text)
        
        qwen_cer = cer(gt_normalized, qwen_normalized) if gt_normalized else (0.0 if not qwen_normalized else 1.0)
        
        pdf_ocr_cer = None
        if info["pdf_text"].strip():
            pdf_normalized = normalize_text(info["pdf_text"])
            pdf_ocr_cer = cer(gt_normalized, pdf_normalized) if gt_normalized else 0.0
        
        results.append({
            "doc_id": doc_id,
            "num_pages": info["num_pages"],
            "qwen_ocr_text": qwen_ocr_text,
            "pdf_ocr_text": info["pdf_text"],
            "ground_truth": info["ground_truth"],
            "qwen_cer": qwen_cer,
            "pdf_ocr_cer": pdf_ocr_cer,
            "inference_time": doc_inference_time,
            "image_source": info.get("image_source", ""),
            "image_size": info.get("image_size", (0, 0)),
            "error": None
        })
        
        # Show comparison
        size_str = f"{info.get('image_size', (0,0))[0]}×{info.get('image_size', (0,0))[1]}"
        print(f"  {doc_id} ({size_str}): Qwen CER: {qwen_cer:.2%}", end="")
        if pdf_ocr_cer is not None:
            diff = pdf_ocr_cer - qwen_cer
            better = "✓ Qwen better" if diff > 0 else ("✗ PDF better" if diff < 0 else "= Same")
            print(f" | PDF OCR CER: {pdf_ocr_cer:.2%} | {better}")
        else:
            print()
    
    timing["cer_calculation"] = time.perf_counter() - cer_start
    
    return results, timing


def main(eval_dir: str, dpi: int = 300):
    eval_path = Path(eval_dir)
    
    if not eval_path.exists():
        print(f"Error: Directory {eval_dir} does not exist")
        sys.exit(1)
    
    # Check for PDF extraction libraries
    if not USE_PYMUPDF and not USE_PDFPLUMBER:
        print("Warning: Neither PyMuPDF nor pdfplumber installed.")
        print("Install with: pip install pymupdf pdfplumber")
        print("PDF text layer extraction will be skipped.\n")
    
    # Find document pairs
    print(f"Scanning {eval_dir} for documents...")
    pairs = find_document_pairs(eval_path)
    
    if not pairs:
        print("No document pairs found!")
        print("Expected structure: <doc_id>.pdf/.tif + <doc_id>_ground_truth.txt")
        sys.exit(1)
    
    print(f"Found {len(pairs)} documents with ground truth")
    print(f"\n*** FULL RESOLUTION MODE ***")
    print(f"TIF files will be processed at native resolution (no downscaling)")
    print(f"PDF render DPI: {dpi}")
    print(f"Make sure vLLM is running with --max-model-len 32768 or higher!\n")
    
    for img_path, _, doc_id in pairs:
        print(f"  {doc_id}: using {img_path.suffix}")
    
    # Process all documents
    print(f"\nProcessing with vLLM (max {MAX_CONCURRENT} concurrent)...")
    start_time = time.perf_counter()
    
    results, timing = asyncio.run(process_all_documents(pairs, dpi))
    
    total_time = time.perf_counter() - start_time
    
    # Calculate statistics
    successful = [r for r in results if r["error"] is None]
    failed = [r for r in results if r["error"] is not None]
    
    qwen_cer_values = [r["qwen_cer"] for r in successful]
    pdf_cer_values = [r["pdf_ocr_cer"] for r in successful if r["pdf_ocr_cer"] is not None]
    inference_times = [r["inference_time"] for r in successful]
    total_pages = sum(r['num_pages'] for r in successful)
    
    # Count image sources
    tif_count = sum(1 for r in successful if r.get("image_source") == ".tif")
    pdf_count = sum(1 for r in successful if r.get("image_source") == ".pdf")
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"EVALUATION RESULTS (FULL RESOLUTION)")
    print(f"{'='*70}")
    print(f"Total documents: {len(results)}")
    print(f"Successful: {len(successful)} (TIF: {tif_count}, PDF: {pdf_count})")
    print(f"Failed: {len(failed)}")
    print(f"Total pages: {total_pages}")
    print(f"{'='*70}")
    
    # TIMING BREAKDOWN
    print(f"\nTIMING BREAKDOWN")
    print(f"{'='*70}")
    print(f"{'Phase':<25} {'Time':>10} {'% of Total':>12}")
    print(f"{'-'*70}")
    print(f"{'PDF text extraction':<25} {timing['pdf_extraction']:>9.2f}s {timing['pdf_extraction']/total_time*100:>11.1f}%")
    print(f"{'Image loading':<25} {timing['image_loading']:>9.2f}s {timing['image_loading']/total_time*100:>11.1f}%")
    print(f"{'Base64 encoding':<25} {timing['base64_encoding']:>9.2f}s {timing['base64_encoding']/total_time*100:>11.1f}%")
    print(f"{'MODEL INFERENCE':<25} {timing['model_inference']:>9.2f}s {timing['model_inference']/total_time*100:>11.1f}%")
    print(f"{'CER calculation':<25} {timing['cer_calculation']:>9.2f}s {timing['cer_calculation']/total_time*100:>11.1f}%")
    print(f"{'-'*70}")
    print(f"{'TOTAL':<25} {total_time:>9.2f}s")
    print(f"{'='*70}")
    
    # MODEL INFERENCE STATISTICS
    if inference_times:
        total_inference = sum(inference_times)
        print(f"\nMODEL INFERENCE STATISTICS")
        print(f"{'='*70}")
        print(f"Total inference time: {total_inference:.2f}s")
        print(f"Avg per page: {total_inference / total_pages:.2f}s")
        print(f"Throughput: {total_pages / timing['model_inference']:.2f} pages/sec (wall clock)")
        print(f"{'='*70}")
    
    # CER Comparison
    print(f"\nCHARACTER ERROR RATE (CER) COMPARISON")
    print(f"{'='*70}")
    print(f"{'Method':<20} {'Mean':>10} {'Median':>10} {'Min':>10} {'Max':>10}")
    print(f"{'-'*70}")
    
    if qwen_cer_values:
        print(f"{'Qwen VLM OCR':<20} {statistics.mean(qwen_cer_values):>9.2%} {statistics.median(qwen_cer_values):>9.2%} {min(qwen_cer_values):>9.2%} {max(qwen_cer_values):>9.2%}")
    
    if pdf_cer_values:
        print(f"{'PDF OCR Layer':<20} {statistics.mean(pdf_cer_values):>9.2%} {statistics.median(pdf_cer_values):>9.2%} {min(pdf_cer_values):>9.2%} {max(pdf_cer_values):>9.2%}")
    
    print(f"{'='*70}")
    
    # Show improvement
    if qwen_cer_values and pdf_cer_values:
        docs_with_both = [r for r in successful if r["pdf_ocr_cer"] is not None]
        qwen_better = sum(1 for r in docs_with_both if r["qwen_cer"] < r["pdf_ocr_cer"])
        pdf_better = sum(1 for r in docs_with_both if r["pdf_ocr_cer"] < r["qwen_cer"])
        same = sum(1 for r in docs_with_both if r["qwen_cer"] == r["pdf_ocr_cer"])
        
        avg_improvement = statistics.mean([r["pdf_ocr_cer"] - r["qwen_cer"] for r in docs_with_both])
        
        print(f"\nCOMPARISON SUMMARY ({len(docs_with_both)} documents with PDF text layer)")
        print(f"{'='*70}")
        print(f"Qwen better:     {qwen_better:>3} documents")
        print(f"PDF OCR better:  {pdf_better:>3} documents")
        print(f"Same:            {same:>3} documents")
        print(f"Avg improvement: {avg_improvement:>+.2%} (positive = Qwen better)")
        print(f"{'='*70}")
    
    # Save detailed results
    output_dir = eval_path / "results_fullres"
    output_dir.mkdir(exist_ok=True)
    
    # Save CER results to CSV
    csv_path = output_dir / "cer_results_fullres.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["doc_id", "num_pages", "image_source", "image_width", "image_height", "qwen_cer", "pdf_ocr_cer", "improvement", "inference_time_sec", "error"])
        for r in results:
            improvement = ""
            if r["pdf_ocr_cer"] is not None and r["error"] is None:
                improvement = f"{r['pdf_ocr_cer'] - r['qwen_cer']:.4f}"
            writer.writerow([
                r["doc_id"],
                r["num_pages"],
                r.get("image_source", ""),
                r.get("image_size", (0, 0))[0],
                r.get("image_size", (0, 0))[1],
                f"{r['qwen_cer']:.4f}",
                f"{r['pdf_ocr_cer']:.4f}" if r["pdf_ocr_cer"] is not None else "",
                improvement,
                f"{r['inference_time']:.2f}",
                r["error"] or ""
            ])
    print(f"\nCER results saved to {csv_path}")
    
    # Save OCR outputs
    ocr_output_dir = output_dir / "ocr_outputs"
    ocr_output_dir.mkdir(exist_ok=True)
    
    for r in successful:
        # Save Qwen OCR (raw)
        qwen_path = ocr_output_dir / f"{r['doc_id']}_qwen_ocr.txt"
        with open(qwen_path, "w", encoding="utf-8") as f:
            f.write(r["qwen_ocr_text"])
        
        # Save Qwen OCR (normalized)
        qwen_norm_path = ocr_output_dir / f"{r['doc_id']}_qwen_ocr_norm.txt"
        with open(qwen_norm_path, "w", encoding="utf-8") as f:
            f.write(normalize_text(r["qwen_ocr_text"]))
        
        # Save PDF OCR if available
        if r["pdf_ocr_text"]:
            pdf_path = ocr_output_dir / f"{r['doc_id']}_pdf_ocr.txt"
            with open(pdf_path, "w", encoding="utf-8") as f:
                f.write(r["pdf_ocr_text"])
            
            pdf_norm_path = ocr_output_dir / f"{r['doc_id']}_pdf_ocr_norm.txt"
            with open(pdf_norm_path, "w", encoding="utf-8") as f:
                f.write(normalize_text(r["pdf_ocr_text"]))
        
        # Save ground truth
        gt_path = ocr_output_dir / f"{r['doc_id']}_ground_truth.txt"
        with open(gt_path, "w", encoding="utf-8") as f:
            f.write(r["ground_truth"])
        
        gt_norm_path = ocr_output_dir / f"{r['doc_id']}_ground_truth_norm.txt"
        with open(gt_norm_path, "w", encoding="utf-8") as f:
            f.write(normalize_text(r["ground_truth"]))
    
    print(f"OCR outputs saved to {ocr_output_dir}")
        
    # Save summary
    summary_path = output_dir / "summary_fullres.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"OCR Evaluation Summary - FULL RESOLUTION\n")
        f.write(f"{'='*70}\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Directory: {eval_dir}\n")
        f.write(f"Mode: FULL RESOLUTION (no downscaling)\n")
        f.write(f"PDF render DPI: {dpi}\n")
        f.write(f"Total documents: {len(results)}\n")
        f.write(f"Successful: {len(successful)} (TIF: {tif_count}, PDF: {pdf_count})\n")
        f.write(f"Failed: {len(failed)}\n")
        f.write(f"Total pages: {total_pages}\n\n")
        
        # Image size stats
        sizes = [r["image_size"] for r in successful if r.get("image_size")]
        if sizes:
            avg_w = sum(s[0] for s in sizes) / len(sizes)
            avg_h = sum(s[1] for s in sizes) / len(sizes)
            f.write(f"Average image size: {avg_w:.0f} × {avg_h:.0f} px\n\n")
        
        f.write(f"TIMING BREAKDOWN:\n")
        f.write(f"  PDF text extraction: {timing['pdf_extraction']:.2f}s\n")
        f.write(f"  Image loading:       {timing['image_loading']:.2f}s\n")
        f.write(f"  Base64 encoding:     {timing['base64_encoding']:.2f}s\n")
        f.write(f"  MODEL INFERENCE:     {timing['model_inference']:.2f}s\n")
        f.write(f"  CER calculation:     {timing['cer_calculation']:.2f}s\n")
        f.write(f"  TOTAL:               {total_time:.2f}s\n\n")
        
        if inference_times:
            f.write(f"MODEL INFERENCE STATISTICS:\n")
            f.write(f"  Total inference time: {sum(inference_times):.2f}s\n")
            f.write(f"  Avg per page:         {sum(inference_times) / total_pages:.2f}s\n")
            f.write(f"  Throughput:           {total_pages / timing['model_inference']:.2f} pages/sec\n\n")
        
        f.write(f"CER Statistics:\n")
        if qwen_cer_values:
            f.write(f"  Qwen VLM OCR:\n")
            f.write(f"    Mean:   {statistics.mean(qwen_cer_values):.2%}\n")
            f.write(f"    Median: {statistics.median(qwen_cer_values):.2%}\n")
            f.write(f"    Min:    {min(qwen_cer_values):.2%}\n")
            f.write(f"    Max:    {max(qwen_cer_values):.2%}\n\n")
        
        if pdf_cer_values:
            f.write(f"  PDF OCR Layer:\n")
            f.write(f"    Mean:   {statistics.mean(pdf_cer_values):.2%}\n")
            f.write(f"    Median: {statistics.median(pdf_cer_values):.2%}\n")
            f.write(f"    Min:    {min(pdf_cer_values):.2%}\n")
            f.write(f"    Max:    {max(pdf_cer_values):.2%}\n\n")
        
        if qwen_cer_values and pdf_cer_values:
            docs_with_both = [r for r in successful if r["pdf_ocr_cer"] is not None]
            qwen_better = sum(1 for r in docs_with_both if r["qwen_cer"] < r["pdf_ocr_cer"])
            pdf_better = sum(1 for r in docs_with_both if r["pdf_ocr_cer"] < r["qwen_cer"])
            avg_improvement = statistics.mean([r["pdf_ocr_cer"] - r["qwen_cer"] for r in docs_with_both])
            
            f.write(f"Comparison:\n")
            f.write(f"  Qwen better: {qwen_better} documents\n")
            f.write(f"  PDF better:  {pdf_better} documents\n")
            f.write(f"  Avg improvement: {avg_improvement:+.2%}\n\n")
        
        f.write(f"Total time: {total_time:.2f}s\n")
    print(f"Summary saved to {summary_path}")
    
    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ocr_eval_vllm_fullres.py <eval_directory> [pdf_dpi]")
        print("\nFULL RESOLUTION MODE - TIF files processed without downscaling")
        print("\nExpected directory structure:")
        print("  <doc_id>.tif (preferred, native resolution)")
        print("  <doc_id>.pdf (fallback)")
        print("  <doc_id>_ground_truth.txt")
        print("\nMake sure vLLM servers are running with --max-model-len 32768 or higher!")
        sys.exit(1)
    
    eval_dir = sys.argv[1]
    dpi = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    
    results = main(eval_dir, dpi)