"""
OCR Evaluation with MAX serve - Compare PDF OCR layer vs Qwen OCR vs Ground Truth

This script:
1. Extracts existing OCR layer from PDFs (using pdfplumber/PyMuPDF)
2. Runs Qwen VLM OCR on the document images
3. Compares both against ground truth
4. Calculates CER for each method

Usage:
1. Start MAX server:
   
   max serve --model-path Qwen/Qwen2.5-VL-7B-Instruct --max-length 8192 --devices gpu:0
   
   Or use your start_max_7b.sh script.

2. Run this script:
   python ocr_eval_max.py /path/to/eval_samples/
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

# MAX serve configuration - SINGLE GPU
MAX_URL = "http://localhost:8000/v1/chat/completions"
MAX_CONCURRENT = 8  # Tune based on VRAM (can try 12-16 if no OOM)

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
    Returns list of tuples: (image_path, ground_truth_path, doc_id)
    """
    pairs = []
    
    # Look for ground truth files first
    gt_files = list(directory.glob("*_ground_truth.txt"))
    
    for gt_path in gt_files:
        # Extract document ID (everything before _ground_truth.txt)
        doc_id = gt_path.stem.replace("_ground_truth", "")
        
        # Look for corresponding PDF or TIF
        pdf_path = directory / f"{doc_id}.pdf"
        tif_path = directory / f"{doc_id}.tif"
        
        if pdf_path.exists():
            pairs.append((pdf_path, gt_path, doc_id))
        elif tif_path.exists():
            pairs.append((tif_path, gt_path, doc_id))
        else:
            print(f"Warning: No image file found for {doc_id}")
    
    return sorted(pairs, key=lambda x: x[2])


def load_image(image_path: Path, dpi: int = 150) -> list:
    """Load image(s) from PDF or TIF file."""
    suffix = image_path.suffix.lower()
    
    if suffix == ".pdf":
        images = convert_from_path(str(image_path), dpi=dpi)
    elif suffix in [".tif", ".tiff"]:
        img = Image.open(image_path)
        images = []
        try:
            while True:
                images.append(img.copy())
                img.seek(img.tell() + 1)
        except EOFError:
            pass
        if not images:
            images = [Image.open(image_path)]
    else:
        images = [Image.open(image_path)]
    
    return images


def image_to_base64(image) -> str:
    """Convert PIL image to base64 string."""
    # Only convert if not already RGB (avoid overhead)
    if image.mode not in ("RGB", "L"):  # L is grayscale, works fine
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
    image,
    image_b64: str,  # Pre-encoded base64
    server_idx: int,
) -> tuple:
    """Send a single image to MAX serve for OCR. Returns (text, inference_time, error)."""
    async with semaphore:
        url = MAX_URL
        
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
            # Only measure the actual API call (model inference)
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


async def process_document(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    image_path: Path,
    gt_path: Path,
    doc_id: str,
    doc_idx: int,
    dpi: int = 150,
) -> dict:
    """Process a single document and compare with ground truth."""
    
    # Load ground truth
    ground_truth = load_ground_truth(gt_path)
    gt_normalized = normalize_text(ground_truth)
    
    # Extract PDF OCR layer (if PDF)
    pdf_ocr_text = ""
    pdf_ocr_cer = None
    if image_path.suffix.lower() == ".pdf":
        print(f"  Extracting PDF text layer...")
        pdf_ocr_text = extract_pdf_text_layer(image_path)
        if pdf_ocr_text.strip():
            pdf_ocr_normalized = normalize_text(pdf_ocr_text)
            pdf_ocr_cer = cer(gt_normalized, pdf_ocr_normalized) if gt_normalized else 0.0
            print(f"  PDF OCR layer CER: {pdf_ocr_cer:.2%}")
        else:
            print(f"  PDF has no text layer")
    
    # Load images for Qwen OCR
    try:
        images = load_image(image_path, dpi=dpi)
    except Exception as e:
        return {
            "doc_id": doc_id,
            "num_pages": 0,
            "qwen_ocr_text": "",
            "pdf_ocr_text": pdf_ocr_text,
            "ground_truth": ground_truth,
            "qwen_cer": 1.0,
            "pdf_ocr_cer": pdf_ocr_cer,
            "time": 0,
            "error": f"Failed to load image: {e}"
        }
    
    # Process all pages with Qwen
    print(f"  Running Qwen OCR on {len(images)} page(s)...")
    start_time = time.perf_counter()
    ocr_texts = []
    
    for page_idx, image in enumerate(images):
        server_idx = doc_idx * len(images) + page_idx
        text, elapsed, error = await process_single_image(
            session, semaphore, image, server_idx
        )
        if error:
            print(f"    Page {page_idx + 1}: ERROR - {error}")
        else:
            print(f"    Page {page_idx + 1}: {elapsed:.2f}s")
        ocr_texts.append(text)
    
    total_time = time.perf_counter() - start_time
    
    # Combine all pages
    qwen_ocr_text = "\n".join(ocr_texts)
    
    # Normalize and calculate CER for Qwen
    qwen_normalized = normalize_text(qwen_ocr_text)
    
    if gt_normalized:
        qwen_cer = cer(gt_normalized, qwen_normalized)
    else:
        qwen_cer = 0.0 if not qwen_normalized else 1.0
    
    return {
        "doc_id": doc_id,
        "num_pages": len(images),
        "qwen_ocr_text": qwen_ocr_text,
        "pdf_ocr_text": pdf_ocr_text,
        "ground_truth": ground_truth,
        "qwen_cer": qwen_cer,
        "pdf_ocr_cer": pdf_ocr_cer,
        "time": total_time,
        "error": None
    }


async def process_all_documents(pairs: list, dpi: int = 150) -> tuple:
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
        if image_path.suffix.lower() == ".pdf":
            pdf_texts[doc_id] = extract_pdf_text_layer(image_path)
        else:
            pdf_texts[doc_id] = ""
    timing["pdf_extraction"] = time.perf_counter() - pdf_extract_start
    print(f"  PDF extraction: {timing['pdf_extraction']:.2f}s")
    
    # Load all images and ground truths
    print("Loading images and ground truths...")
    load_start = time.perf_counter()
    all_tasks = []  # (doc_id, page_idx, image_b64)
    doc_info = {}   # doc_id -> {num_pages, gt, pdf_text}
    
    for image_path, gt_path, doc_id in pairs:
        try:
            images = load_image(image_path, dpi=dpi)
            ground_truth = load_ground_truth(gt_path)
            pdf_text = pdf_texts.get(doc_id, "")
            
            doc_info[doc_id] = {
                "num_pages": len(images),
                "ground_truth": ground_truth,
                "pdf_text": pdf_text
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
    print("Encoding images to base64...")
    encode_start = time.perf_counter()
    encoded_tasks = []
    for doc_id, page_idx, image in all_tasks:
        image_b64 = image_to_base64(image)
        encoded_tasks.append((doc_id, page_idx, image_b64))
    timing["base64_encoding"] = time.perf_counter() - encode_start
    print(f"  Base64 encoding: {timing['base64_encoding']:.2f}s")
    
    print(f"\nProcessing {len(encoded_tasks)} pages from {len(pairs)} documents in parallel...")
    print("(Timing below is MODEL INFERENCE ONLY)")
    
    # Process all pages in parallel - measure only inference time
    inference_start = time.perf_counter()
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = []
        for idx, (doc_id, page_idx, image_b64) in enumerate(encoded_tasks):
            task = process_single_image(session, semaphore, None, image_b64, idx)
            tasks.append((doc_id, page_idx, task))
        
        # Run all tasks
        raw_results = await asyncio.gather(*[t[2] for t in tasks])
    timing["model_inference"] = time.perf_counter() - inference_start
    
    # Reassemble results by document
    doc_pages = {}  # doc_id -> [(page_idx, text, time, error)]
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
            "error": None
        })
        
        # Show comparison
        print(f"  {doc_id}: Qwen CER: {qwen_cer:.2%}", end="")
        if pdf_ocr_cer is not None:
            diff = pdf_ocr_cer - qwen_cer
            better = "✓ Qwen better" if diff > 0 else ("✗ PDF better" if diff < 0 else "= Same")
            print(f" | PDF OCR CER: {pdf_ocr_cer:.2%} | {better}")
        else:
            print()
    
    timing["cer_calculation"] = time.perf_counter() - cer_start
    
    return results, timing


def main(eval_dir: str, dpi: int = 150):
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
    
    # Process all documents
    print(f"\nProcessing with MAX serve (max {MAX_CONCURRENT} concurrent)...")
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
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*70}")
    print(f"Total documents: {len(results)}")
    print(f"Successful: {len(successful)}")
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
    
    # MODEL INFERENCE STATISTICS (what you care about)
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
    output_dir = eval_path / "results_Qw7b"
    output_dir.mkdir(exist_ok=True)
    
    # Save CER results to CSV
    csv_path = output_dir / "cer_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["doc_id", "num_pages", "qwen_cer", "pdf_ocr_cer", "improvement", "inference_time_sec", "error"])
        for r in results:
            improvement = ""
            if r["pdf_ocr_cer"] is not None and r["error"] is None:
                improvement = f"{r['pdf_ocr_cer'] - r['qwen_cer']:.4f}"
            writer.writerow([
                r["doc_id"],
                r["num_pages"],
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
        
        # Save Qwen OCR (normalized) - NEW
        qwen_norm_path = ocr_output_dir / f"{r['doc_id']}_qwen_ocr_norm.txt"
        with open(qwen_norm_path, "w", encoding="utf-8") as f:
            f.write(normalize_text(r["qwen_ocr_text"]))
        
        # Save PDF OCR if available
        if r["pdf_ocr_text"]:
            pdf_path = ocr_output_dir / f"{r['doc_id']}_pdf_ocr.txt"
            with open(pdf_path, "w", encoding="utf-8") as f:
                f.write(r["pdf_ocr_text"])
            
            # Save PDF OCR (normalized) - NEW
            pdf_norm_path = ocr_output_dir / f"{r['doc_id']}_pdf_ocr_norm.txt"
            with open(pdf_norm_path, "w", encoding="utf-8") as f:
                f.write(normalize_text(r["pdf_ocr_text"]))
        
        # Save ground truth (raw)
        gt_path = ocr_output_dir / f"{r['doc_id']}_ground_truth.txt"
        with open(gt_path, "w", encoding="utf-8") as f:
            f.write(r["ground_truth"])
        
        # Save ground truth (normalized) - NEW
        gt_norm_path = ocr_output_dir / f"{r['doc_id']}_ground_truth_norm.txt"
        with open(gt_norm_path, "w", encoding="utf-8") as f:
            f.write(normalize_text(r["ground_truth"]))
        
        print(f"OCR outputs saved to {ocr_output_dir}")
        
    # Save summary
    summary_path = output_dir / "summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"OCR Evaluation Summary\n")
        f.write(f"{'='*70}\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Directory: {eval_dir}\n")
        f.write(f"Total documents: {len(results)}\n")
        f.write(f"Successful: {len(successful)}\n")
        f.write(f"Failed: {len(failed)}\n")
        f.write(f"Total pages: {total_pages}\n\n")
        
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
        print("Usage: python ocr_eval_max.py <eval_directory> [dpi]")
        print("\nExpected directory structure:")
        print("  <doc_id>.pdf or <doc_id>.tif")
        print("  <doc_id>_ground_truth.txt")
        print("\nMake sure MAX server is running first!")
        print("\nStart server with:")
        print("  max serve --model-path Qwen/Qwen2.5-VL-7B-Instruct --max-length 8192 --devices gpu:0")
        print("\nOr use your start_max_7b.sh script.")
        sys.exit(1)
    
    eval_dir = sys.argv[1]
    dpi = int(sys.argv[2]) if len(sys.argv) > 2 else 150
    
    results = main(eval_dir, dpi)