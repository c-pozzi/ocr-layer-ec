"""
OCR Evaluation with vLLM - Compare PDF OCR layer vs Qwen OCR vs Ground Truth

This script:
1. Extracts existing OCR layer from PDFs (using pdfplumber/PyMuPDF)
2. Runs Qwen VLM OCR on the document images
3. Compares both against ground truth
4. Calculates CER for each method

Usage:
1. Start vLLM servers (one per GPU):
   
   CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen2.5-VL-7B-Instruct --host 0.0.0.0 --port 8000 --dtype bfloat16 --max-model-len 8192 &
   CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen2.5-VL-7B-Instruct --host 0.0.0.0 --port 8001 --dtype bfloat16 --max-model-len 8192 &
   # ... etc for more GPUs

2. Run this script:
   python ocr_eval_vllm.py /path/to/eval_samples/
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
MAX_CONCURRENT = 8
PROMPT = "Extract all text from this document image. Return only the extracted text, preserving the original layout as much as possible."


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
    # Convert to RGB if necessary
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
    """Normalize text for CER comparison."""
    # Basic normalization - can be adjusted based on needs
    text = text.strip()
    # Normalize whitespace
    text = " ".join(text.split())
    return text


async def process_single_image(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    image,
    server_idx: int,
) -> tuple:
    """Send a single image to vLLM for OCR."""
    async with semaphore:
        start = time.perf_counter()
        
        url = VLLM_URLS[server_idx % len(VLLM_URLS)]
        image_b64 = image_to_base64(image)
        
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
            "max_tokens": 4096,
            "temperature": 0,
        }
        
        try:
            async with session.post(url, json=payload) as response:
                result = await response.json()
                
                if "choices" in result:
                    text = result["choices"][0]["message"]["content"]
                    elapsed = time.perf_counter() - start
                    return text, elapsed, None
                else:
                    error_msg = result.get("error", {}).get("message", "Unknown error")
                    return "", time.perf_counter() - start, error_msg
        except Exception as e:
            return "", time.perf_counter() - start, str(e)


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


async def process_all_documents(pairs: list, dpi: int = 150) -> list:
    """Process all documents."""
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    timeout = aiohttp.ClientTimeout(total=600)  # 10 min timeout
    
    results = []
    
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for idx, (image_path, gt_path, doc_id) in enumerate(pairs):
            print(f"\n[{idx + 1}/{len(pairs)}] Processing {doc_id}...")
            result = await process_document(
                session, semaphore, image_path, gt_path, doc_id, idx, dpi
            )
            results.append(result)
            
            # Show comparison
            print(f"  Qwen CER: {result['qwen_cer']:.2%}", end="")
            if result['pdf_ocr_cer'] is not None:
                diff = result['pdf_ocr_cer'] - result['qwen_cer']
                better = "✓ Qwen better" if diff > 0 else ("✗ PDF better" if diff < 0 else "= Same")
                print(f" | PDF OCR CER: {result['pdf_ocr_cer']:.2%} | {better}")
            else:
                print()
    
    return results


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
    print(f"\nProcessing with vLLM (max {MAX_CONCURRENT} concurrent)...")
    start_time = time.perf_counter()
    
    results = asyncio.run(process_all_documents(pairs, dpi))
    
    total_time = time.perf_counter() - start_time
    
    # Calculate statistics
    successful = [r for r in results if r["error"] is None]
    failed = [r for r in results if r["error"] is not None]
    
    qwen_cer_values = [r["qwen_cer"] for r in successful]
    pdf_cer_values = [r["pdf_ocr_cer"] for r in successful if r["pdf_ocr_cer"] is not None]
    times = [r["time"] for r in successful]
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*70}")
    print(f"Total documents: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"Total time: {total_time:.2f}s")
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
    
    if times:
        print(f"\nTIMING STATISTICS")
        print(f"{'='*70}")
        print(f"Mean time:   {statistics.mean(times):.2f}s per document")
        print(f"Total pages: {sum(r['num_pages'] for r in successful)}")
        print(f"{'='*70}")
    
    # Save detailed results (relative to this script)
    script_dir = Path(__file__).resolve().parent
    output_dir = script_dir / "results"
    output_dir.mkdir(exist_ok=True)
    
    # Save CER results to CSV
    csv_path = output_dir / "cer_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["doc_id", "num_pages", "qwen_cer", "pdf_ocr_cer", "improvement", "time_sec", "error"])
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
                f"{r['time']:.2f}",
                r["error"] or ""
            ])
    print(f"\nCER results saved to {csv_path}")
    
    # Save OCR outputs
    ocr_output_dir = output_dir / "ocr_outputs"
    ocr_output_dir.mkdir(exist_ok=True)
    
    for r in successful:
        # Save Qwen OCR
        qwen_path = ocr_output_dir / f"{r['doc_id']}_qwen_ocr.txt"
        with open(qwen_path, "w", encoding="utf-8") as f:
            f.write(r["qwen_ocr_text"])
        
        # Save PDF OCR if available
        if r["pdf_ocr_text"]:
            pdf_path = ocr_output_dir / f"{r['doc_id']}_pdf_ocr.txt"
            with open(pdf_path, "w", encoding="utf-8") as f:
                f.write(r["pdf_ocr_text"])
    
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
        f.write(f"Failed: {len(failed)}\n\n")
        
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
        print("Usage: python ocr_eval_vllm.py <eval_directory> [dpi]")
        print("\nExpected directory structure:")
        print("  <doc_id>.pdf or <doc_id>.tif")
        print("  <doc_id>_ground_truth.txt")
        print("\nMake sure vLLM servers are running first!")
        sys.exit(1)
    
    eval_dir = sys.argv[1]
    dpi = int(sys.argv[2]) if len(sys.argv) > 2 else 150
    
    results = main(eval_dir, dpi)