"""
OCR Pipeline with Classification-Based Prompts

This script:
1. Reads classification results from CSV
2. Builds document-specific prompts based on classification
3. Runs OCR with the tailored prompts
4. Compares against ground truth and calculates CER

Usage:
1. First run classify_prompts.py to generate classification_results_{variant}.csv
2. Start vLLM servers:
   ./start_vllm_servers.sh       (for bf16)
   ./start_vllm_servers_int4.sh  (for int4)
3. Run this script:
   python ocr_pipeline.py [--model {bf16,int4}]
"""

import argparse
import asyncio
import aiohttp
import base64
import time
import csv
import statistics
import re
import json
from pathlib import Path
from io import BytesIO
from PIL import Image
import fitz  # PyMuPDF

from prompts import build_prompt

# =============================================================================
# CLI ARGUMENTS
# =============================================================================

MODEL_CONFIGS = {
    "bf16": "Qwen/Qwen2.5-VL-7B-Instruct",
    "int4": "Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
}

def parse_args():
    parser = argparse.ArgumentParser(description="OCR pipeline with classification-based prompts")
    parser.add_argument(
        "--model", choices=list(MODEL_CONFIGS.keys()), default="bf16",
        help="Model variant to use: bf16 (default) or int4 (AWQ quantized)"
    )
    return parser.parse_args()

# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths (relative to this script)
SCRIPT_DIR = Path(__file__).resolve().parent
SAMPLES_DIR = SCRIPT_DIR.parents[2] / "ocr-evaluation-samples"

# vLLM server configuration
VLLM_SERVERS = [f"http://localhost:{8000 + i}" for i in range(8)]
VLLM_ENDPOINT = "/v1/chat/completions"
MODEL_NAME = MODEL_CONFIGS["bf16"]  # overridden in main() based on --model arg

# Processing
MAX_CONCURRENT = 8
DPI = 150

# =============================================================================
# CER CALCULATION
# =============================================================================

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

# =============================================================================
# TEXT NORMALIZATION
# =============================================================================

def normalize_text(text: str) -> str:
    """Normalize text for CER comparison."""
    # Remove table formatting
    text = re.sub(r'[|+\-:]{2,}', ' ', text)  # Remove |, +++, ---, :::
    text = re.sub(r'\|', ' ', text)           # Remove remaining pipes
    text = re.sub(r'<br>', ' ', text)         # Remove <br> tags
    
    # Remove our annotation tags for CER calculation
    text = re.sub(r'\[\[/?[A-Z0-9:]+\]\]', ' ', text)  # Remove [[TAG]] and [[/TAG]]
    
    # Existing normalizations
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)          # Collapse whitespace
    text = text.strip()
    return text

# =============================================================================
# FILE LOADING
# =============================================================================

def load_classification_results(csv_path: Path) -> dict:
    """
    Load classification results from CSV.
    Returns dict: {filename: {page: classification_dict}}
    """
    results = {}
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row['filename']
            page = int(row['page'])
            
            # Parse boolean fields
            classification = {
                'multi_column': row.get('multi_column', '').lower() == 'true',
                'has_tables': row.get('has_tables', '').lower() == 'true',
                'handwritten': row.get('handwritten', '').lower() == 'true',
                'has_stamps': row.get('has_stamps', '').lower() == 'true',
                'poor_quality': row.get('poor_quality', '').lower() == 'true',
                'has_strikethrough': row.get('has_strikethrough', '').lower() == 'true',
                'latin_script': row.get('latin_script', '').lower() != 'false',  # Default True
                'has_footnotes': row.get('has_footnotes', '').lower() == 'true',
                'has_forms': row.get('has_forms', '').lower() == 'true',
            }
            
            if filename not in results:
                results[filename] = {}
            results[filename][page] = classification
    
    return results


def find_ground_truth(pdf_path: Path) -> Path | None:
    """Find ground truth file for a PDF."""
    doc_id = pdf_path.stem
    gt_path = pdf_path.parent / f"{doc_id}_ground_truth.txt"
    return gt_path if gt_path.exists() else None


def load_ground_truth(gt_path: Path) -> str:
    """Load ground truth text from file."""
    with open(gt_path, "r", encoding="utf-8", errors="replace") as f:
        return f.read().strip()


def pdf_to_base64_images(pdf_path: Path, dpi: int = 150) -> list[tuple[int, str]]:
    """
    Convert PDF pages to base64 encoded images.
    Returns list of (page_number, base64_string) tuples
    """
    images = []
    doc = fitz.open(pdf_path)
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        b64_string = base64.standard_b64encode(img_bytes).decode("utf-8")
        images.append((page_num + 1, b64_string))
    
    doc.close()
    return images


def extract_pdf_text_layer(pdf_path: Path) -> str:
    """Extract embedded text layer from PDF."""
    try:
        doc = fitz.open(str(pdf_path))
        text_parts = []
        for page in doc:
            text_parts.append(page.get_text())
        doc.close()
        return "\n".join(text_parts)
    except Exception as e:
        print(f"  PDF text extraction failed: {e}")
        return ""

# =============================================================================
# VLLM CLIENT
# =============================================================================

async def run_ocr(
    session: aiohttp.ClientSession,
    server_url: str,
    image_b64: str,
    prompt: str
) -> tuple[str, float, str | None]:
    """
    Send image to vLLM server for OCR.
    Returns (text, inference_time, error)
    """
    payload = {
        "model": MODEL_NAME,
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
                        "text": prompt
                    }
                ]
            }
        ],
        "max_tokens": 8000,
        "temperature": 0.0
    }
    
    try:
        start = time.perf_counter()
        async with session.post(
            f"{server_url}{VLLM_ENDPOINT}",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=120)
        ) as response:
            result = await response.json()
        inference_time = time.perf_counter() - start
        
        if "choices" in result:
            text = result["choices"][0]["message"]["content"]
            return text, inference_time, None
        else:
            error_msg = result.get("error", {}).get("message", "Unknown error")
            return "", inference_time, error_msg
            
    except asyncio.TimeoutError:
        return "", 0.0, "Request timeout"
    except Exception as e:
        return "", 0.0, str(e)

# =============================================================================
# MAIN PROCESSING
# =============================================================================

async def process_document(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    pdf_path: Path,
    classifications: dict,
    doc_idx: int
) -> dict:
    """Process a single document with classification-based prompts."""
    
    async with semaphore:
        doc_id = pdf_path.stem
        print(f"\nProcessing: {doc_id}")
        
        # Find ground truth
        gt_path = find_ground_truth(pdf_path)
        ground_truth = load_ground_truth(gt_path) if gt_path else ""
        
        # Extract PDF text layer for comparison
        pdf_ocr_text = extract_pdf_text_layer(pdf_path)
        
        # Load images
        try:
            pages = pdf_to_base64_images(pdf_path, DPI)
        except Exception as e:
            return {
                "doc_id": doc_id,
                "num_pages": 0,
                "classified_ocr_text": "",
                "pdf_ocr_text": pdf_ocr_text,
                "ground_truth": ground_truth,
                "classified_cer": 1.0,
                "pdf_ocr_cer": None,
                "inference_time": 0,
                "prompts_used": [],
                "error": f"Failed to load PDF: {e}"
            }
        
        # Process each page with its specific prompt
        ocr_texts = []
        prompts_used = []
        total_inference_time = 0.0
        
        for page_num, image_b64 in pages:
            # Get classification for this page
            classification = classifications.get(page_num, {})
            
            # Build prompt based on classification
            prompt = build_prompt(classification)
            prompts_used.append({
                "page": page_num,
                "classification": classification,
                "prompt_length": len(prompt)
            })
            
            # Select server
            server_idx = (doc_idx * len(pages) + page_num) % len(VLLM_SERVERS)
            server_url = VLLM_SERVERS[server_idx]
            
            # Run OCR
            text, inference_time, error = await run_ocr(
                session, server_url, image_b64, prompt
            )
            
            total_inference_time += inference_time
            
            if error:
                print(f"  Page {page_num}: ERROR - {error}")
                ocr_texts.append("")
            else:
                # Show which blocks were activated
                active_blocks = [k for k, v in classification.items() if v and k != 'latin_script']
                if not classification.get('latin_script', True):
                    active_blocks.append('non_latin')
                blocks_str = ', '.join(active_blocks) if active_blocks else 'base only'
                print(f"  Page {page_num}: {inference_time:.2f}s [{blocks_str}]")
                ocr_texts.append(text)
        
        # Combine all pages
        classified_ocr_text = "\n".join(ocr_texts)
        
        # Calculate CERs
        gt_normalized = normalize_text(ground_truth)
        classified_normalized = normalize_text(classified_ocr_text)
        
        classified_cer = cer(gt_normalized, classified_normalized) if gt_normalized else 0.0
        
        pdf_ocr_cer = None
        if pdf_ocr_text.strip():
            pdf_normalized = normalize_text(pdf_ocr_text)
            pdf_ocr_cer = cer(gt_normalized, pdf_normalized) if gt_normalized else 0.0
        
        return {
            "doc_id": doc_id,
            "num_pages": len(pages),
            "classified_ocr_text": classified_ocr_text,
            "pdf_ocr_text": pdf_ocr_text,
            "ground_truth": ground_truth,
            "classified_cer": classified_cer,
            "pdf_ocr_cer": pdf_ocr_cer,
            "inference_time": total_inference_time,
            "prompts_used": prompts_used,
            "error": None
        }


async def main():
    """Main entry point."""
    args = parse_args()

    # Set model name based on variant
    global MODEL_NAME
    MODEL_NAME = MODEL_CONFIGS[args.model]
    CLASSIFICATION_CSV = SCRIPT_DIR / "results_classify" / f"classification_results_{args.model}.csv"
    OUTPUT_DIR = SCRIPT_DIR / "results_classify" / f"ocr_outputs_{args.model}"

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load classification results
    print(f"Model variant: {args.model} ({MODEL_NAME})")
    print(f"Loading classifications from {CLASSIFICATION_CSV}")
    if not CLASSIFICATION_CSV.exists():
        print(f"ERROR: Classification CSV not found. Run classify_prompts.py --model {args.model} first.")
        return

    classifications = load_classification_results(CLASSIFICATION_CSV)
    print(f"Loaded classifications for {len(classifications)} files")
    
    # Find PDF files that have classifications
    pdf_files = []
    for filename in classifications.keys():
        pdf_path = SAMPLES_DIR / filename
        if pdf_path.exists():
            pdf_files.append(pdf_path)
        else:
            print(f"Warning: {filename} not found in {SAMPLES_DIR}")
    
    pdf_files = sorted(pdf_files)
    print(f"Found {len(pdf_files)} PDFs to process")
    
    if not pdf_files:
        print("No files to process!")
        return
    
    # Process all documents
    print(f"\nProcessing with classification-based prompts...")
    print(f"Using {len(VLLM_SERVERS)} vLLM servers, max concurrency: {MAX_CONCURRENT}")
    print("=" * 70)
    
    start_time = time.perf_counter()
    
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    timeout = aiohttp.ClientTimeout(total=600)
    
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [
            process_document(
                session, semaphore, pdf_path,
                classifications.get(pdf_path.name, {}),
                idx
            )
            for idx, pdf_path in enumerate(pdf_files)
        ]
        results = await asyncio.gather(*tasks)
    
    total_time = time.perf_counter() - start_time
    
    # Calculate statistics
    successful = [r for r in results if r["error"] is None]
    failed = [r for r in results if r["error"] is not None]
    
    classified_cer_values = [r["classified_cer"] for r in successful]
    pdf_cer_values = [r["pdf_ocr_cer"] for r in successful if r["pdf_ocr_cer"] is not None]
    inference_times = [r["inference_time"] for r in successful]
    total_pages = sum(r['num_pages'] for r in successful)
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"EVALUATION RESULTS (Classification-Based Prompts)")
    print(f"{'='*70}")
    print(f"Total documents: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"Total pages: {total_pages}")
    print(f"Total time: {total_time:.2f}s")
    print(f"{'='*70}")
    
    # CER Comparison
    print(f"\nCHARACTER ERROR RATE (CER)")
    print(f"{'='*70}")
    print(f"{'Method':<25} {'Mean':>10} {'Median':>10} {'Min':>10} {'Max':>10}")
    print(f"{'-'*70}")
    
    if classified_cer_values:
        print(f"{'Classified Prompt OCR':<25} {statistics.mean(classified_cer_values):>9.2%} {statistics.median(classified_cer_values):>9.2%} {min(classified_cer_values):>9.2%} {max(classified_cer_values):>9.2%}")
    
    if pdf_cer_values:
        print(f"{'PDF OCR Layer':<25} {statistics.mean(pdf_cer_values):>9.2%} {statistics.median(pdf_cer_values):>9.2%} {min(pdf_cer_values):>9.2%} {max(pdf_cer_values):>9.2%}")
    
    print(f"{'='*70}")
    
    # Comparison summary
    if classified_cer_values and pdf_cer_values:
        docs_with_both = [r for r in successful if r["pdf_ocr_cer"] is not None]
        classified_better = sum(1 for r in docs_with_both if r["classified_cer"] < r["pdf_ocr_cer"])
        pdf_better = sum(1 for r in docs_with_both if r["pdf_ocr_cer"] < r["classified_cer"])
        same = sum(1 for r in docs_with_both if r["classified_cer"] == r["pdf_ocr_cer"])
        
        avg_improvement = statistics.mean([r["pdf_ocr_cer"] - r["classified_cer"] for r in docs_with_both])
        
        print(f"\nCOMPARISON vs PDF OCR ({len(docs_with_both)} documents)")
        print(f"{'='*70}")
        print(f"Classified better: {classified_better:>3} documents")
        print(f"PDF OCR better:    {pdf_better:>3} documents")
        print(f"Same:              {same:>3} documents")
        print(f"Avg improvement:   {avg_improvement:>+.2%} (positive = Classified better)")
        print(f"{'='*70}")
    
    # Save CER results to CSV
    csv_path = OUTPUT_DIR / "cer_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "doc_id", "num_pages", "classified_cer", "pdf_ocr_cer", 
            "improvement", "inference_time_sec", "error"
        ])
        for r in results:
            improvement = ""
            if r["pdf_ocr_cer"] is not None and r["error"] is None:
                improvement = f"{r['pdf_ocr_cer'] - r['classified_cer']:.4f}"
            writer.writerow([
                r["doc_id"],
                r["num_pages"],
                f"{r['classified_cer']:.4f}",
                f"{r['pdf_ocr_cer']:.4f}" if r["pdf_ocr_cer"] is not None else "",
                improvement,
                f"{r['inference_time']:.2f}",
                r["error"] or ""
            ])
    print(f"\nCER results saved to {csv_path}")
    
    # Save OCR outputs
    for r in successful:
        # Save classified OCR (raw)
        ocr_path = OUTPUT_DIR / f"{r['doc_id']}_classified_ocr.txt"
        with open(ocr_path, "w", encoding="utf-8") as f:
            f.write(r["classified_ocr_text"])
        
        # Save classified OCR (normalized)
        ocr_norm_path = OUTPUT_DIR / f"{r['doc_id']}_classified_ocr_norm.txt"
        with open(ocr_norm_path, "w", encoding="utf-8") as f:
            f.write(normalize_text(r["classified_ocr_text"]))
        
        # Save PDF OCR if available
        if r["pdf_ocr_text"]:
            pdf_path = OUTPUT_DIR / f"{r['doc_id']}_pdf_ocr.txt"
            with open(pdf_path, "w", encoding="utf-8") as f:
                f.write(r["pdf_ocr_text"])
            
            pdf_norm_path = OUTPUT_DIR / f"{r['doc_id']}_pdf_ocr_norm.txt"
            with open(pdf_norm_path, "w", encoding="utf-8") as f:
                f.write(normalize_text(r["pdf_ocr_text"]))
        
        # Save ground truth (raw)
        gt_path = OUTPUT_DIR / f"{r['doc_id']}_ground_truth.txt"
        with open(gt_path, "w", encoding="utf-8") as f:
            f.write(r["ground_truth"])
        
        # Save ground truth (normalized)
        gt_norm_path = OUTPUT_DIR / f"{r['doc_id']}_ground_truth_norm.txt"
        with open(gt_norm_path, "w", encoding="utf-8") as f:
            f.write(normalize_text(r["ground_truth"]))
        
        # Save prompts info (for debugging)
        prompts_path = OUTPUT_DIR / f"{r['doc_id']}_prompts.json"
        with open(prompts_path, "w", encoding="utf-8") as f:
            json.dump(r["prompts_used"], f, indent=2)
    
    print(f"OCR outputs saved to {OUTPUT_DIR}")
    
    # Save summary
    summary_path = OUTPUT_DIR / "summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"OCR Evaluation Summary (Classification-Based Prompts)\n")
        f.write(f"{'='*70}\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Samples directory: {SAMPLES_DIR}\n")
        f.write(f"Classification CSV: {CLASSIFICATION_CSV}\n")
        f.write(f"Output directory: {OUTPUT_DIR}\n\n")
        
        f.write(f"Documents: {len(results)} ({len(successful)} successful, {len(failed)} failed)\n")
        f.write(f"Total pages: {total_pages}\n")
        f.write(f"Total time: {total_time:.2f}s\n")
        if total_pages > 0:
            f.write(f"Avg time per page: {total_time / total_pages:.2f}s\n")
            f.write(f"Throughput: {total_pages / total_time:.2f} pages/sec\n")
        f.write(f"\n")
        
        f.write(f"CER Statistics:\n")
        if classified_cer_values:
            f.write(f"  Classified Prompt OCR:\n")
            f.write(f"    Mean:   {statistics.mean(classified_cer_values):.2%}\n")
            f.write(f"    Median: {statistics.median(classified_cer_values):.2%}\n")
            f.write(f"    Min:    {min(classified_cer_values):.2%}\n")
            f.write(f"    Max:    {max(classified_cer_values):.2%}\n\n")
        
        if pdf_cer_values:
            f.write(f"  PDF OCR Layer:\n")
            f.write(f"    Mean:   {statistics.mean(pdf_cer_values):.2%}\n")
            f.write(f"    Median: {statistics.median(pdf_cer_values):.2%}\n")
            f.write(f"    Min:    {min(pdf_cer_values):.2%}\n")
            f.write(f"    Max:    {max(pdf_cer_values):.2%}\n\n")
        
        if classified_cer_values and pdf_cer_values:
            docs_with_both = [r for r in successful if r["pdf_ocr_cer"] is not None]
            if docs_with_both:
                classified_better = sum(1 for r in docs_with_both if r["classified_cer"] < r["pdf_ocr_cer"])
                pdf_better = sum(1 for r in docs_with_both if r["pdf_ocr_cer"] < r["classified_cer"])
                avg_improvement = statistics.mean([r["pdf_ocr_cer"] - r["classified_cer"] for r in docs_with_both])
                
                f.write(f"Comparison vs PDF OCR:\n")
                f.write(f"  Classified better: {classified_better} documents\n")
                f.write(f"  PDF better:        {pdf_better} documents\n")
                f.write(f"  Avg improvement:   {avg_improvement:+.2%}\n\n")
        
        # Per-document details
        f.write(f"\nPer-Document Results:\n")
        f.write(f"{'-'*70}\n")
        for r in results:
            f.write(f"{r['doc_id']}:\n")
            f.write(f"  Pages: {r['num_pages']}\n")
            f.write(f"  Classified CER: {r['classified_cer']:.2%}\n")
            if r['pdf_ocr_cer'] is not None:
                f.write(f"  PDF OCR CER: {r['pdf_ocr_cer']:.2%}\n")
            f.write(f"  Inference time: {r['inference_time']:.2f}s\n")
            if r['error']:
                f.write(f"  ERROR: {r['error']}\n")
            f.write(f"\n")
    
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    asyncio.run(main())