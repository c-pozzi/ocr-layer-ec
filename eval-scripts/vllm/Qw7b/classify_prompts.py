"""
Document Classification Script

Reads PDFs from evaluation samples and classifies them using vLLM server.
Outputs results to CSV.

Usage:
    python classify_prompts.py [--model {bf16,int4}]
"""

import argparse
import asyncio
import aiohttp
import base64
import json
import os
import csv
from pathlib import Path
from datetime import datetime
import fitz  # PyMuPDF for PDF reading

from prompts import get_classify_prompt

# =============================================================================
# CLI ARGUMENTS
# =============================================================================

MODEL_CONFIGS = {
    "bf16": "Qwen/Qwen2.5-VL-7B-Instruct",
    "int4": "Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
}

def parse_args():
    parser = argparse.ArgumentParser(description="Classify documents using vLLM server")
    parser.add_argument(
        "--model", choices=list(MODEL_CONFIGS.keys()), default="bf16",
        help="Model variant to use: bf16 (default) or int4 (AWQ quantized)"
    )
    return parser.parse_args()

# =============================================================================
# CONFIGURATION
# =============================================================================

# Input/Output paths (relative to this script)
SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR.parents[2] / "ocr-evaluation-samples"
OUTPUT_DIR = SCRIPT_DIR / "results_classify"

# vLLM server configuration
VLLM_SERVERS = [f"http://localhost:{8000 + i}" for i in range(8)]
VLLM_ENDPOINT = "/v1/chat/completions"
MODEL_NAME = MODEL_CONFIGS["bf16"]  # overridden in main() based on --model arg

# Concurrency
MAX_CONCURRENCY = 8

# =============================================================================
# PDF PROCESSING
# =============================================================================

def pdf_to_base64_images(pdf_path: Path, dpi: int = 150) -> list[tuple[int, str]]:
    """
    Convert PDF pages to base64 encoded images.
    
    Returns:
        List of (page_number, base64_string) tuples
    """
    images = []
    doc = fitz.open(pdf_path)
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        # Render page to image
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        
        # Convert to PNG bytes
        img_bytes = pix.tobytes("png")
        
        # Encode to base64
        b64_string = base64.standard_b64encode(img_bytes).decode("utf-8")
        images.append((page_num + 1, b64_string))
    
    doc.close()
    return images


def find_pdf_files(input_dir: Path) -> list[Path]:
    """Find all PDF files in the input directory."""
    return sorted(input_dir.glob("*.pdf"))

# =============================================================================
# VLLM CLIENT
# =============================================================================

async def classify_image(
    session: aiohttp.ClientSession,
    server_url: str,
    image_b64: str,
    prompt: str
) -> dict:
    """
    Send image to vLLM server for classification.
    
    Returns:
        Parsed JSON classification or error dict
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
        "max_tokens": 256,
        "temperature": 0.0
    }
    
    try:
        async with session.post(
            f"{server_url}{VLLM_ENDPOINT}",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=60)
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                return {"error": f"HTTP {response.status}: {error_text}"}
            
            result = await response.json()
            content = result["choices"][0]["message"]["content"]
            
            # Parse JSON from response
            try:
                # Handle potential markdown code blocks
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0]
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0]
                
                return json.loads(content.strip())
            except json.JSONDecodeError as e:
                return {"error": f"JSON parse error: {e}", "raw_response": content}
                
    except asyncio.TimeoutError:
        return {"error": "Request timeout"}
    except Exception as e:
        return {"error": str(e)}

# =============================================================================
# MAIN PROCESSING
# =============================================================================

async def process_pdf(
    pdf_path: Path,
    semaphore: asyncio.Semaphore,
    session: aiohttp.ClientSession,
    prompt: str,
    server_idx: int
) -> list[dict]:
    """
    Process a single PDF file, classifying each page.
    
    Returns:
        List of classification results per page
    """
    results = []
    server_url = VLLM_SERVERS[server_idx % len(VLLM_SERVERS)]
    
    async with semaphore:
        print(f"Processing: {pdf_path.name}")
        
        try:
            # Convert PDF to images
            pages = pdf_to_base64_images(pdf_path)
            
            for page_num, image_b64 in pages:
                classification = await classify_image(
                    session, server_url, image_b64, prompt
                )
                
                results.append({
                    "filename": pdf_path.name,
                    "page": page_num,
                    "total_pages": len(pages),
                    "server": server_url,
                    **flatten_classification(classification)
                })
                
                print(f"  Page {page_num}/{len(pages)}: {classification}")
                
        except Exception as e:
            results.append({
                "filename": pdf_path.name,
                "page": 0,
                "total_pages": 0,
                "error": str(e)
            })
            print(f"  Error: {e}")
    
    return results


def flatten_classification(classification: dict) -> dict:
    """Flatten classification dict for CSV output."""
    if "error" in classification:
        return {
            "multi_column": None,
            "has_tables": None,
            "handwritten": None,
            "has_stamps": None,
            "poor_quality": None,
            "has_strikethrough": None,
            "latin_script": None,
            "has_footnotes": None,
            "has_forms": None,
            "error": classification.get("error"),
            "raw_response": classification.get("raw_response", "")
        }
    
    return {
        "multi_column": classification.get("multi_column"),
        "has_tables": classification.get("has_tables"),
        "handwritten": classification.get("handwritten"),
        "has_stamps": classification.get("has_stamps"),
        "poor_quality": classification.get("poor_quality"),
        "has_strikethrough": classification.get("has_strikethrough"),
        "latin_script": classification.get("latin_script"),
        "has_footnotes": classification.get("has_footnotes"),
        "has_forms": classification.get("has_forms"),
        "error": None,
        "raw_response": ""
    }


async def main():
    """Main entry point."""
    args = parse_args()

    # Set model name based on variant
    global MODEL_NAME
    MODEL_NAME = MODEL_CONFIGS[args.model]
    OUTPUT_CSV = OUTPUT_DIR / f"classification_results_{args.model}.csv"

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Find PDF files
    pdf_files = find_pdf_files(INPUT_DIR)

    if not pdf_files:
        print(f"No PDF files found in {INPUT_DIR}")
        return

    print(f"Found {len(pdf_files)} PDF files")
    print(f"Model variant: {args.model} ({MODEL_NAME})")
    print(f"Using {len(VLLM_SERVERS)} vLLM servers")
    print(f"Output: {OUTPUT_CSV}")
    print("-" * 60)
    
    # Get classification prompt
    prompt = get_classify_prompt()
    
    # Process all PDFs
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    
    async with aiohttp.ClientSession() as session:
        tasks = [
            process_pdf(pdf_path, semaphore, session, prompt, idx)
            for idx, pdf_path in enumerate(pdf_files)
        ]
        
        all_results = await asyncio.gather(*tasks)
    
    # Flatten results
    results = [item for sublist in all_results for item in sublist]
    
    # Write CSV
    if results:
        fieldnames = [
            "filename", "page", "total_pages", "server",
            "multi_column", "has_tables", "handwritten", "has_stamps",
            "poor_quality", "has_strikethrough", "latin_script",
            "has_footnotes", "has_forms", "error", "raw_response"
        ]
        
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        print("-" * 60)
        print(f"✅ Results saved to {OUTPUT_CSV}")
        print(f"   Total pages processed: {len(results)}")
        
        # Summary stats
        errors = sum(1 for r in results if r.get("error"))
        if errors:
            print(f"   Errors: {errors}")
    else:
        print("No results to save")


if __name__ == "__main__":
    asyncio.run(main())