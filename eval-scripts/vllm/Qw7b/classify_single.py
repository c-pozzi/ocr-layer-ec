"""
Quick single-file classification using transformers directly (no vLLM).
Usage: python classify_single.py <pdf_path> [--bits 8|4]
"""

import argparse
import base64
import json
import fitz  # PyMuPDF
from pathlib import Path

CLASSIFY_PROMPT = """Analyze this document image and return ONLY a JSON object with these boolean fields:

{
  "multi_column": true/false,    // Text arranged in 2+ parallel columns
  "has_tables": true/false,      // Contains tabular data with rows/columns
  "handwritten": true/false,     // Contains ANY handwritten text (signatures, annotations, filled forms)
  "has_stamps": true/false,      // Contains official stamps or date stamps
  "poor_quality": true/false,    // Faded, stained, skewed, or hard to read
  "has_strikethrough": true/false, // Contains crossed-out/struck text
  "latin_script": true/false,    // ALL text uses Latin alphabet (false if Greek, Cyrillic, Arabic present)
  "has_footnotes": true/false,   // Contains footnotes at bottom of page
  "has_forms": true/false        // Contains pre-printed fields designed to be filled in
}

Return ONLY the JSON, no explanation."""


def pdf_to_pil_images(pdf_path, dpi=150):
    from PIL import Image
    from io import BytesIO
    images = []
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc[page_num]
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        img = Image.open(BytesIO(pix.tobytes("png")))
        images.append((page_num + 1, img))
    doc.close()
    return images


def main():
    parser = argparse.ArgumentParser(description="Classify a single PDF with Qwen2.5-VL")
    parser.add_argument("pdf_path", type=str, help="Path to the PDF file")
    parser.add_argument("--bits", type=int, choices=[4, 8], default=8,
                        help="Quantization: 8-bit (default) or 4-bit")
    args = parser.parse_args()

    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        print(f"File not found: {pdf_path}")
        return

    print(f"Loading Qwen2.5-VL-7B-Instruct with {args.bits}-bit quantization...")

    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
    import torch

    model_name = "Qwen/Qwen2.5-VL-7B-Instruct"

    if args.bits == 8:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float16, load_in_8bit=True, device_map="auto"
        )
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float16, load_in_4bit=True, device_map="auto"
        )

    processor = AutoProcessor.from_pretrained(model_name)

    print(f"Processing: {pdf_path.name}")
    pages = pdf_to_pil_images(pdf_path)

    for page_num, pil_image in pages:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": CLASSIFY_PROMPT},
                ],
            }
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=256, temperature=0.0, do_sample=False)

        # Trim input tokens from output
        generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
        response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        print(f"\nPage {page_num}: raw response:")
        print(response)

        try:
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]
            result = json.loads(response.strip())
            print(f"\nParsed classification:")
            for k, v in result.items():
                print(f"  {k}: {v}")
        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")


if __name__ == "__main__":
    main()
