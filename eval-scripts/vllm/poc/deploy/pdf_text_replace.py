#!/usr/bin/env python3
"""
Replace the invisible OCR text layer on specified PDF pages.

Uses PyMuPDF (fitz) to:
1. Remove existing text via redaction (preserving scanned images)
2. Insert new AI OCR text as invisible overlay (render_mode=3)

Pages not listed in replacements are left untouched.
"""

import logging
from pathlib import Path

import fitz  # PyMuPDF

log = logging.getLogger(__name__)


def replace_text_layer(
    input_pdf: Path,
    output_pdf: Path,
    page_replacements: dict[int, str],
) -> dict:
    """
    Replace the invisible text layer on specified pages of a PDF.

    Args:
        input_pdf: Path to the source PDF.
        output_pdf: Path to write the modified PDF.
        page_replacements: Mapping of 0-based page number → new OCR text.
            Pages not in this dict are copied unchanged.

    Returns:
        Dict with stats: pages_total, pages_modified, pages_skipped.
    """
    doc = fitz.open(input_pdf)
    pages_modified = 0

    for page_num, new_text in page_replacements.items():
        if page_num < 0 or page_num >= len(doc):
            log.warning(
                "Page %d out of range for %s (%d pages), skipping",
                page_num, input_pdf.name, len(doc),
            )
            continue

        if not new_text or not new_text.strip():
            log.warning(
                "Empty OCR text for %s page %d, skipping replacement",
                input_pdf.name, page_num,
            )
            continue

        page = doc[page_num]
        _replace_page_text(page, new_text)
        pages_modified += 1

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(output_pdf), garbage=3, deflate=True)
    doc.close()

    stats = {
        "pages_total": len(doc) if not doc.is_closed else pages_modified,
        "pages_modified": pages_modified,
        "pages_skipped": len(page_replacements) - pages_modified,
    }
    log.info(
        "%s: %d pages modified, saved to %s",
        input_pdf.name, pages_modified, output_pdf.name,
    )
    return stats


def _replace_page_text(page: fitz.Page, new_text: str) -> None:
    """Remove existing text layer and insert new invisible text."""
    # Step 1: Remove existing text via full-page redaction.
    # images=PDF_REDACT_IMAGE_NONE preserves the scanned image underneath.
    page.add_redact_annot(page.rect)
    page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)

    # Step 2: Insert new text as invisible overlay.
    # render_mode=3 is the PDF Tr operator for invisible text —
    # not visible on screen but searchable and copy-pasteable.
    page.insert_textbox(
        page.rect,
        new_text,
        fontsize=3,
        fontname="helv",
        render_mode=3,
        align=0,
    )


def replace_text_layers_batch(
    input_dir: Path,
    output_dir: Path,
    ocr_text_dir: Path,
    complex_pages: list[dict],
) -> list[dict]:
    """
    Batch-replace text layers for all documents that have complex pages.

    Args:
        input_dir: Directory containing source PDFs.
        output_dir: Directory for modified PDFs.
        ocr_text_dir: Directory containing OCR text files named
            ``{pdf_filename}__page{page_num}.txt``.
        complex_pages: List of dicts with at least ``pdf_filename`` and
            ``page_num`` keys (0-based).

    Returns:
        List of per-document stats dicts.
    """
    # Group complex pages by document
    from collections import defaultdict
    doc_pages: dict[str, list[int]] = defaultdict(list)
    for p in complex_pages:
        doc_pages[p["pdf_filename"]].append(p["page_num"])

    all_stats = []
    for pdf_filename, page_nums in sorted(doc_pages.items()):
        input_pdf = input_dir / pdf_filename
        output_pdf = output_dir / pdf_filename

        if not input_pdf.exists():
            log.error("Source PDF not found: %s", input_pdf)
            continue

        # Load OCR text for each complex page
        replacements: dict[int, str] = {}
        for pn in page_nums:
            text_file = ocr_text_dir / f"{pdf_filename}__page{pn}.txt"
            if text_file.exists():
                replacements[pn] = text_file.read_text(encoding="utf-8")
            else:
                log.warning("OCR text missing for %s page %d", pdf_filename, pn)

        if replacements:
            stats = replace_text_layer(input_pdf, output_pdf, replacements)
            stats["pdf_filename"] = pdf_filename
            all_stats.append(stats)

    return all_stats
