#!/usr/bin/env python3
"""
Streamlit app: side-by-side comparison of AI OCR vs ground truth.

Features:
  - Browse ALL documents (ground-truth-driven list)
  - OCR output resolved from simple/complex directories automatically
  - Three view modes: Raw, Normalized, Diff (with coloring)
  - CER / WER metrics per document
  - Combined summary table across all OCR tiers

Usage:
    streamlit run compare_app.py --server.port 8501 --server.address 0.0.0.0
"""

import difflib
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

from cer_eval import (
    compute_metrics,
    extract_pdf_text,
    normalize_full,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = Path(
    "/home/ubuntu/ocr-evaluation-samples/20260216_OCR_files/20260216_OCR_files"
)
RESULTS_BASE = SCRIPT_DIR / "results" / "20260216"

# Synthetic test cases
TEST_CASES_GT_DIR = SCRIPT_DIR / "test_cases" / "ground_truth"
TEST_CASES_OCR_DIR = SCRIPT_DIR / "test_cases" / "ocr_output"

# Dataset definitions: (label, input_dir, list of ocr_dirs)
DATASETS = {
    "20260216 (real documents)": {
        "input_dir": DEFAULT_INPUT_DIR,
        "results_base": RESULTS_BASE,
    },
    "Synthetic test cases": {
        "input_dir": TEST_CASES_GT_DIR,
        "results_base": None,  # single OCR dir, handled specially
        "ocr_dirs": [TEST_CASES_OCR_DIR],
    },
}


# ---------------------------------------------------------------------------
# Helpers: find OCR output across directories
# ---------------------------------------------------------------------------

_ACTIVE_OCR_DIRS = {"ocr_simple_v1", "ocr_complex_v2"}


def discover_ocr_dirs() -> list[Path]:
    """Return active ocr_* directories under RESULTS_BASE."""
    if not RESULTS_BASE.exists():
        return []
    return sorted(
        d for d in RESULTS_BASE.iterdir()
        if d.is_dir() and d.name in _ACTIVE_OCR_DIRS
    )


def find_all_ocr_files(doc_id: str, ocr_dirs: list[Path]) -> list[tuple[Path, str]]:
    """Find all OCR output files for doc_id across all OCR directories.

    Returns list of (path, dir_name) — may be empty, or have multiple entries
    when the same doc was processed by different OCR tiers/versions.
    """
    matches = []
    for d in ocr_dirs:
        candidate = d / f"{doc_id}.txt"
        if candidate.exists():
            matches.append((candidate, d.name))
    return matches


def build_doc_index(input_dir: Path, ocr_dirs: list[Path]) -> list[dict]:
    """Build a deduplicated list of all documents with ground truth.

    Each doc appears once. The ``ocr_versions`` field lists all available
    OCR outputs (e.g. ocr_complex_v1 and ocr_complex_v2).
    """
    gt_files = sorted(input_dir.glob("*_ground_truth.txt"))
    docs = []
    for gt in gt_files:
        doc_id = gt.name.replace("_ground_truth.txt", "")
        versions = find_all_ocr_files(doc_id, ocr_dirs)
        # Default to last version (latest alphabetically, e.g. v2 > v1)
        best = versions[-1] if versions else (None, "")
        docs.append({
            "doc_id": doc_id,
            "gt_path": gt,
            "ocr_path": best[0],
            "ocr_source": best[1],
            "has_ocr": best[0] is not None,
            "ocr_versions": versions,
        })
    return docs


# ---------------------------------------------------------------------------
# Diff rendering
# ---------------------------------------------------------------------------

_DIFF_CSS = """
<style>
.diff-container {
    font-family: 'Courier New', monospace;
    font-size: 13px;
    line-height: 1.5;
    white-space: pre-wrap;
    word-wrap: break-word;
    padding: 8px;
    background: #fafafa;
    border: 1px solid #ddd;
    border-radius: 4px;
    max-height: 600px;
    overflow-y: auto;
}
.diff-ins { background-color: #d4edda; color: #155724; }
.diff-del { background-color: #f8d7da; color: #721c24; }
.diff-eq  { color: #333; }
</style>
"""


def render_char_diff(reference: str, hypothesis: str) -> str:
    sm = difflib.SequenceMatcher(None, reference, hypothesis, autojunk=False)
    parts: list[str] = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            parts.append(f'<span class="diff-eq">{escape(reference[i1:i2])}</span>')
        elif op == "replace":
            parts.append(f'<span class="diff-del">{escape(reference[i1:i2])}</span>')
            parts.append(f'<span class="diff-ins">{escape(hypothesis[j1:j2])}</span>')
        elif op == "delete":
            parts.append(f'<span class="diff-del">{escape(reference[i1:i2])}</span>')
        elif op == "insert":
            parts.append(f'<span class="diff-ins">{escape(hypothesis[j1:j2])}</span>')
    return "".join(parts)


def render_word_diff(reference: str, hypothesis: str) -> str:
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    sm = difflib.SequenceMatcher(None, ref_words, hyp_words, autojunk=False)
    parts: list[str] = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            parts.append(f'<span class="diff-eq">{escape(" ".join(ref_words[i1:i2]))}</span>')
        elif op == "replace":
            parts.append(f'<span class="diff-del">{escape(" ".join(ref_words[i1:i2]))}</span>')
            parts.append(f'<span class="diff-ins">{escape(" ".join(hyp_words[j1:j2]))}</span>')
        elif op == "delete":
            parts.append(f'<span class="diff-del">{escape(" ".join(ref_words[i1:i2]))}</span>')
        elif op == "insert":
            parts.append(f'<span class="diff-ins">{escape(" ".join(hyp_words[j1:j2]))}</span>')
    return " ".join(parts)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(layout="wide", page_title="OCR Compare")
    st.title("OCR Evaluation — Compare & Metrics")

    tab_compare, tab_tests = st.tabs(["Compare", "Unit Tests"])

    with tab_tests:
        _show_test_suite()

    with tab_compare:
        _page_compare()


def _show_test_suite():
    """Run and display unit test results."""
    from test_cer_eval import ALL_TEST_CLASSES, run_tests_programmatic

    st.subheader("CER / WER Unit Tests")
    st.caption("Synthetic test cases illustrating metric behavior and normalization edge cases.")

    if st.button("Run all tests"):
        result, output = run_tests_programmatic()

        # Summary metrics
        total = result.testsRun
        failures = len(result.failures)
        errors = len(result.errors)
        passed = total - failures - errors

        col1, col2, col3 = st.columns(3)
        col1.metric("Passed", passed)
        col2.metric("Failed", failures)
        col3.metric("Errors", errors)

        if failures == 0 and errors == 0:
            st.success(f"All {total} tests passed.")
        else:
            st.error(f"{failures} failures, {errors} errors out of {total} tests.")

        # Full output
        with st.expander("Full test output", expanded=True):
            st.code(output, language="text")

        # Show failures/errors in detail
        for test, traceback in result.failures:
            st.error(f"FAIL: {test}")
            st.code(traceback, language="python")
        for test, traceback in result.errors:
            st.error(f"ERROR: {test}")
            st.code(traceback, language="python")

    # Always show test catalog
    st.markdown("---")
    st.subheader("Test catalog")
    for cls in ALL_TEST_CLASSES:
        with st.expander(f"{cls.__name__}", expanded=False):
            import unittest
            loader = unittest.TestLoader()
            for test_name in loader.getTestCaseNames(cls):
                method = getattr(cls, test_name)
                doc = method.__doc__ or ""
                st.markdown(f"- **{test_name}** — {doc}")


def _page_compare():
    """Main comparison page."""
    # --- Sidebar controls ---
    st.sidebar.header("Settings")

    # Dataset selector
    dataset_names = list(DATASETS.keys())
    selected_dataset = st.sidebar.selectbox("Dataset", dataset_names, index=0)
    ds = DATASETS[selected_dataset]

    input_dir = ds["input_dir"]
    if ds.get("ocr_dirs"):
        ocr_dirs = ds["ocr_dirs"]
    else:
        ocr_dirs = discover_ocr_dirs()

    if not ocr_dirs:
        st.error(f"No OCR output directories found")
        return

    # --- Build document index ---
    doc_index = build_doc_index(input_dir, ocr_dirs)
    if not doc_index:
        st.error(f"No ground truth files found in {input_dir}")
        return

    # Filter by OCR source
    filter_options = ["All documents"] + [d.name for d in ocr_dirs] + ["No OCR output"]
    source_filter = st.sidebar.selectbox("Filter by OCR source", filter_options, index=0)

    if source_filter == "All documents":
        filtered_docs = doc_index
    elif source_filter == "No OCR output":
        filtered_docs = [d for d in doc_index if not d["has_ocr"]]
    else:
        # Show docs that have output in the selected dir (check all versions, not just default)
        filtered_docs = [
            d for d in doc_index
            if any(vname == source_filter for _, vname in d.get("ocr_versions", []))
        ]

    # View mode
    view_mode = st.sidebar.radio(
        "View mode",
        ["Raw", "Normalized", "Diff (character)", "Diff (word)", "Metrics detail"],
        index=0,
    )

    # Document picker — one entry per doc
    if not filtered_docs:
        st.warning("No documents match the selected filter.")
        return

    doc_labels = [
        f"{d['doc_id']}  [{d['ocr_source']}]" if d["has_ocr"] else f"{d['doc_id']}  [no OCR]"
        for d in filtered_docs
    ]
    selected_idx = st.sidebar.selectbox("Document", range(len(doc_labels)), format_func=lambda i: doc_labels[i], index=0)
    selected = filtered_docs[selected_idx]
    doc_id = selected["doc_id"]

    # If multiple OCR versions exist, let user pick
    versions = selected.get("ocr_versions", [])
    ocr_path = selected["ocr_path"]
    ocr_source = selected["ocr_source"]

    if len(versions) > 1:
        version_labels = [v[1] for v in versions]
        # Default to last (latest)
        default_ver = len(version_labels) - 1
        chosen_ver = st.sidebar.selectbox("OCR version", version_labels, index=default_ver)
        for vpath, vname in versions:
            if vname == chosen_ver:
                ocr_path = vpath
                ocr_source = vname
                break

    st.sidebar.caption(f"Source: **{ocr_source or 'none'}**")

    # --- Load texts ---
    gt_raw = selected["gt_path"].read_text(encoding="utf-8")

    if ocr_path:
        ai_raw = ocr_path.read_text(encoding="utf-8")
    else:
        ai_raw = None

    pdf_path = input_dir / f"{doc_id}.pdf"
    legacy_raw = extract_pdf_text(pdf_path) if pdf_path.exists() else None

    # --- Metrics sidebar ---
    gt_norm = normalize_full(gt_raw)
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Metrics (this doc)")

    if ai_raw:
        ai_norm = normalize_full(ai_raw)
        m_ai = compute_metrics(gt_norm, ai_norm)
        col_a, col_b = st.sidebar.columns(2)
        col_a.metric("AI CER", f"{m_ai['cer']:.2%}")
        col_b.metric("AI WER", f"{m_ai['wer']:.2%}")
    else:
        st.sidebar.info("No AI OCR output for this document")
        ai_norm = None

    if legacy_raw:
        legacy_norm = normalize_full(legacy_raw)
        m_pdf = compute_metrics(gt_norm, legacy_norm)
        col_c, col_d = st.sidebar.columns(2)
        col_c.metric("PDF CER", f"{m_pdf['cer']:.2%}")
        col_d.metric("PDF WER", f"{m_pdf['wer']:.2%}")
    else:
        legacy_norm = None

    st.sidebar.markdown(f"**GT**: {len(gt_norm)} chars (norm) / {len(gt_raw)} raw")
    if ai_raw:
        st.sidebar.markdown(f"**AI OCR**: {len(ai_norm)} chars (norm) / {len(ai_raw)} raw")
    if legacy_raw:
        st.sidebar.markdown(f"**PDF OCR**: {len(legacy_norm)} chars (norm) / {len(legacy_raw)} raw")

    # --- Summary table (combined from all OCR dirs) ---
    all_csvs = []
    for d in ocr_dirs:
        csv_path = d / "cer_wer_results.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df["ocr_source"] = d.name
            all_csvs.append(df)

    if all_csvs:
        summary_df = pd.concat(all_csvs, ignore_index=True)
        with st.expander(f"Summary — {len(summary_df)} documents across {len(all_csvs)} OCR dirs", expanded=False):
            def highlight_current(row):
                if row["doc_id"] == doc_id:
                    return ["background-color: #fff3cd"] * len(row)
                return [""] * len(row)

            pct_cols = [c for c in summary_df.columns if c in ("cer", "wer", "pdf_cer", "pdf_wer", "cer_improvement", "wer_improvement")]
            styled = summary_df.style.apply(highlight_current, axis=1).format(
                {c: "{:.2%}" for c in pct_cols}
            )
            st.dataframe(styled, use_container_width=True, hide_index=True)

            st.sidebar.markdown("---")
            with st.sidebar.expander("Refresh CSV data", expanded=False):
                st.code(f"cd {SCRIPT_DIR}\npython cer_eval.py --ocr-dir <dir>", language="bash")

    # --- Document image ---
    if pdf_path.exists():
        with st.expander("Document image", expanded=False):
            try:
                import fitz
                doc = fitz.open(pdf_path)
                page = doc[0]
                pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
                img_bytes = pix.tobytes("png")
                doc.close()
                st.image(img_bytes, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not render PDF: {e}")

    # --- Main content area ---
    if view_mode == "Raw":
        _show_texts(ai_raw, legacy_raw, gt_raw)
    elif view_mode == "Normalized":
        _show_texts(
            ai_norm, legacy_norm, gt_norm,
            label_suffix=" (normalized)",
        )
    elif view_mode.startswith("Diff"):
        char_level = "character" in view_mode
        _show_diff(ai_raw, ai_norm, legacy_raw, legacy_norm, gt_raw, gt_norm, char_level)
    elif view_mode == "Metrics detail":
        _show_metrics_detail(ai_norm, legacy_norm, gt_norm)


def _show_texts(
    ai_text: str | None,
    legacy_text: str | None,
    gt_text: str | None,
    label_suffix: str = "",
):
    """Show available texts side by side."""
    panels = []
    if ai_text is not None:
        panels.append(("AI OCR", ai_text))
    if legacy_text is not None:
        panels.append(("Legacy PDF", legacy_text))
    if gt_text is not None:
        panels.append(("Ground Truth", gt_text))

    if not panels:
        st.warning("No texts to display.")
        return

    cols = st.columns(len(panels))
    for col, (label, text) in zip(cols, panels):
        with col:
            st.subheader(f"{label}{label_suffix}")
            st.text_area(label, text, height=600, label_visibility="collapsed")


def _show_diff(
    ai_raw: str | None,
    ai_norm: str | None,
    legacy_raw: str | None,
    legacy_norm: str | None,
    gt_raw: str | None,
    gt_norm: str | None,
    char_level: bool,
):
    """Show diff-colored comparison against ground truth."""
    if not gt_norm:
        st.warning("No ground truth available for diff view.")
        return

    render_fn = render_char_diff if char_level else render_word_diff
    diff_type = "character" if char_level else "word"

    st.markdown(_DIFF_CSS, unsafe_allow_html=True)
    st.caption(
        f"Showing {diff_type}-level diff against ground truth (normalized). "
        "<span class='diff-del'>Red = in ground truth but missing from OCR</span> · "
        "<span class='diff-ins'>Green = in OCR but not in ground truth</span>",
        unsafe_allow_html=True,
    )

    # Build columns for available sources
    panels = []
    if ai_norm is not None:
        panels.append(("AI OCR vs Ground Truth", ai_norm))
    if legacy_norm is not None:
        panels.append(("Legacy PDF vs Ground Truth", legacy_norm))

    if not panels:
        st.warning("No OCR output available for diff.")
        return

    cols = st.columns(len(panels))
    for col, (label, hyp_norm) in zip(cols, panels):
        with col:
            st.subheader(label)
            html = render_fn(gt_norm, hyp_norm)
            st.markdown(f'<div class="diff-container">{html}</div>', unsafe_allow_html=True)

    with st.expander("Ground truth (normalized)", expanded=False):
        st.text_area("gt_norm_ref", gt_norm, height=400, label_visibility="collapsed")


def _edit_ops(reference: str, hypothesis: str) -> dict:
    """Count edit operations (char-level) between reference and hypothesis."""
    import Levenshtein
    ops = Levenshtein.editops(reference, hypothesis)
    counts = {"replace": 0, "delete": 0, "insert": 0}
    for op, _, _ in ops:
        counts[op] += 1
    counts["total"] = sum(counts.values())
    return counts


def _edit_ops_words(reference: str, hypothesis: str) -> dict:
    """Count edit operations (word-level) between reference and hypothesis."""
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    sm = difflib.SequenceMatcher(None, ref_words, hyp_words, autojunk=False)
    counts = {"replace": 0, "delete": 0, "insert": 0}
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "replace":
            counts["replace"] += max(i2 - i1, j2 - j1)
        elif op == "delete":
            counts["delete"] += i2 - i1
        elif op == "insert":
            counts["insert"] += j2 - j1
    counts["total"] = sum(counts.values())
    return counts


def _show_metrics_detail(
    ai_norm: str | None,
    legacy_norm: str | None,
    gt_norm: str | None,
):
    """Show normalized texts with full CER/WER breakdown so you can verify by hand."""
    if not gt_norm:
        st.warning("No ground truth available.")
        return

    st.markdown(_DIFF_CSS, unsafe_allow_html=True)

    sources = []
    if ai_norm is not None:
        sources.append(("AI OCR", ai_norm))
    if legacy_norm is not None:
        sources.append(("Legacy PDF", legacy_norm))

    if not sources:
        st.warning("No OCR output available.")
        return

    # --- Ground truth ---
    gt_words = gt_norm.split()
    st.subheader("Ground Truth (normalized)")
    st.code(gt_norm, language=None)
    st.caption(f"{len(gt_norm)} characters, {len(gt_words)} words")

    st.markdown("---")

    # --- Per-source breakdown ---
    for label, hyp_norm in sources:
        st.subheader(f"{label} (normalized)")
        st.code(hyp_norm, language=None)

        hyp_words = hyp_norm.split()
        st.caption(f"{len(hyp_norm)} characters, {len(hyp_words)} words")

        # Character-level
        char_ops = _edit_ops(gt_norm, hyp_norm)
        m = compute_metrics(gt_norm, hyp_norm)

        st.markdown("#### Character Error Rate (CER)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Substitutions", char_ops["replace"])
        c2.metric("Deletions", char_ops["delete"])
        c3.metric("Insertions", char_ops["insert"])
        c4.metric("Total edits", char_ops["total"])

        st.markdown(
            f"**CER = {char_ops['total']} / {len(gt_norm)} = {m['cer']:.4f} ({m['cer']:.2%})**"
        )
        st.caption(
            f"Formula: (substitutions + deletions + insertions) / reference_length "
            f"= ({char_ops['replace']} + {char_ops['delete']} + {char_ops['insert']}) / {len(gt_norm)}"
        )

        # Word-level
        word_ops = _edit_ops_words(gt_norm, hyp_norm)

        st.markdown("#### Word Error Rate (WER)")
        w1, w2, w3, w4 = st.columns(4)
        w1.metric("Substitutions", word_ops["replace"])
        w2.metric("Deletions", word_ops["delete"])
        w3.metric("Insertions", word_ops["insert"])
        w4.metric("Total edits", word_ops["total"])

        st.markdown(
            f"**WER = {word_ops['total']} / {len(gt_words)} = {m['wer']:.4f} ({m['wer']:.2%})**"
        )
        st.caption(
            f"Formula: (substitutions + deletions + insertions) / reference_words "
            f"= ({word_ops['replace']} + {word_ops['delete']} + {word_ops['insert']}) / {len(gt_words)}"
        )

        st.markdown("---")


if __name__ == "__main__":
    main()
