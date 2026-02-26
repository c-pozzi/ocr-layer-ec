"""
OCR Comparison Tool
Compare AI Model vs PDF OCR against Ground Truth
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import difflib
import re
from typing import Tuple, List

# Configuration (relative to this script)
SCRIPT_DIR = Path(__file__).resolve().parent
OCR_OUTPUTS_DIR = SCRIPT_DIR.parent / "results" / "ocr_outputs"

st.set_page_config(
    page_title="OCR Comparison Tool",
    page_icon="📄",
    layout="wide"
)

def load_documents(ocr_dir: Path, use_normalized: bool = False) -> dict:
    """
    Load all document IDs and their associated files.
    Scans directory for triplets:
      - {doc_id}_ground_truth.txt
      - {doc_id}_pdf_ocr.txt  
      - {doc_id}_qwen_ocr.txt
    If use_normalized=True, loads the _norm.txt versions instead.
    """
    documents = {}
    
    # Suffix for normalized files
    suffix = "_norm" if use_normalized else ""
    
    # First, collect all files and group by prefix
    all_files = list(ocr_dir.glob("*.txt"))
    
    # Find all ground truth files to identify document IDs
    # Always look for non-normalized files to get doc_ids, then load appropriate version
    for file in all_files:
        filename = file.name
        
        # Skip normalized files when scanning for doc_ids
        if "_norm.txt" in filename:
            continue
        
        if filename.endswith("_ground_truth.txt"):
            # Extract doc_id: everything before _ground_truth.txt
            doc_id = filename.replace("_ground_truth.txt", "")
            
            # Build paths with appropriate suffix
            gt_file = ocr_dir / f"{doc_id}_ground_truth{suffix}.txt"
            qwen_file = ocr_dir / f"{doc_id}_qwen_ocr{suffix}.txt"
            pdf_file = ocr_dir / f"{doc_id}_pdf_ocr{suffix}.txt"
            
            documents[doc_id] = {
                'ground_truth': gt_file if gt_file.exists() else None,
                'qwen_ocr': qwen_file if qwen_file.exists() else None,
                'pdf_ocr': pdf_file if pdf_file.exists() else None
            }
    
    # Sort by doc_id for consistent ordering
    return dict(sorted(documents.items()))

def load_cer_results(ocr_dir: Path) -> pd.DataFrame:
    """Load CER results if available."""
    # Check in parent directory (results/) first, then current directory
    cer_file = ocr_dir.parent / "cer_results.csv"
    if not cer_file.exists():
        cer_file = ocr_dir / "cer_results.csv"
    if cer_file.exists():
        return pd.read_csv(cer_file)
    return None

def read_file(filepath: Path) -> str:
    """Read text file content."""
    if filepath and filepath.exists():
        return filepath.read_text(encoding='utf-8')
    return ""

def compute_char_diff(ground_truth: str, ocr_text: str) -> Tuple[str, int, int]:
    """
    Compute character-level differences and return HTML with highlights.
    Green = correct, Red = error (missing or wrong)
    """
    if not ocr_text:
        return "<em>No OCR output available</em>", 0, 0
    
    # Use difflib to find differences
    matcher = difflib.SequenceMatcher(None, ground_truth, ocr_text, autojunk=False)
    
    html_parts = []
    correct_chars = 0
    error_chars = 0
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            # Matching text - no highlight
            text = ocr_text[j1:j2]
            html_parts.append(escape_html(text))
            correct_chars += len(text)
        elif tag == 'replace':
            # Different text - highlight in red
            text = ocr_text[j1:j2]
            html_parts.append(f'<span style="background-color: #ffcccc; color: #cc0000;">{escape_html(text)}</span>')
            error_chars += len(text)
        elif tag == 'insert':
            # Extra text in OCR - highlight in orange
            text = ocr_text[j1:j2]
            html_parts.append(f'<span style="background-color: #ffe0b3; color: #cc6600;">{escape_html(text)}</span>')
            error_chars += len(text)
        elif tag == 'delete':
            # Missing text from ground truth - show placeholder
            missing_len = i2 - i1
            if missing_len <= 20:
                html_parts.append(f'<span style="background-color: #e0e0e0; color: #666;" title="Missing: {escape_html(ground_truth[i1:i2])}">[...{missing_len}]</span>')
            else:
                html_parts.append(f'<span style="background-color: #e0e0e0; color: #666;">[...{missing_len} chars missing]</span>')
    
    return ''.join(html_parts), correct_chars, error_chars

def compute_word_diff(ground_truth: str, ocr_text: str) -> str:
    """
    Compute word-level differences and return HTML with highlights.
    """
    if not ocr_text:
        return "<em>No OCR output available</em>"
    
    gt_words = ground_truth.split()
    ocr_words = ocr_text.split()
    
    matcher = difflib.SequenceMatcher(None, gt_words, ocr_words)
    
    html_parts = []
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            text = ' '.join(ocr_words[j1:j2])
            html_parts.append(escape_html(text))
        elif tag == 'replace':
            text = ' '.join(ocr_words[j1:j2])
            html_parts.append(f'<span style="background-color: #ffcccc; color: #cc0000;">{escape_html(text)}</span>')
        elif tag == 'insert':
            text = ' '.join(ocr_words[j1:j2])
            html_parts.append(f'<span style="background-color: #ffe0b3; color: #cc6600;">{escape_html(text)}</span>')
        elif tag == 'delete':
            missing = ' '.join(gt_words[i1:i2])
            if len(missing) <= 50:
                html_parts.append(f'<span style="background-color: #e0e0e0; color: #666;" title="Missing: {escape_html(missing)}">[...]</span>')
            else:
                html_parts.append(f'<span style="background-color: #e0e0e0; color: #666;">[...{i2-i1} words missing]</span>')
    
    return ' '.join(html_parts)

def escape_html(text: str) -> str:
    """Escape HTML special characters."""
    return (text
            .replace('&', '&amp;')
            .replace('<', '&lt;')
            .replace('>', '&gt;')
            .replace('\n', '<br>')
            .replace('  ', '&nbsp;&nbsp;'))

def calculate_simple_cer(ground_truth: str, ocr_text: str) -> float:
    """Calculate Character Error Rate."""
    if not ground_truth or not ocr_text:
        return 1.0
    
    # Simple Levenshtein-based CER
    import difflib
    matcher = difflib.SequenceMatcher(None, ground_truth, ocr_text)
    matching = sum(block.size for block in matcher.get_matching_blocks())
    total = max(len(ground_truth), len(ocr_text))
    
    return 1 - (matching / total) if total > 0 else 0

# Main App
st.title("📄 OCR Comparison Tool")
st.markdown("Compare **AI Model** vs **PDF OCR** against Ground Truth")

# Sidebar
st.sidebar.header("Settings")

# Allow custom path input
custom_path = st.sidebar.text_input("OCR Outputs Directory", str(OCR_OUTPUTS_DIR))
ocr_dir = Path(custom_path)

if not ocr_dir.exists():
    st.error(f"Directory not found: {ocr_dir}")
    st.stop()

# Normalized text toggle - NEW
use_normalized = st.sidebar.checkbox("📐 Show Normalized Text", value=False, 
    help="Show normalized text (lowercase, collapsed whitespace, no table formatting) - matches CER calculation")

# Load documents with appropriate version
documents = load_documents(ocr_dir, use_normalized=use_normalized)
cer_df = load_cer_results(ocr_dir)

if not documents:
    st.error(f"No documents found in {ocr_dir}")
    st.stop()

st.sidebar.success(f"Found {len(documents)} documents")

# Show document completeness
complete = sum(1 for d in documents.values() if all(d.values()))
partial = len(documents) - complete
st.sidebar.info(f"✅ {complete} complete triplets\n⚠️ {partial} partial")

# Document selector
doc_ids = sorted(documents.keys())
selected_doc = st.sidebar.selectbox("Select Document", doc_ids)

# Diff mode
diff_mode = st.sidebar.radio("Diff Mode", ["Word-level", "Character-level"])

# Show CER metrics if available
if cer_df is not None:
    st.sidebar.markdown("---")
    st.sidebar.header("CER Metrics")
    
    # Try to match the selected doc to CER results
    # First try exact match, then try partial match on the BAC-xxxx part
doc_match = cer_df[cer_df['doc_id'] == selected_doc]
if doc_match.empty:
    # Try matching on the base document ID (e.g., BAC-0020-1973-0401)
    base_id = '_'.join(selected_doc.split('_')[:-1]) if '_' in selected_doc else selected_doc
    if '-' in base_id and len(base_id.split('-')) >= 2:
        doc_match = cer_df[cer_df['doc_id'].str.contains(base_id.split('-')[0] + '-' + base_id.split('-')[1], na=False)]
if not doc_match.empty:
    row = doc_match.iloc[0]
    st.sidebar.metric("AI Model", f"{row['qwen_cer']:.1%}" if pd.notna(row['qwen_cer']) else "N/A")
    st.sidebar.metric("PDF CoDe", f"{row['pdf_ocr_cer']:.1%}" if pd.notna(row['pdf_ocr_cer']) else "N/A")
    if pd.notna(row.get('improvement')):
        st.sidebar.metric("Improvement", f"{row['improvement']:.1%}")

# Load selected document
doc_files = documents[selected_doc]
ground_truth = read_file(doc_files['ground_truth'])
qwen_ocr = read_file(doc_files['qwen_ocr'])
pdf_ocr = read_file(doc_files['pdf_ocr'])

# Get CER metrics for this document
qwen_cer_value = None
pdf_cer_value = None
if cer_df is not None:
    doc_match = cer_df[cer_df['doc_id'] == selected_doc]
    if not doc_match.empty:
        row = doc_match.iloc[0]
        qwen_cer_value = row['qwen_cer'] if pd.notna(row['qwen_cer']) else None
        pdf_cer_value = row['pdf_ocr_cer'] if pd.notna(row['pdf_ocr_cer']) else None

# Main content area
st.header(f"Document: {selected_doc}")

# Show normalized indicator
if use_normalized:
    st.info("📐 Showing **normalized** text (lowercase, collapsed whitespace, no table formatting)")

# Create three columns
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📋 Ground Truth")
    st.markdown(f"*{len(ground_truth)} characters*")
    
    # Display ground truth in a scrollable container
    st.markdown(
        f'<div style="height: 600px; overflow-y: scroll; padding: 10px; background-color: #f9f9f9; border: 1px solid #ddd; border-radius: 5px; white-space: pre-wrap; font-family: monospace; font-size: 12px;">{escape_html(ground_truth)}</div>',
        unsafe_allow_html=True
    )

with col2:
    st.subheader("🤖 AI Model")
    if qwen_cer_value is not None:
        cer_color = "green" if qwen_cer_value < 0.1 else ("orange" if qwen_cer_value < 0.3 else "red")
        st.markdown(f"**CER: :{cer_color}[{qwen_cer_value:.1%}]**")
    if qwen_ocr:
        st.markdown(f"*{len(qwen_ocr)} characters*")
        
        # Compute diff
        if diff_mode == "Word-level":
            diff_html = compute_word_diff(ground_truth, qwen_ocr)
        else:
            diff_html, correct, errors = compute_char_diff(ground_truth, qwen_ocr)
        
        st.markdown(
            f'<div style="height: 600px; overflow-y: scroll; padding: 10px; background-color: #f0fff0; border: 1px solid #ddd; border-radius: 5px; white-space: pre-wrap; font-family: monospace; font-size: 12px;">{diff_html}</div>',
            unsafe_allow_html=True
        )
    else:
        st.warning("No Qwen OCR output available")

with col3:
    st.subheader("📄 PDF CoDe")
    if pdf_cer_value is not None:
        cer_color = "green" if pdf_cer_value < 0.1 else ("orange" if pdf_cer_value < 0.3 else "red")
        st.markdown(f"**CER: :{cer_color}[{pdf_cer_value:.1%}]**")
    if pdf_ocr:
        st.markdown(f"*{len(pdf_ocr)} characters*")
        
        # Compute diff
        if diff_mode == "Word-level":
            diff_html = compute_word_diff(ground_truth, pdf_ocr)
        else:
            diff_html, correct, errors = compute_char_diff(ground_truth, pdf_ocr)
        
        st.markdown(
            f'<div style="height: 600px; overflow-y: scroll; padding: 10px; background-color: #fff0f0; border: 1px solid #ddd; border-radius: 5px; white-space: pre-wrap; font-family: monospace; font-size: 12px;">{diff_html}</div>',
            unsafe_allow_html=True
        )
    else:
        st.warning("No PDF OCR output available")

# Legend
st.markdown("---")
st.markdown("""
**Legend:**
- <span style="background-color: #ffcccc; color: #cc0000; padding: 2px 5px;">Red</span> = Incorrect/replaced text
- <span style="background-color: #ffe0b3; color: #cc6600; padding: 2px 5px;">Orange</span> = Extra text (not in ground truth)  
- <span style="background-color: #e0e0e0; color: #666; padding: 2px 5px;">[...]</span> = Missing text from ground truth
""", unsafe_allow_html=True)

# Summary statistics
st.markdown("---")
st.header("📊 Summary Statistics")

if cer_df is not None:
    st.subheader("Overall CER Results")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_qwen = cer_df['qwen_cer'].mean()
        st.metric("AI Model CER", f"{avg_qwen:.1%}")
    
    with col2:
        avg_pdf = cer_df['pdf_ocr_cer'].mean()
        st.metric("Avg PDF CoDe CER", f"{avg_pdf:.1%}")
    
    with col3:
        avg_improvement = cer_df['improvement'].mean()
        st.metric("Avg Improvement", f"{avg_improvement:.1%}")
    
    with col4:
        st.metric("Documents", len(cer_df))
    
    # Show full table
    with st.expander("View All Results"):
        #st.dataframe(cer_df, use_container_width=True)
        # Rename columns for display
        display_df = cer_df.rename(columns={
            'qwen_cer': 'AI Model CER',
            'pdf_ocr_cer': 'PDF CoDe CER',
            'inference_time_sec': 'Inference Time (s)'
        })
        st.dataframe(display_df, use_container_width=True)