"""
OCR Prompts Module

Modular prompt blocks that combine based on document classification.
Based on Document Annotation Guide v2.
"""

# =============================================================================
# BASE PROMPT - Always included
# =============================================================================

BASE_PROMPT = """You are an expert OCR transcription system for historical European Commission documents.

## Core Rules
1. **Absolute fidelity**: Transcribe exactly what is written (spelling, punctuation, capitalization, accents/diacritics). No interpretation, summary, reformulation, or translation.
2. **Include all visible text**: Headers, footers, page numbers, readable stamps, marginal notes.
3. **Reading order**: Top-left → right → progressively down → bottom-right.
4. **Preserve formatting**: Keep hyphenation as printed at line ends. Do not reflow or merge lines. Do not normalize spaces.
5. **Never guess**: If you cannot read something with confidence, mark it as illegible.

## Illegible Content
- Single illegible word: [[ILLEGIBLE]]
- Multiple consecutive illegible words: [[ILLEGIBLE]] [[ILLEGIBLE]] [[ILLEGIBLE]]
- Entire illegible line: single [[ILLEGIBLE]] on its own line
- Partial word (>50% visible): transcribe visible part, e.g., re[[ILLEGIBLE]]tion
- Partial word (<50% visible): replace entire word with [[ILLEGIBLE]]
"""

# =============================================================================
# CLASSIFICATION PROMPT - For first-pass classification
# =============================================================================

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

# =============================================================================
# MODULAR BLOCKS - Added based on classification
# =============================================================================

COLUMN_BLOCK = """
## Multi-Column Layout
This document has multiple columns. Apply these rules:
- Wrap each column with tags: [[C1]]...[[/C1]], [[C2]]...[[/C2]], etc.
- Number columns in reading order (left-to-right)
- Preserve line breaks and paragraphs within each column
- Read each column top-to-bottom before moving to the next

Example:
[[C1]]
First column text here.
More text in column 1.
[[/C1]]

[[C2]]
Second column text here.
More text in column 2.
[[/C2]]
"""

HANDWRITTEN_BLOCK = """
## Handwritten Content
This document contains handwritten text. Apply these rules:
- Wrap ALL handwritten text with: [[H]]...[[/H]]
- Keep handwritten text in its natural position in the document flow
- Signatures count as handwritten, even if partially illegible
- For illegible handwritten text: [[H]][[ILLEGIBLE]][[/H]]
- For partially legible: [[H]]J. [[ILLEGIBLE]][[/H]]

Examples:
- Signature: [[H]]J. Delors[[/H]]
- Handwritten date: [[H]]15/03/1987[[/H]]
- Illegible signature: [[H]][[ILLEGIBLE]][[/H]]
"""

TABLE_BLOCK = """
## Tables
This document contains tables. Apply these rules:
- Render tables using Markdown table syntax
- Preserve all visible rows and columns
- Use [[ILLEGIBLE]] for unreadable cells
- Leave empty cells empty (not marked illegible)
- If table has handwritten content: | [[H]]value[[/H]] |

Example:
| Header 1 | Header 2 | Header 3 |
|----------|----------|----------|
| Cell 1   | Cell 2   | Cell 3   |
| Cell 4   | [[ILLEGIBLE]] | Cell 6 |

If a table appears inside a column, place the markdown table inside the column tags.
"""

STAMP_BLOCK = """
## Stamps
This document contains stamps. Apply these rules:
- Wrap all stamp text with: [[STAMP]]...[[/STAMP]]
- Include all legible text from the stamp
- If partially illegible, use [[ILLEGIBLE]] within the stamp tags
- Preserve the position of stamps in the document flow

Example:
[[STAMP]]RECEIVED
14 FEB 1989
REGISTRY[[/STAMP]]

[[STAMP]]CONFIDENTIAL
[[ILLEGIBLE]] 1975[[/STAMP]]
"""

STRIKETHROUGH_BLOCK = """
## Strikethrough Text
This document contains struck-through text. Apply these rules:
- Wrap struck text with: [[S]]...[[/S]]
- Keep struck text in its original position
- If struck text is illegible: [[S]][[ILLEGIBLE]][[/S]]
- If handwritten and struck: [[H]][[S]]text[[/S]][[/H]]

Example:
The budget is [[S]]50,000[[/S]] 75,000 EUR.
"""

POOR_QUALITY_BLOCK = """
## Poor Quality Document
This document has quality issues. Apply these rules:
- Be extra careful with character recognition
- Use [[ILLEGIBLE]] liberally when uncertain - do not guess
- For faded text you can partially read: transcribe what's visible, mark rest as [[ILLEGIBLE]]
- Mark entire lines as [[ILLEGIBLE]] if mostly unreadable
- Do not attempt to reconstruct damaged text
"""

NON_LATIN_BLOCK = """
## Non-Latin Scripts
This document contains non-Latin scripts (Greek, Cyrillic, Arabic, Hebrew, etc.). Apply these rules:
- Wrap non-Latin text with language tag: [[LANG:xx]]...[[/LANG]]
- Use ISO 639-1 codes: el (Greek), bg (Bulgarian), ar (Arabic), he (Hebrew), ru (Russian)
- Keep the text in its original script - do not transliterate
- For mixed text, only wrap the non-Latin portions

Examples:
- Greek: [[LANG:el]]Ευρωπαϊκή Ένωση[[/LANG]]
- Bulgarian: [[LANG:bg]]Европейски съюз[[/LANG]]
- Mixed: The term [[LANG:el]]Ελλάδα[[/LANG]] means Greece.

Note: For bilingual column layouts (e.g., FR/DE side-by-side), use column tags only - the column structure implies the language.
"""

FOOTNOTE_BLOCK = """
## Footnotes
This document contains footnotes. Apply these rules:
- Mark footnote references in main text with: [[SUP]]1[[/SUP]], [[SUP]]2[[/SUP]], etc.
- Wrap footnote content at bottom of page with: [[FN]]...[[/FN]]
- Include the footnote number in the FN tag

Example (in main text):
See regulation[[SUP]]1[[/SUP]] for details.

Example (at bottom of page):
[[FN]]1. OJ L 184, 17.7.1999, p. 23.[[/FN]]
[[FN]]2. See Commission Decision 2001/527/EC.[[/FN]]
"""

FORMS_BLOCK = """
## Forms
This document is a form with fields. Apply these rules:
- Transcribe both the printed field labels AND the filled content
- Mark handwritten entries with [[H]]...[[/H]]
- Preserve the structure: label followed by value
- Empty fields should show the label with no value

Example:
1. Applicant Name: [[H]]Schmidt, Johann[[/H]]
2. Address: [[H]]Hauptstraße 45, München[[/H]]
3. Date of Birth: [[H]]12/05/1952[[/H]]
4. Telephone: (empty)
"""

# =============================================================================
# HEADER/FOOTER BLOCK - Always included for multi-page awareness
# =============================================================================

HEADER_FOOTER_BLOCK = """
## Headers and Footers
If this page has running headers or footers:
- Wrap headers with: [[HEADER]]...[[/HEADER]]
- Wrap footers with: [[FOOTER]]...[[/FOOTER]]
- Include page numbers in footer tags

Example:
[[HEADER]]Official Journal of the European Communities[[/HEADER]]

[main content]

[[FOOTER]]L 123/45[[/FOOTER]]
"""

# =============================================================================
# OUTPUT FORMAT - Always included at the end
# =============================================================================

OUTPUT_FORMAT = """
## Output
Provide ONLY the transcription with appropriate tags. Do not include any explanation, commentary, or metadata. Start directly with the document content."""

# =============================================================================
# PROMPT BUILDER
# =============================================================================

def build_prompt(classification: dict) -> str:
    """
    Build OCR prompt by combining relevant blocks based on classification.
    
    Args:
        classification: Dict with boolean flags from classifier
        
    Returns:
        Complete prompt string
    """
    blocks = [BASE_PROMPT]
    
    # Add blocks based on classification
    if classification.get("multi_column", False):
        blocks.append(COLUMN_BLOCK)
    
    if classification.get("handwritten", False):
        blocks.append(HANDWRITTEN_BLOCK)
    
    if classification.get("has_tables", False):
        blocks.append(TABLE_BLOCK)
    
    if classification.get("has_stamps", False):
        blocks.append(STAMP_BLOCK)
    
    if classification.get("has_strikethrough", False):
        blocks.append(STRIKETHROUGH_BLOCK)
    
    if classification.get("poor_quality", False):
        blocks.append(POOR_QUALITY_BLOCK)
    
    if not classification.get("latin_script", True):  # Default True, add block if False
        blocks.append(NON_LATIN_BLOCK)
    
    if classification.get("has_footnotes", False):
        blocks.append(FOOTNOTE_BLOCK)
    
    if classification.get("has_forms", False):
        blocks.append(FORMS_BLOCK)
    
    # Always add header/footer awareness and output format
    blocks.append(HEADER_FOOTER_BLOCK)
    blocks.append(OUTPUT_FORMAT)
    
    return "\n".join(blocks)


def get_classify_prompt() -> str:
    """Return the classification prompt."""
    return CLASSIFY_PROMPT


# =============================================================================
# UTILITY: List all available blocks
# =============================================================================

BLOCK_REGISTRY = {
    "multi_column": COLUMN_BLOCK,
    "handwritten": HANDWRITTEN_BLOCK,
    "has_tables": TABLE_BLOCK,
    "has_stamps": STAMP_BLOCK,
    "has_strikethrough": STRIKETHROUGH_BLOCK,
    "poor_quality": POOR_QUALITY_BLOCK,
    "latin_script_false": NON_LATIN_BLOCK,
    "has_footnotes": FOOTNOTE_BLOCK,
    "has_forms": FORMS_BLOCK,
}


def list_blocks() -> list:
    """Return list of available block names."""
    return list(BLOCK_REGISTRY.keys())


def get_block(name: str) -> str:
    """Get a specific block by name."""
    return BLOCK_REGISTRY.get(name, "")


# =============================================================================
# DEBUG: Preview prompt for a classification
# =============================================================================

if __name__ == "__main__":
    # Example: Preview prompt for a complex document
    example_classification = {
        "multi_column": True,
        "has_tables": False,
        "handwritten": True,
        "has_stamps": True,
        "poor_quality": False,
        "has_strikethrough": False,
        "latin_script": True,
        "has_footnotes": True,
        "has_forms": False,
    }
    
    print("=" * 60)
    print("CLASSIFY PROMPT:")
    print("=" * 60)
    print(get_classify_prompt())
    
    print("\n" + "=" * 60)
    print("OCR PROMPT FOR EXAMPLE CLASSIFICATION:")
    print(f"Classification: {example_classification}")
    print("=" * 60)
    print(build_prompt(example_classification))