#!/usr/bin/env python3
"""
Unit tests for CER/WER evaluation with table-aware normalization.

Covers:
  - Text normalization stages (markup, tables, unicode, dashes)
  - CER/WER on synthetic examples illustrating different error types
  - Edge cases (empty, identical, near-blank, CER > 1.0)
  - Table formatting differences between GT and OCR
  - Real-world-like scenarios (French text, mixed languages)

Run:
    python -m pytest test_cer_eval.py -v
    python -m unittest test_cer_eval -v
"""

import unittest

from cer_eval import (
    compute_metrics,
    extract_table_text,
    normalize_full,
    normalize_text_core,
    strip_markup,
)


# ===================================================================
# 1. Normalization stages
# ===================================================================

class TestStripMarkup(unittest.TestCase):
    """Stage 1: annotation tags, code fences, HTML."""

    def test_annotation_tags_removed(self):
        text = "[[HEADER]]Title[[/HEADER]]\nBody text\n[[FOOTER]]page 1[[/FOOTER]]"
        result = strip_markup(text)
        self.assertNotIn("[[HEADER]]", result)
        self.assertNotIn("[[/FOOTER]]", result)
        self.assertIn("Title", result)
        self.assertIn("Body text", result)

    def test_code_fences_removed(self):
        text = "```markdown\nSome content\n```"
        result = strip_markup(text)
        self.assertNotIn("```", result)
        self.assertIn("Some content", result)

    def test_br_tags_become_space(self):
        text = "first<br>second<br/>third<BR>fourth"
        result = strip_markup(text)
        self.assertEqual(result, "first second third fourth")

    def test_illegible_tag(self):
        text = "visible [[ILLEGIBLE]] more text"
        result = strip_markup(text)
        self.assertEqual(result, "visible  more text")

    def test_column_tags(self):
        text = "[[C1]]column one[[/C1]] [[C2]]column two[[/C2]]"
        result = strip_markup(text)
        self.assertIn("column one", result)
        self.assertIn("column two", result)

    def test_no_markup_passthrough(self):
        text = "Plain text with no markup at all."
        self.assertEqual(strip_markup(text), text)


class TestExtractTableText(unittest.TestCase):
    """Stage 2: markdown table parsing → flat cell text."""

    def test_simple_table(self):
        table = (
            "| Name | Age | City |\n"
            "|------|-----|------|\n"
            "| Alice | 30 | Paris |\n"
            "| Bob | 25 | Rome |"
        )
        result = extract_table_text(table)
        lines = result.strip().split("\n")
        self.assertEqual(lines[0], "Name Age City")
        self.assertEqual(lines[1], "Alice 30 Paris")
        self.assertEqual(lines[2], "Bob 25 Rome")

    def test_separator_rows_dropped(self):
        table = "| A | B |\n|---|---|\n| 1 | 2 |"
        result = extract_table_text(table)
        self.assertNotIn("---", result)

    def test_empty_cells_skipped(self):
        """Empty cells should not produce extra whitespace."""
        table = "| hello | | world | | |"
        result = extract_table_text(table)
        self.assertEqual(result.strip(), "hello world")

    def test_non_table_text_preserved(self):
        text = "This is a paragraph.\nAnother line."
        self.assertEqual(extract_table_text(text), text)

    def test_mixed_table_and_text(self):
        text = (
            "Title here\n"
            "| Col1 | Col2 |\n"
            "|------|------|\n"
            "| a | b |\n"
            "Some footer text"
        )
        result = extract_table_text(text)
        self.assertIn("Title here", result)
        self.assertIn("Col1 Col2", result)
        self.assertIn("a b", result)
        self.assertIn("Some footer text", result)
        self.assertNotIn("---", result)

    def test_alignment_markers_in_separator(self):
        """Separators with :--- or :---: alignment markers."""
        table = "| Left | Center | Right |\n| :--- | :---: | ---: |\n| x | y | z |"
        result = extract_table_text(table)
        self.assertNotIn(":", result.split("\n")[1] if len(result.split("\n")) > 1 else "")
        self.assertIn("x y z", result)

    def test_column_count_mismatch(self):
        """Different column counts should still extract cell text."""
        gt_row = "| A | B | C | D |"
        ocr_row = "| A | B C | D |"
        gt_result = extract_table_text(gt_row).strip()
        ocr_result = extract_table_text(ocr_row).strip()
        # Both should extract the text content
        self.assertEqual(gt_result, "A B C D")
        self.assertEqual(ocr_result, "A B C D")


class TestNormalizeTextCore(unittest.TestCase):
    """Stage 3: unicode, case, dashes, whitespace."""

    def test_lowercase(self):
        self.assertEqual(normalize_text_core("HELLO World"), "hello world")

    def test_whitespace_collapse(self):
        self.assertEqual(normalize_text_core("a   b\n\nc   d"), "a b c d")

    def test_unicode_nfkc(self):
        # fi ligature → "fi"
        self.assertEqual(normalize_text_core("\ufb01nance"), "finance")

    def test_dash_normalization(self):
        # em-dash, en-dash, figure-dash all → hyphen
        for dash in ["\u2013", "\u2014", "\u2012", "\u2212"]:
            result = normalize_text_core(f"a{dash}b")
            self.assertEqual(result, "a-b", f"Failed for U+{ord(dash):04X}")

    def test_strip(self):
        self.assertEqual(normalize_text_core("  hello  "), "hello")


class TestNormalizeFull(unittest.TestCase):
    """Full pipeline: markup → tables → text normalization."""

    def test_full_pipeline(self):
        raw = (
            "```markdown\n"
            "[[HEADER]]Test[[/HEADER]]\n"
            "Some text.\n"
            "| Col1 | Col2 |\n"
            "|------|------|\n"
            "| val<br>ue | data |\n"
            "[[FOOTER]]123[[/FOOTER]]\n"
            "```"
        )
        result = normalize_full(raw)
        self.assertEqual(result, "test some text. col1 col2 val ue data 123")

    def test_identical_after_normalization(self):
        """GT and OCR with different formatting but same content."""
        gt = "| A | B |\n|---|---|\n| Hello | World |"
        ocr = "```markdown\n| A | B |\n| --- | --- |\n| Hello | World |\n```"
        self.assertEqual(normalize_full(gt), normalize_full(ocr))


# ===================================================================
# 2. CER metric — synthetic cases
# ===================================================================

class TestCERBasic(unittest.TestCase):
    """Character Error Rate on controlled examples."""

    def test_identical_texts(self):
        """Perfect OCR: CER = 0."""
        m = compute_metrics("hello world", "hello world")
        self.assertAlmostEqual(m["cer"], 0.0)

    def test_single_substitution(self):
        """One character wrong in 10-char string → CER ≈ 0.1."""
        # "abcdefghij" → "abcXefghij" (1 substitution in 10 chars)
        m = compute_metrics("abcdefghij", "abcXefghij")
        self.assertAlmostEqual(m["cer"], 0.1, places=2)

    def test_single_deletion(self):
        """One character missing → CER = 1/len(ref)."""
        m = compute_metrics("abcde", "abde")  # 'c' deleted
        self.assertAlmostEqual(m["cer"], 0.2, places=2)

    def test_single_insertion(self):
        """One extra character → CER = 1/len(ref)."""
        m = compute_metrics("abcde", "abcXde")  # 'X' inserted
        self.assertAlmostEqual(m["cer"], 0.2, places=2)

    def test_completely_wrong(self):
        """Entirely different text → CER = 1.0 (if same length)."""
        m = compute_metrics("aaaa", "bbbb")
        self.assertAlmostEqual(m["cer"], 1.0)

    def test_empty_reference(self):
        """Empty reference, non-empty hypothesis → CER = 1.0."""
        m = compute_metrics("", "something")
        self.assertEqual(m["cer"], 1.0)

    def test_both_empty(self):
        """Both empty → CER = 0."""
        m = compute_metrics("", "")
        self.assertEqual(m["cer"], 0.0)

    def test_empty_hypothesis(self):
        """Non-empty reference, empty hypothesis → CER = 1.0."""
        m = compute_metrics("hello", "")
        self.assertAlmostEqual(m["cer"], 1.0)

    def test_cer_above_one(self):
        """Hypothesis much longer than reference → CER > 1.0."""
        # ref=4 chars, hyp has 4 substitutions + many insertions
        m = compute_metrics("ab", "xyzxyzxyz")
        self.assertGreater(m["cer"], 1.0)

    def test_case_matters_before_normalization(self):
        """Raw CER is case-sensitive (normalization lowercases)."""
        m = compute_metrics("hello", "HELLO")
        self.assertGreater(m["cer"], 0.0)  # All 5 chars differ


class TestCERNormalized(unittest.TestCase):
    """CER after full normalization pipeline — real-world-like."""

    def _cer(self, gt_raw: str, ocr_raw: str) -> float:
        return compute_metrics(normalize_full(gt_raw), normalize_full(ocr_raw))["cer"]

    def test_case_insensitive(self):
        """After normalization, case doesn't matter."""
        self.assertAlmostEqual(self._cer("Hello World", "hello world"), 0.0)

    def test_whitespace_insensitive(self):
        """Extra whitespace, newlines don't affect CER."""
        self.assertAlmostEqual(self._cer("hello world", "hello   \n  world"), 0.0)

    def test_table_formatting_insensitive(self):
        """Same cell content, different table formatting → CER ≈ 0."""
        gt = "| Name | Age |\n|------|-----|\n| Alice | 30 |"
        ocr = "Name Age\nAlice 30"
        self.assertAlmostEqual(self._cer(gt, ocr), 0.0)

    def test_markup_insensitive(self):
        """Annotation tags don't affect CER."""
        gt = "[[HEADER]]Title[[/HEADER]]\nBody text"
        ocr = "Title\nBody text"
        self.assertAlmostEqual(self._cer(gt, ocr), 0.0)

    def test_dash_variants_insensitive(self):
        """Different dash types normalize to same."""
        gt = "pages 10\u201320"  # en-dash
        ocr = "pages 10-20"  # hyphen
        self.assertAlmostEqual(self._cer(gt, ocr), 0.0)

    def test_br_tag_insensitive(self):
        """<br> in table cells vs line breaks."""
        gt = "| first<br>second |"
        ocr = "| first second |"
        self.assertAlmostEqual(self._cer(gt, ocr), 0.0)

    def test_ocr_typo_in_table(self):
        """One character wrong inside a table cell."""
        gt = "| Allemagne | 442,8 |\n|---|---|\n| Belgique | 133,2 |"
        ocr = "| Allemagne | 442,8 |\n|---|---|\n| Belqique | 133,2 |"
        cer = self._cer(gt, ocr)
        # Only 1 char wrong ('g' → 'q') in ~30 chars
        self.assertGreater(cer, 0.0)
        self.assertLess(cer, 0.1)

    def test_missing_table_row(self):
        """OCR misses an entire table row."""
        gt = "| A | 1 |\n| B | 2 |\n| C | 3 |"
        ocr = "| A | 1 |\n| C | 3 |"
        cer = self._cer(gt, ocr)
        self.assertGreater(cer, 0.15)  # Missing "B 2" ≈ 1/3 of content

    def test_extra_hallucinated_text(self):
        """OCR adds text not in ground truth."""
        gt = "Simple page."
        ocr = "Simple page. EXTRA HALLUCINATED TEXT HERE."
        cer = self._cer(gt, ocr)
        self.assertGreater(cer, 0.5)

    def test_near_blank_page(self):
        """Very short GT, longer OCR → CER >> 1.0."""
        gt = "0030"
        ocr = "[[HEADER]]Archives[[/HEADER]]\n[[FOOTER]]0030[[/FOOTER]]\n801"
        cer = self._cer(gt, ocr)
        self.assertGreater(cer, 1.0)

    def test_french_accents(self):
        """French accented characters must match exactly."""
        gt = "Communaut\u00e9 europ\u00e9enne"
        ocr = "Communaute europeenne"  # missing accents
        cer = self._cer(gt, ocr)
        self.assertGreater(cer, 0.0)
        # 2 accent chars wrong in ~22 chars
        self.assertLess(cer, 0.15)

    def test_column_reorder_penalty(self):
        """Swapped columns should produce errors (not silently pass)."""
        gt = "| Name | Age |\n|---|---|\n| Alice | 30 |"
        ocr = "| Age | Name |\n|---|---|\n| 30 | Alice |"
        cer = self._cer(gt, ocr)
        # After table extraction: GT="name age alice 30", OCR="age name 30 alice"
        self.assertGreater(cer, 0.0)


# ===================================================================
# 3. WER metric — synthetic cases
# ===================================================================

class TestWERBasic(unittest.TestCase):
    """Word Error Rate on controlled examples."""

    def test_identical_texts(self):
        m = compute_metrics("hello world", "hello world")
        self.assertAlmostEqual(m["wer"], 0.0)

    def test_one_word_wrong(self):
        """1 word wrong in 4 → WER = 0.25."""
        m = compute_metrics("the cat sat down", "the dog sat down")
        self.assertAlmostEqual(m["wer"], 0.25)

    def test_missing_word(self):
        """1 word deleted in 4 → WER = 0.25."""
        m = compute_metrics("the cat sat down", "the cat down")
        self.assertAlmostEqual(m["wer"], 0.25)

    def test_extra_word(self):
        """1 word inserted in 4-word ref → WER = 0.25."""
        m = compute_metrics("the cat sat down", "the big cat sat down")
        self.assertAlmostEqual(m["wer"], 0.25)

    def test_all_wrong(self):
        m = compute_metrics("one two three", "four five six")
        self.assertAlmostEqual(m["wer"], 1.0)

    def test_empty_reference(self):
        m = compute_metrics("", "some words")
        self.assertEqual(m["wer"], 1.0)

    def test_both_empty(self):
        m = compute_metrics("", "")
        self.assertEqual(m["wer"], 0.0)


class TestWERNormalized(unittest.TestCase):
    """WER after full normalization — illustrating word-level robustness."""

    def _wer(self, gt_raw: str, ocr_raw: str) -> float:
        return compute_metrics(normalize_full(gt_raw), normalize_full(ocr_raw))["wer"]

    def test_cer_vs_wer_single_char_typo(self):
        """A single-char typo costs 1 word in WER but few chars in CER."""
        gt = "the quick brown fox jumps"
        ocr = "the quikc brown fox jumps"  # typo in 'quick'
        m = compute_metrics(normalize_full(gt), normalize_full(ocr))
        self.assertAlmostEqual(m["wer"], 0.2)  # 1/5 words
        self.assertLess(m["cer"], 0.1)  # only ~2 chars off

    def test_number_transcription_error(self):
        """Numeric OCR errors: 442,8 → 442,6 — one word wrong."""
        gt = "| Allemagne | 442,8 |"
        ocr = "| Allemagne | 442,6 |"
        wer = self._wer(gt, ocr)
        self.assertAlmostEqual(wer, 0.5, places=1)  # 1 of 2 content words


# ===================================================================
# 4. CER vs WER comparison — illustrative
# ===================================================================

class TestCERvsWER(unittest.TestCase):
    """Show how CER and WER capture different aspects of quality."""

    def _metrics(self, gt: str, ocr: str) -> dict:
        return compute_metrics(normalize_full(gt), normalize_full(ocr))

    def test_many_small_typos(self):
        """Many 1-char typos: high WER (every word wrong), moderate CER."""
        gt = "cat dog fox hen cow pig"   # 6 words
        ocr = "cot dag fax han caw pug"  # 6 words, each 1 char off
        m = self._metrics(gt, ocr)
        self.assertAlmostEqual(m["wer"], 1.0)   # every word wrong
        self.assertLess(m["cer"], 0.35)          # only 6 chars off in ~23

    def test_one_long_word_wrong(self):
        """One long word wrong: low WER, but higher CER."""
        gt = "the characterization was good"
        ocr = "the XXXXXXXXXXXXXXX was good"  # 1 word garbled
        m = self._metrics(gt, ocr)
        self.assertAlmostEqual(m["wer"], 0.25)  # 1/4 words
        self.assertGreater(m["cer"], 0.3)        # many chars differ

    def test_word_boundary_shift(self):
        """Merged/split words: affects both CER and WER differently."""
        gt = "hello world test"
        ocr = "helloworld test"  # words merged
        m = self._metrics(gt, ocr)
        self.assertLess(m["cer"], 0.1)   # just a missing space
        self.assertGreater(m["wer"], 0.3)  # word count mismatch


# ===================================================================
# Run
# ===================================================================

# Collect all test cases for programmatic access (used by Streamlit app)
ALL_TEST_CLASSES = [
    TestStripMarkup,
    TestExtractTableText,
    TestNormalizeTextCore,
    TestNormalizeFull,
    TestCERBasic,
    TestCERNormalized,
    TestWERBasic,
    TestWERNormalized,
    TestCERvsWER,
]


def run_tests_programmatic() -> tuple[unittest.TestResult, str]:
    """Run all tests and return (result, text_output) for embedding in apps."""
    import io
    stream = io.StringIO()
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    for cls in ALL_TEST_CLASSES:
        suite.addTests(loader.loadTestsFromTestCase(cls))
    runner = unittest.TextTestRunner(stream=stream, verbosity=2)
    result = runner.run(suite)
    return result, stream.getvalue()


if __name__ == "__main__":
    unittest.main()
