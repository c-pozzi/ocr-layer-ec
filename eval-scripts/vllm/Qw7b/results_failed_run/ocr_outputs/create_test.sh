#!/bin/bash
# Run this on your instance to create the test files
# Usage: bash create_test_files.sh /path/to/ocr_outputs

OUTPUT_DIR="${1:-/home/ubuntu/ocr-evaluation-samples/results/ocr_outputs}"

echo "Creating test files in: $OUTPUT_DIR"

# Ground truth (raw)
cat > "$OUTPUT_DIR/unit_test_ground_truth.txt" << 'EOF'
The quick brown fox jumps over the lazy dog.
This is a test document for OCR evaluation.
Numbers: 12345 and symbols: @#$%
EOF

# Ground truth (normalized)
cat > "$OUTPUT_DIR/unit_test_ground_truth_norm.txt" << 'EOF'
the quick brown fox jumps over the lazy dog. this is a test document for ocr evaluation. numbers: 12345 and symbols: @#$%
EOF

# Qwen OCR (raw) - typo errors
cat > "$OUTPUT_DIR/unit_test_qwen_ocr.txt" << 'EOF'
The quikk brown fax jumps ovre the lasy dog.
This is a tset documnet for OCR evaluaton.
Numbers: 12345 and simbols: @#$%
EOF

# Qwen OCR (normalized)
cat > "$OUTPUT_DIR/unit_test_qwen_ocr_norm.txt" << 'EOF'
the quikk brown fax jumps ovre the lasy dog. this is a tset documnet for ocr evaluaton. numbers: 12345 and simbols: @#$%
EOF

# PDF OCR (raw) - OCR confusion errors (l→1, o→0)
cat > "$OUTPUT_DIR/unit_test_pdf_ocr.txt" << 'EOF'
The quick brown fox jumps over the lazy dog.
Th1s 1s a test d0cument f0r 0CR evaluat1on.
Numbers: I2345 and symb0ls: @#$%
EOF

# PDF OCR (normalized)
cat > "$OUTPUT_DIR/unit_test_pdf_ocr_norm.txt" << 'EOF'
the quick brown fox jumps over the lazy dog. th1s 1s a test d0cument f0r 0cr evaluat1on. numbers: i2345 and symb0ls: @#$%
EOF

echo "Done! Created 6 files:"
ls -la "$OUTPUT_DIR"/unit_test*