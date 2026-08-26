from pathlib import Path
import pdfplumber
import pytest
from app.extractors.pdfplumber_extractor import PDFPlumberExtractor


def test_pdfplumber_text_and_coordinates(digital_pdf_path: Path):
    extractor = PDFPlumberExtractor()
    with pdfplumber.open(str(digital_pdf_path)) as pdf:
        page = pdf.pages[0]
        extraction = extractor.extract(page, page_number=1, document_id="doc_test")

        assert extraction.page_number == 1
        assert extraction.width == 612.0
        assert extraction.height == 792.0
        assert extraction.method == "pdfplumber"
        assert len(extraction.words) > 0
        assert len(extraction.chars) > 0
        assert len(extraction.blocks) > 0

        # Verify coordinates structure
        for block in extraction.blocks:
            bbox = block["bbox"]
            assert len(bbox) == 4
            x0, y0, x1, y1 = bbox
            assert 0 <= x0 <= x1 <= 612.0
            assert 0 <= y0 <= y1 <= 792.0

        # Check font metadata preservation
        fonts = [w.get("fontname") for w in extraction.words if w.get("fontname")]
        assert len(fonts) > 0


def test_pdfplumber_two_column_page(digital_pdf_path: Path):
    extractor = PDFPlumberExtractor()
    with pdfplumber.open(str(digital_pdf_path)) as pdf:
        page = pdf.pages[1]  # Page 2 has 2 columns
        extraction = extractor.extract(page, page_number=2, document_id="doc_test")

        # Words should exist in both left and right halves of the page
        left_words = [w for w in extraction.words if w["x0"] < 300]
        right_words = [w for w in extraction.words if w["x0"] > 300]

        assert len(left_words) > 0
        assert len(right_words) > 0
