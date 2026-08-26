from pathlib import Path
import pdfplumber
import pytest
from app.pipeline.analyzer import PDFPageAnalyzer


def test_analyzer_digital_pdf(digital_pdf_path: Path):
    analyzer = PDFPageAnalyzer(quality_threshold=0.70)
    with pdfplumber.open(str(digital_pdf_path)) as pdf:
        page = pdf.pages[0]
        report = analyzer.analyze(page, page_number=1)

        assert report.text_layer_available is True
        assert report.text_character_count > 50
        assert report.word_count > 10
        assert report.printable_ratio > 0.95
        assert report.garbage_ratio == 0.0
        assert report.text_quality_score >= 0.70
        assert report.should_use_text_layer is True


def test_analyzer_scanned_pdf(scanned_pdf_path: Path):
    analyzer = PDFPageAnalyzer(quality_threshold=0.70)
    with pdfplumber.open(str(scanned_pdf_path)) as pdf:
        page = pdf.pages[0]
        report = analyzer.analyze(page, page_number=1)

        assert report.text_layer_available is False
        assert report.text_character_count == 0
        assert report.word_count == 0
        assert report.text_quality_score == 0.0
        assert report.should_use_text_layer is False


def test_analyzer_corrupted_garbage_text(corrupted_text_pdf_path: Path):
    analyzer = PDFPageAnalyzer(quality_threshold=0.70)
    with pdfplumber.open(str(corrupted_text_pdf_path)) as pdf:
        page = pdf.pages[0]
        report = analyzer.analyze(page, page_number=1)

        assert report.text_layer_available is True
        assert report.text_quality_score < 0.70
        assert report.should_use_text_layer is False
