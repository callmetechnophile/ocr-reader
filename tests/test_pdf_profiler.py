from pathlib import Path
import pymupdf
import pytest
from app.pipeline.analyzer import PDFPageAnalyzer
from app.schemas.page import PageRoute


def create_sample_pdf(path: Path, num_pages: int = 3) -> Path:
    doc = pymupdf.open()
    for i in range(num_pages):
        page = doc.new_page(width=612, height=792)
        page.insert_text(
            (72, 100),
            f"Chapter {i+1}: Advanced Electronic Systems\n\n"
            f"This is page {i+1} containing rich textbook content with valid sentences and numbers like 42 V and 100 kHz.\n"
            "Carrier dynamics in semiconductors dominate device characteristics.",
            fontsize=12,
        )
    doc.save(str(path))
    doc.close()
    return path


def test_pdf_profiler(tmp_path: Path):
    pdf_file = tmp_path / "sample_book.pdf"
    create_sample_pdf(pdf_file, num_pages=3)

    analyzer = PDFPageAnalyzer()
    profile = analyzer.profile_document(pdf_file)

    assert profile.page_count == 3
    assert profile.text_layer_pages == 3
    assert profile.poor_text_pages == 0
    assert profile.zero_text_pages == 0
    assert profile.average_characters_per_page > 50
    assert profile.average_words_per_page > 10
    assert len(profile.page_dimensions) == 3
    assert profile.file_size > 0
