import warnings
from pathlib import Path
import pymupdf
import pytest
from app.pipeline.orchestrator import DocumentPipelineOrchestrator
from app.pipeline.renderer import PDFPageRenderer


def test_1_pdf_opening(tmp_path: Path):
    """Test 1 — PDF opening: open valid PDF with pymupdf.open, verify page count and page access."""
    pdf_file = tmp_path / "opening_test.pdf"
    with pymupdf.open() as doc:
        p = doc.new_page(width=612, height=792)
        p.insert_text(pymupdf.Point(72, 72), "Chapter 1: Opening Test", fontsize=14)
        doc.save(str(pdf_file))

    with pymupdf.open(str(pdf_file)) as doc:
        assert len(doc) == 1
        page = doc[0]
        assert page.number == 0
        assert page.rect.width == 612.0
        assert page.rect.height == 792.0


def test_2_rendering(tmp_path: Path):
    """Test 2 — Rendering: render one page, verify pixmap, image file existence and non-zero dimensions."""
    pdf_file = tmp_path / "rendering_test.pdf"
    with pymupdf.open() as doc:
        p = doc.new_page(width=612, height=792)
        p.insert_text(pymupdf.Point(72, 100), "Rendering Pixmap Test", fontsize=16)
        doc.save(str(pdf_file))

    renderer = PDFPageRenderer(dpi=150)
    result = renderer.render_page(pdf_file, 0)

    assert result.dpi == 150
    assert result.pixel_width > 0
    assert result.pixel_height > 0
    assert result.original_image is not None
    assert result.processed_image is not None
    assert result.processed_image.shape[0] > 0
    assert result.processed_image.shape[1] > 0


def test_3_metadata(tmp_path: Path):
    """Test 3 — Metadata: verify PDF metadata access and manipulation using PyMuPDF."""
    pdf_file = tmp_path / "metadata_test.pdf"
    with pymupdf.open() as doc:
        doc.set_metadata({
            "title": "Digital Textbook Systems",
            "author": "Engineering Author",
            "subject": "Computer Science & OCR",
        })
        p = doc.new_page(width=612, height=792)
        p.insert_text((72, 72), "Content with metadata", fontsize=12)
        doc.save(str(pdf_file))

    with pymupdf.open(str(pdf_file)) as doc:
        meta = doc.metadata
        assert meta["title"] == "Digital Textbook Systems"
        assert meta["author"] == "Engineering Author"
        assert meta["subject"] == "Computer Science & OCR"


def test_4_large_document_handle_lifecycle(tmp_path: Path):
    """Test 4 — Large-document behavior: open multi-page PDF without leaving file handles open."""
    pdf_file = tmp_path / "multi_page_test.pdf"
    with pymupdf.open() as doc:
        for i in range(10):
            p = doc.new_page(width=612, height=792)
            p.insert_text((72, 100), f"Page {i+1} Content", fontsize=12)
        doc.save(str(pdf_file))

    # Open and verify context manager clean close
    doc_ref = None
    with pymupdf.open(str(pdf_file)) as doc:
        doc_ref = doc
        assert len(doc) == 10
        for i in range(10):
            _ = doc[i]

    assert doc_ref.is_closed


@pytest.mark.asyncio
async def test_5_ocr_pipeline_pymupdf_integration(tmp_path: Path):
    """Test 5 — OCR pipeline: run orchestrator on test PDF and verify profiling, routing, extraction, JSON."""
    pdf_file = tmp_path / "pipeline_integration.pdf"
    with pymupdf.open() as doc:
        # Page 1
        p1 = doc.new_page(width=612, height=792)
        p1.insert_text((72, 100), "Chapter 1: Pipeline Verification", fontsize=18)
        p1.insert_text((72, 140), "1.1 Integration Checks", fontsize=14)
        p1.insert_text((72, 180), "This page verifies end-to-end processing with PyMuPDF.", fontsize=11)
        # Page 2
        p2 = doc.new_page(width=612, height=792)
        p2.insert_text((72, 100), "1.2 Second Section", fontsize=14)
        p2.insert_text((72, 140), "Verifying page normalization and chunking pipeline.", fontsize=11)
        doc.save(str(pdf_file))

    out_dir = tmp_path / "out_processed"
    orchestrator = DocumentPipelineOrchestrator()
    manifest = await orchestrator.process_document(
        pdf_path=pdf_file,
        output_dir=out_dir,
        debug=True,
    )

    assert manifest.page_count == 2
    assert manifest.processed_pages == 2
    assert manifest.failed_pages == 0
    assert manifest.chapters == 1
    assert (out_dir / "manifest.json").exists()
    assert (out_dir / "profile.json").exists()
    assert (out_dir / "report.json").exists()
    assert (out_dir / "pages" / "0001.json").exists()
    assert (out_dir / "pages" / "0002.json").exists()


def test_6_no_fitz_deprecation_warning(tmp_path: Path):
    """Verify that importing and running pipeline modules raises zero fitz deprecation warnings."""
    with warnings.catch_warnings(record=True) as recorded_warnings:
        warnings.simplefilter("always")

        from app.api.documents import router
        from app.main import app
        from app.pipeline.analyzer import PDFPageAnalyzer
        from app.pipeline.debug_visualizer import DebugVisualizer
        from app.pipeline.orchestrator import DocumentPipelineOrchestrator
        from app.pipeline.renderer import PDFPageRenderer

        pdf_path = tmp_path / "sample.pdf"
        with pymupdf.open() as doc:
            p = doc.new_page(width=612, height=792)
            p.insert_text((72, 72), "No fitz warning test", fontsize=12)
            doc.save(str(pdf_path))

        renderer = PDFPageRenderer(dpi=72)
        res = renderer.render_page(pdf_path, 0)
        assert res.processed_image is not None

        fitz_warnings = [
            w for w in recorded_warnings
            if "fitz" in str(w.message).lower()
        ]
        assert len(fitz_warnings) == 0, f"Found unexpected fitz deprecation warnings: {fitz_warnings}"
