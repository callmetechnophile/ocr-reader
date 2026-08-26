import json
import shutil
from pathlib import Path
import numpy as np
import pymupdf
import pytest
from app.pipeline.orchestrator import DocumentPipelineOrchestrator, calculate_file_sha256
from app.schemas.document import DocumentStatus


def create_mock_textbook_pdf(path: Path, num_pages: int = 4) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open()

    for i in range(1, num_pages + 1):
        page = doc.new_page(width=612, height=792)
        if i == 1:
            page.insert_text((72, 100), "Chapter 1: Foundations of Computing", fontsize=18)
            page.insert_text((72, 140), "1.1 Computer Architecture Basics", fontsize=14)
            page.insert_text((72, 180), "This chapter explores memory hierarchies and arithmetic logic units.", fontsize=11)
        elif i == 2:
            page.insert_text((72, 100), "1.2 Instruction Set Architecture", fontsize=14)
            page.insert_text((72, 140), "Modern processors execute RISC and CISC instruction formats.", fontsize=11)
        elif i == 3:
            page.insert_text((72, 100), "Chapter 2: Memory Systems", fontsize=18)
            page.insert_text((72, 140), "2.1 Cache Memory and Coherence", fontsize=14)
            page.insert_text((72, 180), "Cache levels L1, L2, and L3 reduce main memory access latency.", fontsize=11)
        else:
            page.insert_text((72, 100), "2.2 Virtual Memory Management", fontsize=14)
            page.insert_text((72, 140), "Page tables translate virtual addresses to physical frames.", fontsize=11)

    doc.save(str(path))
    doc.close()
    return path


@pytest.mark.asyncio
async def test_end_to_end_digital_textbook_pipeline(tmp_path: Path):
    pdf_path = tmp_path / "engineering_textbook.pdf"
    create_mock_textbook_pdf(pdf_path, num_pages=4)
    out_dir = tmp_path / "processed_book"

    orchestrator = DocumentPipelineOrchestrator()
    manifest = await orchestrator.process_document(
        pdf_path=pdf_path,
        output_dir=out_dir,
        debug=True,
    )

    # 1. Manifest
    assert manifest.page_count == 4
    assert manifest.processed_pages == 4
    assert manifest.failed_pages == 0
    assert manifest.chapters == 2
    assert manifest.sections >= 2
    assert manifest.chunks >= 2
    assert not manifest.partial_run

    # 2. Output files exist
    assert (out_dir / "metadata.json").exists()
    assert (out_dir / "profile.json").exists()
    assert (out_dir / "manifest.json").exists()
    assert (out_dir / "job.json").exists()
    assert (out_dir / "report.json").exists()
    assert (out_dir / "performance.json").exists()

    # 3. Pages directory
    pages_dir = out_dir / "pages"
    assert (pages_dir / "0001.json").exists()
    assert (pages_dir / "0004.json").exists()

    # 4. Chapters directory
    chapters_dir = out_dir / "chapters"
    assert (chapters_dir / "ch_001.json").exists()
    assert (chapters_dir / "ch_002.json").exists()

    # 5. Chunks directory
    chunks_dir = out_dir / "chunks"
    assert (chunks_dir / "ch_001.jsonl").exists()
    assert (chunks_dir / "ch_002.jsonl").exists()

    # 6. Debug directory
    debug_dir = out_dir / "debug"
    assert (debug_dir / "page_0001.png").exists()
    assert (debug_dir / "page_0001_text.txt").exists()


@pytest.mark.asyncio
async def test_content_based_document_id_and_renaming(tmp_path: Path):
    pdf1 = tmp_path / "original_book.pdf"
    create_mock_textbook_pdf(pdf1, num_pages=2)

    pdf2 = tmp_path / "renamed_copy.pdf"
    shutil.copy(pdf1, pdf2)

    hash1 = calculate_file_sha256(pdf1)
    hash2 = calculate_file_sha256(pdf2)

    assert hash1 == hash2
    doc_id1 = f"doc_{hash1[:16]}"
    doc_id2 = f"doc_{hash2[:16]}"
    assert doc_id1 == doc_id2


@pytest.mark.asyncio
async def test_partial_run_page_range(tmp_path: Path):
    pdf_path = tmp_path / "large_textbook.pdf"
    create_mock_textbook_pdf(pdf_path, num_pages=4)
    out_dir = tmp_path / "processed_partial"

    orchestrator = DocumentPipelineOrchestrator()
    manifest = await orchestrator.process_document(
        pdf_path=pdf_path,
        output_dir=out_dir,
        start_page=2,
        end_page=3,
    )

    assert manifest.page_count == 4
    assert manifest.processed_pages == 2
    assert manifest.partial_run is True
    assert manifest.page_range == [2, 3]
    assert (out_dir / "pages" / "0002.json").exists()
    assert (out_dir / "pages" / "0003.json").exists()
    assert not (out_dir / "pages" / "0001.json").exists()


@pytest.mark.asyncio
async def test_resumability_and_force(tmp_path: Path):
    pdf_path = tmp_path / "resume_textbook.pdf"
    create_mock_textbook_pdf(pdf_path, num_pages=2)
    out_dir = tmp_path / "processed_resume"

    orchestrator = DocumentPipelineOrchestrator()
    # First run
    await orchestrator.process_document(pdf_path=pdf_path, output_dir=out_dir)

    page1_file = out_dir / "pages" / "0001.json"
    mtime_initial = page1_file.stat().st_mtime

    # Second run without force -> should not overwrite page 1
    await orchestrator.process_document(pdf_path=pdf_path, output_dir=out_dir, force=False)
    assert page1_file.stat().st_mtime == mtime_initial

    # Third run with force -> reprocesses page 1
    await orchestrator.process_document(pdf_path=pdf_path, output_dir=out_dir, force=True)
    assert page1_file.exists()
