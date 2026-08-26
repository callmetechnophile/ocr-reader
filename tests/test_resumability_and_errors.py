from pathlib import Path
from unittest.mock import patch
import pymupdf
import pytest
from app.pipeline.orchestrator import DocumentPipelineOrchestrator
from app.schemas.document import DocumentStatus
from app.storage.object_store import LocalFileSystemStore


@pytest.mark.asyncio
async def test_resumability_skips_processed_pages(digital_pdf_path: Path, temp_storage: LocalFileSystemStore):
    orchestrator = DocumentPipelineOrchestrator(object_store=temp_storage)
    doc_id = "doc_resumable_test"

    # Pre-populate page 1 in object store
    dummy_page1 = {
        "page_id": f"{doc_id}_p0001",
        "document_id": doc_id,
        "page_number": 1,
        "width": 612.0,
        "height": 792.0,
        "extraction_method": "pdfplumber",
        "text_quality_score": 1.0,
        "regions": [],
        "metadata": {"pre_existing": True},
    }
    await temp_storage.put_json(f"processed/{doc_id}/pages/0001.json", dummy_page1)

    # Run orchestrator
    manifest = await orchestrator.process_document(document_id=doc_id, pdf_path=digital_pdf_path)

    assert manifest.page_count == 2
    assert len(manifest.pages) == 2

    # Verify page 1 was preserved without being overwritten
    loaded_page1 = await temp_storage.get_json(f"processed/{doc_id}/pages/0001.json")
    assert loaded_page1.get("metadata", {}).get("pre_existing") is True

    # Verify page 2 was newly generated
    loaded_page2 = await temp_storage.get_json(f"processed/{doc_id}/pages/0002.json")
    assert loaded_page2["page_number"] == 2


@pytest.mark.asyncio
async def test_corrupted_page_error_isolation(digital_pdf_path: Path, temp_storage: LocalFileSystemStore):
    orchestrator = DocumentPipelineOrchestrator(object_store=temp_storage)
    doc_id = "doc_isolated_error_test"

    # Mock analyzer to raise error only on page 1
    original_analyze = orchestrator.analyzer.analyze

    def mock_analyze(page, page_num):
        if page_num == 1:
            raise RuntimeError("Simulated corrupt font rendering crash on page 1")
        return original_analyze(page, page_num)

    with patch.object(orchestrator.analyzer, "analyze", side_effect=mock_analyze):
        manifest = await orchestrator.process_document(document_id=doc_id, pdf_path=digital_pdf_path)

    # Document should not crash entirely; page 2 should succeed
    assert manifest.metadata["pages_processed"] == 1
    assert manifest.metadata["pages_failed"] == 1
    assert len(manifest.metadata["errors"]) == 1
    assert manifest.metadata["errors"][0]["page"] == 1

    # Page 2 exists in storage
    assert await temp_storage.exists(f"processed/{doc_id}/pages/0002.json")
