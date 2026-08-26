from pathlib import Path
import pytest
from app.pipeline.orchestrator import DocumentPipelineOrchestrator
from app.storage.object_store import LocalFileSystemStore


@pytest.mark.asyncio
async def test_pipeline_routing_digital_pdf(digital_pdf_path: Path, temp_storage: LocalFileSystemStore):
    orchestrator = DocumentPipelineOrchestrator(object_store=temp_storage)
    doc_id = "doc_digital_test"

    manifest = await orchestrator.process_document(document_id=doc_id, pdf_path=digital_pdf_path)

    assert manifest.document_id == doc_id
    assert manifest.page_count == 2
    assert len(manifest.pages) == 2

    # Check page 1 JSON content
    page1_data = await temp_storage.get_json(f"processed/{doc_id}/pages/0001.json")
    assert page1_data["extraction_method"] == "pdfplumber"
    assert page1_data["text_quality_score"] >= 0.70
    assert len(page1_data["regions"]) > 0

    # Ensure provenance is present on regions
    for r in page1_data["regions"]:
        assert r["provenance"]["document_id"] == doc_id
        assert r["provenance"]["source"] == "pdfplumber"
        assert len(r["bbox"]) == 4


@pytest.mark.asyncio
async def test_pipeline_routing_scanned_pdf(scanned_pdf_path: Path, temp_storage: LocalFileSystemStore):
    orchestrator = DocumentPipelineOrchestrator(object_store=temp_storage)
    doc_id = "doc_scanned_test"

    manifest = await orchestrator.process_document(document_id=doc_id, pdf_path=scanned_pdf_path)

    assert manifest.document_id == doc_id
    assert manifest.page_count == 1

    # Check page 1 JSON content
    page1_data = await temp_storage.get_json(f"processed/{doc_id}/pages/0001.json")
    assert page1_data["extraction_method"] in ("cnn_ocr", "baseline_ocr")
    assert page1_data["text_quality_score"] < 0.70
    assert len(page1_data["regions"]) > 0

    for r in page1_data["regions"]:
        assert r["provenance"]["document_id"] == doc_id
        assert r["provenance"]["extraction_method"] in ("cnn_ocr", "baseline_ocr")
