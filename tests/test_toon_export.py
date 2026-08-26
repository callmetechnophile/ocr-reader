import json
from pathlib import Path
import pytest
from app.formats.toon_writer import ToonValidator, ToonWriter
from app.pipeline.orchestrator import DocumentPipelineOrchestrator
from app.schemas.document import ChapterInfo, DocumentManifest, Section


def test_toon_writer_and_validator(tmp_path: Path):
    writer = ToonWriter()
    doc_id = "doc_test_toon"
    filename = "SampleTextbook.pdf"
    manifest = DocumentManifest(
        document_id=doc_id,
        filename=filename,
        page_count=2,
        processed_pages=2,
        failed_pages=0,
        chapters=1,
        sections=1,
        chunks=1,
        pages=["pages/0001.json", "pages/0002.json"],
    )

    chapters = [
        ChapterInfo(
            chapter_id="ch_001",
            number=1,
            title="Chapter 1: Intro",
            page_start=1,
            page_end=2,
            sections=[Section(section_id="sec_001", title="1.1 Section", page_start=1, page_end=1)],
        )
    ]

    chunks = [
        {
            "chunk_id": "chk_001",
            "document_id": doc_id,
            "chapter_id": "ch_001",
            "section_id": "sec_001",
            "text": "Canonical TOON export content test.",
            "token_count": 80,
            "page_start": 1,
            "page_end": 1,
        }
    ]

    output_toon = tmp_path / "SampleTextbook_parsed.toon"
    written_path = writer.write(
        document_id=doc_id,
        filename=filename,
        manifest=manifest,
        pages=[],
        chapters=chapters,
        chunks=chunks,
        output_path=output_toon,
    )

    assert written_path.exists()
    assert written_path.stat().st_size > 0

    with open(written_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["format"] == "TOON_V1"
    assert data["document_id"] == doc_id
    assert data["filename"] == filename
    assert len(data["structure"]["chapters"]) == 1

    validation_json = tmp_path / "toon_validation.json"
    val = ToonValidator.validate(
        toon_path=written_path,
        expected_document_id=doc_id,
        manifest=manifest,
        output_validation_path=validation_json,
    )

    assert val["valid"] is True
    assert val["pages"] == 2
    assert val["chapters"] == 1
    assert val["sections"] == 1
    assert val["chunks"] == 1
    assert validation_json.exists()


@pytest.mark.asyncio
async def test_end_to_end_pipeline_with_toon(tmp_path: Path, digital_pdf_path: Path):
    orchestrator = DocumentPipelineOrchestrator()
    out_dir = tmp_path / "proc_toon_test"

    manifest = await orchestrator.process_document(
        pdf_path=digital_pdf_path,
        output_dir=out_dir,
        generate_toon=True,
    )

    expected_toon_name = f"{digital_pdf_path.stem}_parsed.toon"
    toon_file = out_dir / expected_toon_name
    assert toon_file.exists()
    assert (out_dir / "audit" / "toon_validation.json").exists()
    assert (out_dir / "audit" / "structural_validation.json").exists()
    assert (out_dir / "audit" / "structural_validation.md").exists()
