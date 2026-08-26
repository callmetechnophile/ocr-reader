import json
import shutil
from pathlib import Path
import pytest
import pymupdf

from app.pipeline.orchestrator import DocumentPipelineOrchestrator


def create_minimal_pdf(file_path: Path, text: str = "Test textbook content for path test.") -> Path:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 72), text, fontsize=14)
    doc.save(str(file_path))
    doc.close()
    return file_path


@pytest.mark.asyncio
async def test_pdf_in_arbitrary_directory_and_toon_output(tmp_path: Path):
    orchestrator = DocumentPipelineOrchestrator()

    pdf_dir = tmp_path / "custom_input" / "subfolder"
    pdf_path = create_minimal_pdf(pdf_dir / "physics_vol1.pdf")

    processed_out = tmp_path / "custom_processed"
    toon_out_dir = tmp_path / "custom_toon"

    manifest = await orchestrator.process_document(
        pdf_path=pdf_path,
        output_dir=processed_out,
        generate_toon=True,
        toon_output_dir=toon_out_dir,
    )

    expected_toon = toon_out_dir / "physics_vol1.toon"
    assert expected_toon.exists()
    assert expected_toon.stat().st_size > 0
    assert manifest.toon["enabled"] is True
    assert manifest.toon["path"] == str(expected_toon.resolve())

    # Verify report.json contains toon fields
    report_file = processed_out / "report.json"
    assert report_file.exists()
    with open(report_file, "r", encoding="utf-8") as f:
        report_data = json.load(f)
    assert report_data["toon_enabled"] is True
    assert report_data["toon_output_path"] == str(expected_toon.resolve())
    assert report_data["toon_size_bytes"] > 0


@pytest.mark.asyncio
async def test_toon_directory_auto_creation(tmp_path: Path):
    orchestrator = DocumentPipelineOrchestrator()
    pdf_path = create_minimal_pdf(tmp_path / "biology.pdf")

    non_existent_toon_dir = tmp_path / "nested" / "auto_created" / "toon_dir"
    assert not non_existent_toon_dir.exists()

    await orchestrator.process_document(
        pdf_path=pdf_path,
        output_dir=tmp_path / "proc_bio",
        generate_toon=True,
        toon_output_dir=non_existent_toon_dir,
    )

    assert non_existent_toon_dir.exists()
    expected_toon = non_existent_toon_dir / "biology.toon"
    assert expected_toon.exists()


@pytest.mark.asyncio
async def test_paths_containing_spaces(tmp_path: Path):
    orchestrator = DocumentPipelineOrchestrator()

    spaced_pdf_dir = tmp_path / "My Documents" / "Engineering Books"
    pdf_path = create_minimal_pdf(spaced_pdf_dir / "Engineering Mathematics Vol 1.pdf")

    spaced_toon_dir = tmp_path / "My Output" / "TOON Exports"
    spaced_proc_dir = tmp_path / "My Output" / "Processed Data"

    manifest = await orchestrator.process_document(
        pdf_path=pdf_path,
        output_dir=spaced_proc_dir,
        generate_toon=True,
        toon_output_dir=spaced_toon_dir,
    )

    expected_toon = spaced_toon_dir / "Engineering Mathematics Vol 1.toon"
    assert expected_toon.exists()
    assert manifest.toon["path"] == str(expected_toon.resolve())


@pytest.mark.asyncio
async def test_complex_multi_dot_pdf_stem(tmp_path: Path):
    orchestrator = DocumentPipelineOrchestrator()
    pdf_path = create_minimal_pdf(tmp_path / "book.final.version.pdf")
    toon_dir = tmp_path / "toons"

    await orchestrator.process_document(
        pdf_path=pdf_path,
        output_dir=tmp_path / "proc_multidot",
        generate_toon=True,
        toon_output_dir=toon_dir,
    )

    expected_toon = toon_dir / "book.final.version.toon"
    assert expected_toon.exists()
    assert not (toon_dir / "book.final.version_parsed.toon").exists()


@pytest.mark.asyncio
async def test_toon_overwrite_protection_without_and_with_force(tmp_path: Path):
    orchestrator = DocumentPipelineOrchestrator()
    pdf_path = create_minimal_pdf(tmp_path / "chemistry.pdf")
    toon_dir = tmp_path / "toons_overwrite"
    toon_dir.mkdir(parents=True)
    existing_toon = toon_dir / "chemistry.toon"

    # Create dummy existing file
    with open(existing_toon, "w", encoding="utf-8") as f:
        f.write("ORIGINAL_EXISTING_CONTENT")

    # 1. Run without force -> should preserve existing content
    await orchestrator.process_document(
        pdf_path=pdf_path,
        output_dir=tmp_path / "proc_chem_1",
        generate_toon=True,
        toon_output_dir=toon_dir,
        force=False,
    )

    with open(existing_toon, "r", encoding="utf-8") as f:
        content_after_no_force = f.read()
    assert content_after_no_force == "ORIGINAL_EXISTING_CONTENT"

    # 2. Run with force=True -> should overwrite with valid JSON
    await orchestrator.process_document(
        pdf_path=pdf_path,
        output_dir=tmp_path / "proc_chem_2",
        generate_toon=True,
        toon_output_dir=toon_dir,
        force=True,
    )

    with open(existing_toon, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["format"] == "TOON_V1"


@pytest.mark.asyncio
async def test_input_pdf_and_toon_directory_same_location(tmp_path: Path):
    orchestrator = DocumentPipelineOrchestrator()
    books_dir = tmp_path / "shared_folder"
    pdf_path = create_minimal_pdf(books_dir / "stuvia.pdf")

    # TOON output directory is the same directory as PDF
    manifest = await orchestrator.process_document(
        pdf_path=pdf_path,
        output_dir=tmp_path / "proc_stuvia",
        generate_toon=True,
        toon_output_dir=books_dir,
    )

    expected_toon = books_dir / "stuvia.toon"
    assert expected_toon.exists()
    # Ensure input PDF was not modified or corrupted
    assert pdf_path.exists()
    assert pdf_path.stat().st_size > 0
    assert manifest.toon["path"] == str(expected_toon.resolve())


@pytest.mark.asyncio
async def test_toon_disabled_by_default(tmp_path: Path):
    orchestrator = DocumentPipelineOrchestrator()
    pdf_path = create_minimal_pdf(tmp_path / "sample.pdf")
    proc_out = tmp_path / "proc_default"

    manifest = await orchestrator.process_document(
        pdf_path=pdf_path,
        output_dir=proc_out,
        generate_toon=False,
        toon_output_dir=None,
    )

    assert manifest.toon["enabled"] is False
    assert manifest.toon["path"] is None
    assert not list(proc_out.glob("*.toon"))


@pytest.mark.asyncio
async def test_custom_debug_output_directory(tmp_path: Path):
    orchestrator = DocumentPipelineOrchestrator()
    pdf_path = create_minimal_pdf(tmp_path / "debug_test.pdf")
    custom_debug = tmp_path / "custom_debug_folder"

    await orchestrator.process_document(
        pdf_path=pdf_path,
        output_dir=tmp_path / "proc_debug_test",
        debug=True,
        debug_output_dir=custom_debug,
    )

    assert custom_debug.exists()
