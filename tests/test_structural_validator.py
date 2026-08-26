from pathlib import Path
import pytest
from app.schemas.document import ChapterInfo, DocumentManifest, Section
from app.schemas.page import PageSchema
from app.schemas.region import Provenance, Region, RegionType
from app.structure.validator import DocumentStructuralValidator


def test_structural_validator_basic_consistency(tmp_path: Path):
    validator = DocumentStructuralValidator()

    # Create dummy pages
    prov1 = Provenance(
        document_id="doc_test_audit",
        page_number=1,
        region_id="r001",
        bbox=[72.0, 72.0, 500.0, 100.0],
        source="pdfplumber",
        extraction_method="pdfplumber",
        confidence=0.99,
    )
    r1 = Region(
        region_id="r001",
        type=RegionType.HEADING,
        bbox=[72.0, 72.0, 500.0, 100.0],
        text="Chapter 1: Foundations",
        source="pdfplumber",
        reading_order=1,
        provenance=prov1,
    )
    p1 = PageSchema(
        page_id="doc_test_audit_p0001",
        document_id="doc_test_audit",
        page_number=1,
        width=612.0,
        height=792.0,
        extraction_method="pdfplumber",
        text_quality_score=0.99,
        regions=[r1],
    )

    chapters = [
        ChapterInfo(
            chapter_id="ch_001",
            number=1,
            title="Chapter 1: Foundations",
            page_start=1,
            page_end=1,
            sections=[Section(section_id="sec_001", title="1.1 Intro", page_start=1, page_end=1)],
        )
    ]

    chunks = [
        {
            "chunk_id": "chk_001",
            "document_id": "doc_test_audit",
            "chapter_id": "ch_001",
            "section_id": "sec_001",
            "text": "Introduction chunk text for testing.",
            "token_count": 120,
            "page_start": 1,
            "page_end": 1,
        }
    ]

    report = validator.validate(
        document_id="doc_test_audit",
        pages=[p1],
        chapters=chapters,
        chunks=chunks,
    )

    assert report.document_id == "doc_test_audit"
    assert report.metrics["parent_child_integrity"] == 1.0
    assert report.metrics["page_boundary_integrity"] == 1.0
    assert report.metrics["chunk_alignment"] == 1.0

    # Test audit writing
    audit_dir = tmp_path / "audit"
    validator.write_audit_reports(audit_dir, report)

    assert (audit_dir / "structural_validation.json").exists()
    assert (audit_dir / "structural_validation.md").exists()
    assert (audit_dir / "chapter_audit.json").exists()
    assert (audit_dir / "section_audit.json").exists()
    assert (audit_dir / "chunk_audit.json").exists()
    assert (audit_dir / "reading_order_audit.json").exists()


def test_structural_validator_detects_orphan_and_boundary_issues():
    validator = DocumentStructuralValidator()

    p1 = PageSchema(
        page_id="doc_test_p0001",
        document_id="doc_test",
        page_number=1,
        width=612.0,
        height=792.0,
        extraction_method="pdfplumber",
        text_quality_score=0.99,
        regions=[],
    )

    chapters = [
        ChapterInfo(
            chapter_id="ch_001",
            number=1,
            title="Chapter 1",
            page_start=1,
            page_end=5,  # Exceeds max page 1
            sections=[Section(section_id="sec_001", title="4.1 Mismatched Prefix", page_start=1, page_end=1)],
        )
    ]

    # Orphan chunk and chunk pointing to nonexistent chapter
    chunks = [
        {
            "chunk_id": "chk_001",
            "document_id": "doc_test",
            "chapter_id": "ch_nonexistent",
            "section_id": "sec_001",
            "text": "Chunk text",
            "token_count": 50,
            "page_start": 1,
            "page_end": 1,
        }
    ]

    report = validator.validate(
        document_id="doc_test",
        pages=[p1],
        chapters=chapters,
        chunks=chunks,
    )

    assert report.parent_child_integrity["invalid_parent_refs"] == 1
    assert report.page_boundary_integrity["boundary_violations"] >= 1
    assert len(report.detected_issues) > 0
