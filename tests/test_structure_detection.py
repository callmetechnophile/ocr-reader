import pytest
from app.schemas.page import PageRoute, PageSchema
from app.schemas.region import Provenance, Region, RegionType
from app.structure.chapter_detector import ChapterDetector
from app.structure.section_detector import SectionDetector


def make_page(
    page_num: int,
    regions: list[tuple[str, RegionType]],
    doc_id: str = "doc_test_123",
) -> PageSchema:
    region_objs = []
    for i, (text, r_type) in enumerate(regions):
        region_objs.append(
            Region(
                region_id=f"r_{page_num:04d}_{i+1:03d}",
                type=r_type,
                bbox=[50.0, float(50 + i * 40), 500.0, float(80 + i * 40)],
                text=text,
                confidence=0.98,
                reading_order=i + 1,
                provenance=Provenance(
                    document_id=doc_id,
                    page_number=page_num,
                    region_id=f"r_{page_num:04d}_{i+1:03d}",
                    bbox=[50.0, float(50 + i * 40), 500.0, float(80 + i * 40)],
                    source="pdfplumber",
                    extraction_method="pdfplumber",
                ),
            )
        )

    return PageSchema(
        page_id=f"{doc_id}_p{page_num:04d}",
        document_id=doc_id,
        page_number=page_num,
        width=612.0,
        height=792.0,
        route=PageRoute.DIGITAL_TEXT,
        extraction_method="pdfplumber",
        text_quality_score=0.99,
        regions=region_objs,
    )


def test_chapter_detector_explicit_patterns():
    detector = ChapterDetector()

    p1 = make_page(1, [("Chapter 1: Getting Started", RegionType.HEADING), ("This is intro text.", RegionType.BODY)])
    p2 = make_page(2, [("Some ongoing body text.", RegionType.BODY)])
    p3 = make_page(3, [("Chapter 2: Memory Layout", RegionType.HEADING), ("Memory is organized into blocks.", RegionType.BODY)])

    chapters = detector.detect_chapters([p1, p2, p3], "doc_test_123")

    assert len(chapters) == 2
    assert chapters[0].chapter_id == "ch_001"
    assert chapters[0].number == 1
    assert "Getting Started" in chapters[0].title
    assert chapters[0].page_start == 1
    assert chapters[0].page_end == 2
    assert chapters[0].confidence > 0.90

    assert chapters[1].chapter_id == "ch_002"
    assert chapters[1].number == 2
    assert "Memory Layout" in chapters[1].title
    assert chapters[1].page_start == 3
    assert chapters[1].page_end == 3


def test_section_detector_hierarchy():
    ch_detector = ChapterDetector()
    sec_detector = SectionDetector()

    p1 = make_page(1, [
        ("Chapter 1: Basics", RegionType.HEADING),
        ("1.1 Imperative Programming", RegionType.SUBHEADING),
        ("Variables and assignments.", RegionType.BODY),
    ])
    p2 = make_page(2, [
        ("1.2 Compilation Workflow", RegionType.SUBHEADING),
        ("GCC turns C into machine code.", RegionType.BODY),
    ])

    chapters = ch_detector.detect_chapters([p1, p2], "doc_test_123")
    chapters_with_sections = sec_detector.detect_sections(chapters, [p1, p2])

    assert len(chapters_with_sections) == 1
    ch1 = chapters_with_sections[0]
    assert len(ch1.sections) == 2
    assert ch1.sections[0].section_id == "ch_001_s001"
    assert "1.1" in ch1.sections[0].title
    assert ch1.sections[0].page_start == 1
    assert ch1.sections[1].section_id == "ch_001_s002"
    assert "1.2" in ch1.sections[1].title
    assert ch1.sections[1].page_start == 2
