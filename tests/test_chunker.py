from app.processing.chunker import DocumentChunker
from app.schemas.document import Chapter, Section
from app.schemas.page import PageRoute, PageSchema
from app.schemas.region import Provenance, Region, RegionType


def make_page_with_text(page_num: int, text_blocks: list[str]) -> PageSchema:
    regions = []
    for i, t in enumerate(text_blocks):
        regions.append(
            Region(
                region_id=f"r_{page_num:04d}_{i+1:03d}",
                type=RegionType.BODY,
                bbox=[50.0, float(50 + i * 50), 500.0, float(90 + i * 50)],
                text=t,
                confidence=0.99,
                reading_order=i + 1,
                provenance=Provenance(
                    document_id="doc_chk_test",
                    page_number=page_num,
                    region_id=f"r_{page_num:04d}_{i+1:03d}",
                    bbox=[50.0, float(50 + i * 50), 500.0, float(90 + i * 50)],
                    source="pdfplumber",
                    extraction_method="pdfplumber",
                ),
            )
        )
    return PageSchema(
        page_id=f"doc_chk_test_p{page_num:04d}",
        document_id="doc_chk_test",
        page_number=page_num,
        width=612.0,
        height=792.0,
        route=PageRoute.DIGITAL_TEXT,
        extraction_method="pdfplumber",
        text_quality_score=0.99,
        regions=regions,
    )


def test_chunker_basic_chunking():
    chunker = DocumentChunker(min_tokens=20, max_tokens=50, overlap_tokens=5)

    # 40 tokens block
    block1 = " ".join(["word"] * 30)
    block2 = " ".join(["data"] * 30)
    block3 = " ".join(["info"] * 30)

    p1 = make_page_with_text(1, [block1, block2])
    p2 = make_page_with_text(2, [block3])

    chapter = Chapter(
        chapter_id="ch_001",
        number=1,
        title="Testing Chunks",
        page_start=1,
        page_end=2,
        sections=[
            Section(section_id="ch_001_s001", title="1.1 Section", page_start=1, page_end=2)
        ],
        source_pages=["doc_chk_test_p0001", "doc_chk_test_p0002"],
    )

    chunks_dict = chunker.chunk_document([chapter], [p1, p2])

    assert "ch_001" in chunks_dict
    chunks = chunks_dict["ch_001"]
    assert len(chunks) >= 2

    for c in chunks:
        assert c.chapter_id == "ch_001"
        assert c.section_id == "ch_001_s001"
        assert len(c.source_regions) > 0
        assert c.page_start in (1, 2)
        assert c.page_end in (1, 2)
        assert c.token_count > 0
        assert len(c.text) > 0
