import pytest
from app.layout.detector import BaselineLayoutDetector
from app.layout.reading_order import ReadingOrderSorter
from app.schemas.page import PageExtraction
from app.schemas.region import Region, RegionType


def test_layout_detector_classification():
    detector = BaselineLayoutDetector()

    extraction = PageExtraction(
        page_number=1,
        width=612.0,
        height=792.0,
        method="pdfplumber",
        blocks=[
            {"text": "CHAPTER 1: INTRODUCTION", "bbox": [72, 30, 400, 50], "avg_font_size": 10.0},
            {"text": "1.1 Main Title", "bbox": [72, 80, 400, 110], "avg_font_size": 18.0},
            {"text": "Regular body text explaining MOSFET physics.", "bbox": [72, 120, 500, 180], "avg_font_size": 10.0},
            {"text": "1", "bbox": [540, 760, 560, 780], "avg_font_size": 10.0},
        ],
    )

    regions = detector.detect(extraction, document_id="doc_test", page_number=1)
    assert len(regions) == 4

    types = [r.type for r in regions]
    assert RegionType.HEADER in types
    assert RegionType.HEADING in types
    assert RegionType.BODY in types
    assert RegionType.PAGE_NUMBER in types


def test_reading_order_two_columns():
    sorter = ReadingOrderSorter()

    # Create synthetic 2-column regions
    header = Region(
        region_id="r001",
        type=RegionType.HEADER,
        bbox=[72, 30, 540, 50],
        text="Chapter Header",
        confidence=1.0,
        source="pdfplumber",
        reading_order=0,
    )
    title = Region(
        region_id="r002",
        type=RegionType.HEADING,
        bbox=[72, 70, 540, 95],
        text="Full Width Title Spanning Columns",
        confidence=1.0,
        source="pdfplumber",
        reading_order=0,
    )
    left1 = Region(
        region_id="r003",
        type=RegionType.BODY,
        bbox=[72, 110, 280, 200],
        text="Left column first paragraph",
        confidence=1.0,
        source="pdfplumber",
        reading_order=0,
    )
    left2 = Region(
        region_id="r004",
        type=RegionType.BODY,
        bbox=[72, 220, 280, 320],
        text="Left column second paragraph",
        confidence=1.0,
        source="pdfplumber",
        reading_order=0,
    )
    right1 = Region(
        region_id="r005",
        type=RegionType.BODY,
        bbox=[330, 110, 540, 200],
        text="Right column first paragraph",
        confidence=1.0,
        source="pdfplumber",
        reading_order=0,
    )
    footer = Region(
        region_id="r006",
        type=RegionType.PAGE_NUMBER,
        bbox=[540, 760, 560, 780],
        text="2",
        confidence=1.0,
        source="pdfplumber",
        reading_order=0,
    )

    regions = [right1, left2, header, footer, left1, title]
    ordered = sorter.sort_regions(regions, page_width=612.0, page_height=792.0)

    # Validate ordering: Header -> Title -> Left1 -> Left2 -> Right1 -> Footer
    assert len(ordered) == 6
    for idx, r in enumerate(ordered):
        assert r.reading_order == idx + 1

    assert ordered[0].region_id == "r001"  # Header
    assert ordered[1].region_id == "r002"  # Title
    assert ordered[2].region_id == "r003"  # Left 1
    assert ordered[3].region_id == "r004"  # Left 2
    assert ordered[4].region_id == "r005"  # Right 1
    assert ordered[5].region_id == "r006"  # Footer / Page Number
