import pytest
from pydantic import ValidationError
from app.schemas.document import DocumentManifest, DocumentMetadata, DocumentStatus
from app.schemas.page import PageSchema
from app.schemas.region import Provenance, Region, RegionType


def test_valid_region_and_provenance():
    prov = Provenance(
        document_id="doc_12345",
        page_number=1,
        region_id="r001",
        bbox=[72.0, 140.0, 520.0, 180.0],
        source="pdfplumber",
        extraction_method="pdfplumber",
        confidence=0.99,
    )
    region = Region(
        region_id="r001",
        type=RegionType.BODY,
        bbox=[72.0, 140.0, 520.0, 180.0],
        text="MOSFET Operation",
        confidence=0.99,
        source="pdfplumber",
        reading_order=1,
        provenance=prov,
    )
    assert region.region_id == "r001"
    assert region.type == RegionType.BODY
    assert region.bbox == [72.0, 140.0, 520.0, 180.0]


def test_invalid_region_type():
    with pytest.raises(ValidationError):
        Region(
            region_id="r001",
            type="INVALID_TYPE",  # Not a valid RegionType
            bbox=[10, 10, 50, 50],
            text="Invalid",
            source="pdfplumber",
            reading_order=1,
        )


def test_invalid_bounding_box_coordinates():
    # Invalid length
    with pytest.raises(ValidationError):
        Region(
            region_id="r001",
            type=RegionType.BODY,
            bbox=[10.0, 20.0, 50.0],
            text="Invalid length bbox",
            source="pdfplumber",
            reading_order=1,
        )

    # Clamping behavior for negative coordinates
    r = Region(
        region_id="r001",
        type=RegionType.BODY,
        bbox=[-12.98, -8.0, 650.92, 792.0],
        text="Clamped negative bbox",
        source="pdfplumber",
        reading_order=1,
    )
    assert r.bbox == [0.0, 0.0, 650.92, 792.0]


def test_page_schema_serialization():
    page = PageSchema(
        page_id="doc_123_p0001",
        document_id="doc_123",
        page_number=1,
        width=612.0,
        height=792.0,
        extraction_method="pdfplumber",
        text_quality_score=0.98,
        regions=[],
    )
    data = page.model_dump()
    assert data["page_id"] == "doc_123_p0001"
    assert data["text_quality_score"] == 0.98
    assert data["regions"] == []
