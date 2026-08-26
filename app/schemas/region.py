from enum import StrEnum
from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator


class RegionType(StrEnum):
    BODY = "BODY"
    HEADING = "HEADING"
    SUBHEADING = "SUBHEADING"
    EQUATION = "EQUATION"
    TABLE = "TABLE"
    FIGURE = "FIGURE"
    CAPTION = "CAPTION"
    HEADER = "HEADER"
    FOOTER = "FOOTER"
    PAGE_NUMBER = "PAGE_NUMBER"
    UNKNOWN = "UNKNOWN"


class Provenance(BaseModel):
    document_id: str
    page_number: int
    region_id: str
    bbox: list[float] = Field(..., description="[x0, y0, x1, y1] coordinates")
    source: str = Field(..., description="Source e.g. pdfplumber, baseline_ocr, cnn_ocr, layout_model")
    extraction_method: str = Field(..., description="pdfplumber or ocr fallback")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, v: list[float]) -> list[float]:
        if len(v) != 4:
            raise ValueError("Bounding box must contain exactly 4 coordinates [x0, y0, x1, y1]")
        x0 = max(0.0, float(v[0]))
        y0 = max(0.0, float(v[1]))
        x1 = max(x0, float(v[2]))
        y1 = max(y0, float(v[3]))
        return [round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)]


class Region(BaseModel):
    region_id: str = Field(..., description="Unique ID within page e.g. r001")
    type: RegionType = Field(default=RegionType.BODY, description="Classified semantic region type")
    bbox: list[float] = Field(
        ...,
        description="Bounding box [x0, y0, x1, y1] in PDF point or normalized coordinate space",
    )
    text: str = Field(..., description="Extracted text content for this region")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score [0.0, 1.0]")
    source: str = Field(default="pdfplumber", description="Source tool or model e.g. pdfplumber, baseline_ocr, cnn_ocr")
    reading_order: int = Field(..., ge=0, description="Sequential reading order index on the page")
    provenance: Optional[Provenance] = None
    metadata: dict[str, Any] = Field(default_factory=dict, description="Auxiliary data (fonts, lines, etc.)")

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, v: list[float]) -> list[float]:
        if len(v) != 4:
            raise ValueError("Bounding box must contain exactly 4 coordinates [x0, y0, x1, y1]")
        x0 = max(0.0, float(v[0]))
        y0 = max(0.0, float(v[1]))
        x1 = max(x0, float(v[2]))
        y1 = max(y0, float(v[3]))
        return [round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)]
