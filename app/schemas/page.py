from enum import StrEnum
from typing import Any, Optional
from pydantic import BaseModel, Field
from app.schemas.region import Region


class ExtractionMethod(StrEnum):
    PDFPLUMBER = "pdfplumber"
    CNN_OCR = "cnn_ocr"
    BASELINE_OCR = "baseline_ocr"
    HYBRID = "hybrid"


class PageRoute(StrEnum):
    DIGITAL_TEXT = "DIGITAL_TEXT"
    POOR_TEXT_LAYER = "POOR_TEXT_LAYER"
    SCANNED_OR_IMAGE_ONLY = "SCANNED_OR_IMAGE_ONLY"


class PageExtraction(BaseModel):
    """Internal intermediate container produced by extractors before layout normalization."""
    page_number: int
    width: float
    height: float
    method: str
    confidence: float = 1.0
    raw_text: str = ""
    blocks: list[dict[str, Any]] = Field(default_factory=list)
    words: list[dict[str, Any]] = Field(default_factory=list)
    chars: list[dict[str, Any]] = Field(default_factory=list)
    tables: list[dict[str, Any]] = Field(default_factory=list)
    images: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class PageSchema(BaseModel):
    """Canonical public page schema."""
    page_id: str = Field(..., description="Unique page ID e.g. doc_xxxxx_p0293")
    document_id: str = Field(..., description="Parent document identifier")
    page_number: int = Field(..., ge=1, description="1-indexed page number")
    width: float = Field(..., gt=0, description="Page width in points/pixels")
    height: float = Field(..., gt=0, description="Page height in points/pixels")
    route: PageRoute = Field(default=PageRoute.DIGITAL_TEXT, description="Evaluated extraction route")
    extraction_method: str = Field(..., description="e.g. pdfplumber, cnn_ocr, baseline_ocr")
    text_quality_score: float = Field(..., ge=0.0, le=1.0, description="Evaluated text quality score")
    regions: list[Region] = Field(default_factory=list, description="Ordered list of page regions")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Page level metadata")
