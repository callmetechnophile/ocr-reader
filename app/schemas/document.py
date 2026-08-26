from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Optional
from pydantic import BaseModel, Field


class DocumentStatus(StrEnum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    COMPLETED_WITH_ERRORS = "completed_with_errors"
    FAILED = "failed"


class DocumentCreateResponse(BaseModel):
    document_id: str
    filename: Optional[str] = None
    status: DocumentStatus = DocumentStatus.QUEUED


class DocumentStatusResponse(BaseModel):
    document_id: str
    filename: Optional[str] = None
    status: DocumentStatus
    progress: float = Field(0.0, ge=0.0, le=100.0, description="Progress percentage 0-100")
    pages_processed: int = Field(0, ge=0)
    pages_failed: int = Field(0, ge=0)
    total_pages: int = Field(0, ge=0)
    current_page: Optional[int] = None
    error: Optional[str] = None


class Section(BaseModel):
    section_id: str = Field(..., description="e.g. ch_001_s001")
    title: str
    page_start: int
    page_end: int


class Chapter(BaseModel):
    chapter_id: str = Field(..., description="e.g. ch_001")
    number: Optional[int | str] = Field(default=1, description="Chapter number if present")
    title: str
    page_start: int
    page_end: int
    sections: list[Section] = Field(default_factory=list)
    source_pages: list[str] = Field(default_factory=list, description="e.g. ['doc_xxx_p0001', ...]")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    detection_method: str = Field(default="pattern")

    @property
    def start_page(self) -> int:
        return self.page_start

    @property
    def end_page(self) -> int:
        return self.page_end


ChapterInfo = Chapter


class Chunk(BaseModel):
    chunk_id: str = Field(..., description="e.g. ch_001_chk_0001")
    chapter_id: Optional[str] = None
    section_id: Optional[str] = None
    page_start: int
    page_end: int
    source_regions: list[str] = Field(default_factory=list, description="List of region IDs")
    token_count: int = Field(..., ge=0)
    text: str


class PDFProfile(BaseModel):
    file_size: int
    page_count: int
    page_dimensions: list[dict[str, float]] = Field(default_factory=list)
    text_layer_pages: int = 0
    poor_text_pages: int = 0
    image_pages: int = 0
    zero_text_pages: int = 0
    average_characters_per_page: float = 0.0
    average_words_per_page: float = 0.0


class DocumentManifest(BaseModel):
    document_id: str
    filename: Optional[str] = None
    page_count: int = 0
    processed_pages: int = 0
    failed_pages: int = 0
    chapters: int = 0
    sections: int = 0
    chunks: int = 0
    paths: dict[str, Optional[str]] = Field(
        default_factory=lambda: {
            "pages": "pages/",
            "chapters": "chapters/",
            "chunks": "chunks/",
        }
    )
    pages: list[str] = Field(default_factory=list)
    partial_run: bool = False
    page_range: Optional[list[int]] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    toon: Optional[dict[str, Any]] = Field(default=None, description="TOON export info {'enabled': bool, 'path': str}")


class DocumentMetadata(BaseModel):
    document_id: str
    filename: str
    sha256: str
    file_size: int
    page_count: int
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: Optional[str] = None
    pipeline_version: str = "ocr_v0"
    pdf_title: Optional[str] = None
    status: DocumentStatus = DocumentStatus.QUEUED
    error: Optional[str] = None
    config_snapshot: dict[str, Any] = Field(default_factory=dict)


class ProcessingReport(BaseModel):
    document_id: str
    filename: Optional[str] = None
    processing: dict[str, Any] = Field(default_factory=dict)
    pages: dict[str, int] = Field(default_factory=dict)
    routing: dict[str, int] = Field(default_factory=dict)
    extraction: dict[str, int] = Field(default_factory=dict)
    structure: dict[str, int] = Field(default_factory=dict)
    performance: dict[str, Any] = Field(default_factory=dict)
    toon_enabled: bool = False
    toon_output_path: Optional[str] = None
    toon_size_bytes: Optional[int] = None
    toon_generation_time: Optional[float] = None


class LocalJobState(BaseModel):
    document_id: str
    filename: Optional[str] = None
    status: DocumentStatus = DocumentStatus.QUEUED
    progress: float = 0.0
    pages_total: int = 0
    pages_processed: int = 0
    pages_failed: int = 0
    current_page: Optional[int] = None
    error: Optional[str] = None
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
