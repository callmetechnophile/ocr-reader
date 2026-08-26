from typing import Any, Optional
from pydantic import BaseModel, Field


class RegionGT(BaseModel):
    region_id: str
    type: str = "BODY"  # BODY, HEADING, SUBHEADING, EQUATION, TABLE, FIGURE, CAPTION, HEADER, FOOTER, PAGE_NUMBER
    bbox: list[float] = Field(..., description="[x1, y1, x2, y2]")
    polygon: Optional[list[list[float]]] = None
    text: Optional[str] = None
    reading_order: Optional[int] = None


class PageGT(BaseModel):
    page_number: int
    width: Optional[float] = None
    height: Optional[float] = None
    text: Optional[str] = None
    regions: list[RegionGT] = Field(default_factory=list)


class SectionGT(BaseModel):
    section_id: str
    title: str
    page_start: int
    page_end: int


class ChapterGT(BaseModel):
    chapter_id: str
    number: Optional[int | str] = None
    title: str
    page_start: int
    page_end: int
    sections: list[SectionGT] = Field(default_factory=list)


class ChunkGT(BaseModel):
    chunk_id: str
    chapter_id: Optional[str] = None
    section_id: Optional[str] = None
    page_start: int
    page_end: int
    text: str


class DocumentGT(BaseModel):
    document_id: Optional[str] = None
    filename: Optional[str] = None
    page_count: Optional[int] = None
    pages: list[PageGT] = Field(default_factory=list)
    chapters: list[ChapterGT] = Field(default_factory=list)
    chunks: list[ChunkGT] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
