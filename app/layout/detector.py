import re
from abc import ABC, abstractmethod
from typing import Any, Optional
from app.schemas.page import PageExtraction
from app.schemas.region import Provenance, Region, RegionType


class LayoutDetector(ABC):
    """Abstract interface for page layout detection and region classification."""

    @abstractmethod
    def detect(
        self,
        extraction: PageExtraction,
        document_id: str,
        page_number: int,
    ) -> list[Region]:
        """
        Detect and classify semantic regions on a page.

        Args:
            extraction: The PageExtraction containing raw blocks/tables/images.
            document_id: Parent document identifier.
            page_number: Page number.

        Returns:
            List of classified Region objects with bounding boxes and initial properties.
        """
        pass


class BaselineLayoutDetector(LayoutDetector):
    """
    Production-grade rule and geometry-based Layout Detector.
    Classifies regions into HEADER, FOOTER, PAGE_NUMBER, HEADING, SUBHEADING,
    TABLE, FIGURE, CAPTION, EQUATION, and BODY.
    """

    PAGE_NUMBER_PATTERN = re.compile(
        r"^(?:page\s*)?(\d+|[ivxlcdm]+)(?:\s*(?:of|/|-)\s*\d+)?$", re.IGNORECASE
    )
    EQUATION_PATTERN = re.compile(r"(\b[a-zA-Z]\s*=\s*[\w\d\+\-\*/\(\)]+|\(\d+\.\d+\)|\bint\b|\bsum\b)")
    CAPTION_PATTERN = re.compile(r"^(?:figure|fig\.|table|chart|diagram)\s*\d+[\.:]", re.IGNORECASE)

    def detect(
        self,
        extraction: PageExtraction,
        document_id: str,
        page_number: int,
    ) -> list[Region]:
        page_width = extraction.width
        page_height = extraction.height
        regions: list[Region] = []
        region_counter = 1

        # 1. Create table regions from extracted tables
        for table in extraction.tables:
            bbox = table["bbox"]
            rows_data = table.get("data", [])
            table_text = "\n".join(
                " | ".join(str(cell or "") for cell in row) for row in rows_data if row
            )
            region_id = f"r{region_counter:03d}"
            prov = Provenance(
                document_id=document_id,
                page_number=page_number,
                region_id=region_id,
                bbox=bbox,
                source=extraction.method,
                extraction_method=extraction.method,
                confidence=0.98,
            )
            regions.append(
                Region(
                    region_id=region_id,
                    type=RegionType.TABLE,
                    bbox=bbox,
                    text=table_text if table_text else "[Table]",
                    confidence=0.98,
                    source=extraction.method,
                    reading_order=0,
                    provenance=prov,
                    metadata={"table_meta": table},
                )
            )
            region_counter += 1

        # 2. Determine median body font size from all text blocks
        font_sizes = [
            b["avg_font_size"]
            for b in extraction.blocks
            if b.get("avg_font_size") and b.get("avg_font_size") > 0
        ]
        median_font_size = (
            sorted(font_sizes)[len(font_sizes) // 2] if font_sizes else 10.0
        )

        # 3. Classify blocks
        for block in extraction.blocks:
            bbox = block["bbox"]
            text = block["text"].strip()
            if not text:
                continue

            # Check if this block is already covered by an extracted table
            if self._is_inside_any_table(bbox, extraction.tables):
                continue

            x0, y0, x1, y1 = bbox
            font_size = block.get("avg_font_size", median_font_size)

            region_type = self._classify_block(
                text=text,
                bbox=bbox,
                page_width=page_width,
                page_height=page_height,
                font_size=font_size,
                median_font_size=median_font_size,
            )

            region_id = f"r{region_counter:03d}"
            confidence = block.get("confidence", 0.95 if extraction.method == "pdfplumber" else 0.85)

            prov = Provenance(
                document_id=document_id,
                page_number=page_number,
                region_id=region_id,
                bbox=bbox,
                source=extraction.method,
                extraction_method=extraction.method,
                confidence=confidence,
            )

            regions.append(
                Region(
                    region_id=region_id,
                    type=region_type,
                    bbox=bbox,
                    text=text,
                    confidence=confidence,
                    source=extraction.method,
                    reading_order=0,
                    provenance=prov,
                    metadata={
                        "font_size": font_size,
                        "font_name": block.get("fontname"),
                    },
                )
            )
            region_counter += 1

        # 4. Extract figures from extraction.images if any
        for img in extraction.images:
            bbox = [img["x0"], img["top"], img["x1"], img["bottom"]]
            if not self._is_inside_any_table(bbox, extraction.tables):
                region_id = f"r{region_counter:03d}"
                prov = Provenance(
                    document_id=document_id,
                    page_number=page_number,
                    region_id=region_id,
                    bbox=bbox,
                    source=extraction.method,
                    extraction_method=extraction.method,
                    confidence=0.90,
                )
                regions.append(
                    Region(
                        region_id=region_id,
                        type=RegionType.FIGURE,
                        bbox=bbox,
                        text="[Figure Image]",
                        confidence=0.90,
                        source=extraction.method,
                        reading_order=0,
                        provenance=prov,
                        metadata={"image_meta": img},
                    )
                )
                region_counter += 1

        return regions

    def _classify_block(
        self,
        text: str,
        bbox: list[float],
        page_width: float,
        page_height: float,
        font_size: float,
        median_font_size: float,
    ) -> RegionType:
        x0, y0, x1, y1 = bbox
        text_clean = text.strip()

        # Top Margin (< 8% of page height)
        if y0 < page_height * 0.08:
            if self.PAGE_NUMBER_PATTERN.match(text_clean) and len(text_clean) < 12:
                return RegionType.PAGE_NUMBER
            return RegionType.HEADER

        # Bottom Margin (> 92% of page height)
        if y1 > page_height * 0.92:
            if self.PAGE_NUMBER_PATTERN.match(text_clean) and len(text_clean) < 12:
                return RegionType.PAGE_NUMBER
            return RegionType.FOOTER

        # Caption detection
        if self.CAPTION_PATTERN.match(text_clean):
            return RegionType.CAPTION

        # Heading detection based on font size hierarchy
        if font_size >= median_font_size * 1.5:
            return RegionType.HEADING
        elif font_size >= median_font_size * 1.25:
            return RegionType.SUBHEADING

        # Equation detection (short line with math symbols / equation tags)
        if len(text_clean) < 80 and self.EQUATION_PATTERN.search(text_clean):
            # Check if it has an equation label like (3.14) or math symbols
            if "(" in text_clean and ")" in text_clean and ("=" in text_clean or "+" in text_clean):
                return RegionType.EQUATION

        return RegionType.BODY

    def _is_inside_any_table(self, bbox: list[float], tables: list[dict[str, Any]]) -> bool:
        bx0, by0, bx1, by1 = bbox
        for t in tables:
            tx0, ty0, tx1, ty1 = t["bbox"]
            # If bbox center is inside table bbox
            cx = (bx0 + bx1) / 2.0
            cy = (by0 + by1) / 2.0
            if tx0 <= cx <= tx1 and ty0 <= cy <= ty1:
                return True
        return False
