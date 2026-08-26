from typing import Any, Optional, Sequence
from app.layout.detector import BaselineLayoutDetector, LayoutDetector
from app.layout.reading_order import ReadingOrderDetector, XYCutReadingOrderDetector
from app.schemas.page import PageExtraction, PageSchema
from app.schemas.region import Provenance, Region


class PageNormalizer:
    """Normalizes page geometry, bounds, coordinates, provenance, and schema construction."""

    @staticmethod
    def normalize_page(
        document_id: str,
        page_number: int,
        width: float,
        height: float,
        extraction_method: str,
        text_quality_score: float,
        regions: Sequence[Region],
        metadata: dict[str, Any] | None = None,
    ) -> PageSchema:
        page_id = f"{document_id}_p{page_number:04d}"
        normalized_regions: list[Region] = []

        for r in regions:
            # 1. Clamp bounding box to page boundaries
            x0, y0, x1, y1 = r.bbox
            nx0 = max(0.0, min(float(x0), width))
            ny0 = max(0.0, min(float(y0), height))
            nx1 = max(nx0, min(float(x1), width))
            ny1 = max(ny0, min(float(y1), height))
            clamped_bbox = [round(nx0, 2), round(ny0, 2), round(nx1, 2), round(ny1, 2)]

            # 2. Ensure provenance is populated and synchronized
            prov = r.provenance
            if prov is None:
                prov = Provenance(
                    document_id=document_id,
                    page_number=page_number,
                    region_id=r.region_id,
                    bbox=clamped_bbox,
                    source=r.source,
                    extraction_method=extraction_method,
                    confidence=r.confidence,
                )
            else:
                prov_dict = prov.model_dump()
                prov_dict["document_id"] = document_id
                prov_dict["page_number"] = page_number
                prov_dict["region_id"] = r.region_id
                prov_dict["bbox"] = clamped_bbox
                prov = Provenance(**prov_dict)

            # 3. Create updated region
            r_dict = r.model_dump()
            r_dict["bbox"] = clamped_bbox
            r_dict["provenance"] = prov
            normalized_regions.append(Region(**r_dict))

        return PageSchema(
            page_id=page_id,
            document_id=document_id,
            page_number=page_number,
            width=round(width, 2),
            height=round(height, 2),
            extraction_method=extraction_method,
            text_quality_score=round(text_quality_score, 3),
            regions=normalized_regions,
            metadata=metadata or {},
        )


class LayoutNormalizer:
    """
    Combines layout region detection, reading-order ordering, and page normalization.
    """

    def __init__(
        self,
        layout_detector: Optional[LayoutDetector] = None,
        reading_order_detector: Optional[ReadingOrderDetector] = None,
    ):
        self.layout_detector = layout_detector or BaselineLayoutDetector()
        self.reading_order_detector = reading_order_detector or XYCutReadingOrderDetector()

    def normalize_page(
        self,
        extraction: PageExtraction,
        document_id: str,
        page_number: int,
        quality_score: float = 1.0,
    ) -> PageSchema:
        # 1. Detect regions from extraction blocks/tables/images
        regions = self.layout_detector.detect(
            extraction=extraction,
            document_id=document_id,
            page_number=page_number,
        )

        # 2. Establish reading order
        ordered_regions = self.reading_order_detector.order_regions(
            regions=regions,
            page_width=extraction.width,
            page_height=extraction.height,
        )

        # 3. Normalize page coordinates and build PageSchema
        return PageNormalizer.normalize_page(
            document_id=document_id,
            page_number=page_number,
            width=extraction.width,
            height=extraction.height,
            extraction_method=extraction.method,
            text_quality_score=quality_score,
            regions=ordered_regions,
            metadata=extraction.metadata,
        )
