from typing import Any, Optional
import pdfplumber
from app.extractors.base import OCRExtractor
from app.schemas.page import PageExtraction


class PDFPlumberExtractor(OCRExtractor):
    """Extractor utilizing pdfplumber to extract text layer, words, chars, bboxes, and tables."""

    @property
    def name(self) -> str:
        return "pdfplumber"

    def extract(self, page: pdfplumber.page.Page, **kwargs: Any) -> PageExtraction:
        page_number = kwargs.get("page_number", getattr(page, "page_number", 1))
        width = float(page.width)
        height = float(page.height)

        # 1. Raw text with and without layout preservation
        raw_text = page.extract_text() or ""
        layout_text = page.extract_text(layout=True) or ""

        # 2. Extract words with coordinates and font metadata
        raw_words = page.extract_words(
            extra_attrs=["fontname", "size", "adv"],
            keep_blank_chars=False,
            x_tolerance=3,
            y_tolerance=3,
        )
        words = []
        for w in raw_words:
            raw_x0 = float(w.get("x0", 0.0))
            raw_top = float(w.get("top", 0.0))
            raw_x1 = float(w.get("x1", 0.0))
            raw_bottom = float(w.get("bottom", 0.0))
            x0 = max(0.0, min(raw_x0, width))
            top = max(0.0, min(raw_top, height))
            x1 = max(x0, min(raw_x1, width))
            bottom = max(top, min(raw_bottom, height))
            words.append({
                "text": w.get("text", ""),
                "x0": round(x0, 2),
                "top": round(top, 2),
                "x1": round(x1, 2),
                "bottom": round(bottom, 2),
                "fontname": w.get("fontname"),
                "size": round(float(w.get("size", 10.0)), 2) if w.get("size") else None,
                "adv": w.get("adv"),
            })

        # 3. Extract characters with granular geometry
        raw_chars = page.chars
        chars = []
        for c in raw_chars:
            raw_x0 = float(c.get("x0", 0.0))
            raw_top = float(c.get("top", 0.0))
            raw_x1 = float(c.get("x1", 0.0))
            raw_bottom = float(c.get("bottom", 0.0))
            x0 = max(0.0, min(raw_x0, width))
            top = max(0.0, min(raw_top, height))
            x1 = max(x0, min(raw_x1, width))
            bottom = max(top, min(raw_bottom, height))
            chars.append({
                "text": c.get("text", ""),
                "x0": round(x0, 2),
                "top": round(top, 2),
                "x1": round(x1, 2),
                "bottom": round(bottom, 2),
                "fontname": c.get("fontname"),
                "size": round(float(c.get("size", 10.0)), 2) if c.get("size") else None,
            })

        # 4. Extract tables and table bounding boxes
        tables = []
        try:
            table_objs = page.find_tables()
            for t in table_objs:
                tx0 = max(0.0, min(float(t.bbox[0]), width))
                ty0 = max(0.0, min(float(t.bbox[1]), height))
                tx1 = max(tx0, min(float(t.bbox[2]), width))
                ty1 = max(ty0, min(float(t.bbox[3]), height))
                table_bbox = [
                    round(tx0, 2),
                    round(ty0, 2),
                    round(tx1, 2),
                    round(ty1, 2),
                ]
                extracted_data = t.extract()
                tables.append({
                    "bbox": table_bbox,
                    "data": extracted_data,
                    "rows": len(extracted_data),
                    "cols": len(extracted_data[0]) if extracted_data else 0,
                })
        except Exception:
            # Fallback if table extraction fails on complex vectors
            tables = []

        # 5. Extract image references
        images = []
        for img in getattr(page, "images", []):
            ix0 = max(0.0, min(float(img.get("x0", 0.0)), width))
            itop = max(0.0, min(float(img.get("top", 0.0)), height))
            ix1 = max(ix0, min(float(img.get("x1", 0.0)), width))
            ibottom = max(itop, min(float(img.get("bottom", 0.0)), height))
            images.append({
                "x0": round(ix0, 2),
                "top": round(itop, 2),
                "x1": round(ix1, 2),
                "bottom": round(ibottom, 2),
                "width": round(max(0.0, ix1 - ix0), 2),
                "height": round(max(0.0, ibottom - itop), 2),
            })

        # 6. Group words into lines/blocks if not empty
        blocks = self._cluster_words_into_blocks(words, width, height)

        return PageExtraction(
            page_number=page_number,
            width=width,
            height=height,
            method="pdfplumber",
            confidence=1.0,
            raw_text=raw_text,
            blocks=blocks,
            words=words,
            chars=chars,
            tables=tables,
            images=images,
            metadata={
                "layout_text": layout_text,
                "word_count": len(words),
                "char_count": len(chars),
                "table_count": len(tables),
                "image_count": len(images),
            },
        )

    def _cluster_words_into_blocks(
        self, words: list[dict[str, Any]], page_width: float, page_height: float
    ) -> list[dict[str, Any]]:
        """Cluster adjacent words on the same line into coherent text lines/blocks."""
        if not words:
            return []

        # Group words by vertical line proximity (within 3pt tolerance)
        sorted_words = sorted(words, key=lambda w: (w["top"], w["x0"]))
        lines: list[list[dict[str, Any]]] = []
        for w in sorted_words:
            matched = False
            for line in lines:
                # Compare line median/first top
                if abs(line[0]["top"] - w["top"]) <= 3.5:
                    line.append(w)
                    matched = True
                    break
            if not matched:
                lines.append([w])

        blocks = []
        for line_words in lines:
            line_words_sorted = sorted(line_words, key=lambda w: w["x0"])
            text = " ".join(w["text"] for w in line_words_sorted)
            x0 = min(w["x0"] for w in line_words_sorted)
            top = min(w["top"] for w in line_words_sorted)
            x1 = max(w["x1"] for w in line_words_sorted)
            bottom = max(w["bottom"] for w in line_words_sorted)
            font_sizes = [w["size"] for w in line_words_sorted if w.get("size")]
            avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 10.0
            font_names = [w["fontname"] for w in line_words_sorted if w.get("fontname")]
            primary_font = font_names[0] if font_names else "Unknown"

            blocks.append({
                "text": text,
                "bbox": [x0, top, x1, bottom],
                "avg_font_size": avg_font_size,
                "fontname": primary_font,
                "word_count": len(line_words_sorted),
            })

        return blocks
