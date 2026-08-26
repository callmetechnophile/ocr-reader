from pathlib import Path
from typing import Sequence
import cv2
import numpy as np
import pymupdf
from app.schemas.page import PageSchema
from app.schemas.region import RegionType


class DebugVisualizer:
    """
    Renders visual debug overlays on textbook pages showing bounding boxes,
    region labels, reading order, and confidence scores.
    """

    COLOR_MAP = {
        RegionType.HEADING: (0, 0, 200),        # Red
        RegionType.SUBHEADING: (0, 100, 200),   # Orange-Red
        RegionType.BODY: (200, 0, 0),          # Blue
        RegionType.CAPTION: (0, 180, 180),     # Yellow/Gold
        RegionType.TABLE: (0, 150, 0),         # Green
        RegionType.HEADER: (150, 150, 150),    # Grey
        RegionType.FOOTER: (150, 150, 150),    # Grey
        RegionType.PAGE_NUMBER: (120, 120, 120),
        RegionType.FIGURE: (150, 0, 150),      # Purple
    }

    def render_debug_page(
        self,
        pdf_path: str | Path,
        page: PageSchema,
        output_dir: str | Path,
        dpi: int = 150,
    ) -> Path:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        with pymupdf.open(str(pdf_path)) as doc:
            doc_page = doc[page.page_number - 1]
            zoom = dpi / 72.0
            mat = pymupdf.Matrix(zoom, zoom)
            pix = doc_page.get_pixmap(matrix=mat)

            # Convert pixmap to OpenCV image (RGB -> BGR)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
            if pix.n == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            scale_x = pix.w / page.width
            scale_y = pix.h / page.height

            # Draw regions
            for r in page.regions:
                x0 = int(round(r.bbox[0] * scale_x))
                y0 = int(round(r.bbox[1] * scale_y))
                x1 = int(round(r.bbox[2] * scale_x))
                y1 = int(round(r.bbox[3] * scale_y))

                color = self.COLOR_MAP.get(r.type, (0, 120, 255))
                cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

                label = f"#{r.reading_order} [{r.type.value}] ({r.confidence:.2f})"
                # Draw text label banner
                cv2.rectangle(img, (x0, max(0, y0 - 18)), (x0 + len(label) * 8, y0), color, -1)
                cv2.putText(
                    img,
                    label,
                    (x0 + 2, max(12, y0 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        # Save debug image
        img_file = out_dir / f"page_{page.page_number:04d}.png"
        cv2.imwrite(str(img_file), img)

        # Save debug text dump
        text_file = out_dir / f"page_{page.page_number:04d}_text.txt"
        lines = [f"=== Page {page.page_number} (Route: {page.route.value}, Score: {page.text_quality_score}) ==="]
        for r in page.regions:
            lines.append(f"\n[{r.reading_order}] {r.type.value} (conf={r.confidence:.2f}, bbox={r.bbox}):")
            lines.append(r.text)

        with open(text_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        return img_file
