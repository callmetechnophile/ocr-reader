from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
import pymupdf
from app.core.config import settings
from app.core.logging import logger


@dataclass
class RenderResult:
    original_image: np.ndarray
    processed_image: np.ndarray
    dpi: int
    width_pt: float
    height_pt: float
    pixel_width: int
    pixel_height: int
    preprocessing_applied: list[str]


class ImagePreprocessor:
    """Modular OpenCV-based image preprocessing hooks for scanned document enhancement."""

    @staticmethod
    def to_grayscale(image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def denoise(image: np.ndarray) -> np.ndarray:
        gray = ImagePreprocessor.to_grayscale(image)
        return cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

    @staticmethod
    def normalize_contrast(image: np.ndarray) -> np.ndarray:
        gray = ImagePreprocessor.to_grayscale(image)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)

    @staticmethod
    def deskew(image: np.ndarray) -> tuple[np.ndarray, float]:
        """Detect skew angle via minAreaRect and rotate image to upright orientation."""
        gray = ImagePreprocessor.to_grayscale(image)
        # Invert colors so text is foreground
        thresh = cv2.bitwise_not(gray)
        _, thresh = cv2.threshold(thresh, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        coords = np.column_stack(np.where(thresh > 0))
        if len(coords) < 50:
            return image, 0.0

        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        elif angle > 45:
            angle = 90 - angle
        else:
            angle = -angle

        # If angle is minor, don't rotate
        if abs(angle) < 0.3:
            return image, 0.0

        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        m = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )
        return rotated, angle

    @staticmethod
    def adaptive_threshold(image: np.ndarray) -> np.ndarray:
        gray = ImagePreprocessor.to_grayscale(image)
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 8
        )

    @staticmethod
    def remove_borders(image: np.ndarray, border_ratio: float = 0.02) -> np.ndarray:
        """Remove dark scan borders/shadows along the page perimeter."""
        h, w = image.shape[:2]
        mask = np.ones((h, w), dtype=np.uint8) * 255
        bx = int(w * border_ratio)
        by = int(h * border_ratio)

        # Retain inner region
        processed = image.copy()
        if len(processed.shape) == 2:
            processed[:by, :] = 255
            processed[-by:, :] = 255
            processed[:, :bx] = 255
            processed[:, -bx:] = 255
        else:
            processed[:by, :, :] = 255
            processed[-by:, :, :] = 255
            processed[:, :bx, :] = 255
            processed[:, -bx:, :] = 255
        return processed


class PDFPageRenderer:
    """Renders PDF pages to high-resolution images via PyMuPDF with selective preprocessing."""

    def __init__(self, dpi: Optional[int] = None):
        self.dpi = dpi or settings.RENDER_DPI
        self.preprocessor = ImagePreprocessor()

    def render_page(
        self,
        doc_or_path: pymupdf.Document | str | Path,
        page_index: int,
        apply_contrast: bool = True,
        apply_deskew: bool = True,
        apply_denoise: bool = False,
        apply_border_removal: bool = True,
    ) -> RenderResult:
        if isinstance(doc_or_path, (str, Path)):
            doc = pymupdf.open(str(doc_or_path))
            should_close = True
        else:
            doc = doc_or_path
            should_close = False

        try:
            page = doc[page_index]
            width_pt = float(page.rect.width)
            height_pt = float(page.rect.height)

            # Render at specified DPI
            zoom = self.dpi / 72.0
            mat = pymupdf.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)

            # Convert pixmap to numpy array (RGB)
            img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )
            # PyMuPDF samples are RGB, convert to BGR for OpenCV standard handling
            original_bgr = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)

            processed = original_bgr.copy()
            applied_steps: list[str] = []

            if apply_border_removal:
                processed = self.preprocessor.remove_borders(processed)
                applied_steps.append("border_removal")

            if apply_deskew:
                processed, angle = self.preprocessor.deskew(processed)
                if abs(angle) >= 0.3:
                    applied_steps.append(f"deskew({round(angle, 2)}deg)")

            if apply_contrast:
                processed = self.preprocessor.normalize_contrast(processed)
                applied_steps.append("clahe_contrast")

            if apply_denoise:
                processed = self.preprocessor.denoise(processed)
                applied_steps.append("denoise")

            return RenderResult(
                original_image=original_bgr,
                processed_image=processed,
                dpi=self.dpi,
                width_pt=width_pt,
                height_pt=height_pt,
                pixel_width=pix.width,
                pixel_height=pix.height,
                preprocessing_applied=applied_steps,
            )
        finally:
            if should_close:
                doc.close()
