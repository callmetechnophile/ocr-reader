from pathlib import Path
from typing import Any, Optional
import cv2
import numpy as np
from app.core.config import settings
from app.core.logging import logger
from app.extractors.base import OCRExtractor
from app.ml.inference.recognizer import CRNNRecognizer, get_recognizer
from app.schemas.page import PageExtraction


class CNNOCRExtractor(OCRExtractor):
    """
    CRNN OCR Extractor utilizing CNN -> BiLSTM -> CTC recognition model.

    Extracts text-line bounding box regions from rendered page images,
    preprocesses cropped lines with aspect-ratio preservation, feeds them to the
    CRNNRecognizer, and returns normalized PageExtraction conforming to the OCR contract.
    """

    def __init__(
        self,
        backend: Optional[str] = None,
        model_path: Optional[str | Path] = None,
        vocab_path: Optional[str | Path] = None,
    ):
        self.model_path = Path(model_path or getattr(settings, "MODEL_PATH", "./models/ocr/crnn_v1_best.pt"))
        self.vocab_path = Path(vocab_path or getattr(settings, "VOCAB_PATH", "./models/ocr/vocab.json"))
        self.backend = backend or settings.BASELINE_BACKEND
        self._recognizer: Optional[CRNNRecognizer] = None
        self._model_loaded = False
        self._check_for_models()

    @property
    def name(self) -> str:
        return f"{self.backend}_ocr"

    def _check_for_models(self) -> None:
        """Check if PyTorch CRNN weights exist."""
        if self.model_path.exists() and self.vocab_path.exists():
            logger.info("Found CRNN OCR checkpoint", extra={"path": str(self.model_path)})
            self.backend = "cnn_ocr"
            self._model_loaded = True
            try:
                self._recognizer = CRNNRecognizer(
                    model_path=self.model_path,
                    vocab_path=self.vocab_path,
                    device=getattr(settings, "MODEL_DEVICE", "cpu"),
                    model_version=getattr(settings, "MODEL_VERSION", "crnn_v1"),
                )
            except Exception as exc:
                logger.warning(f"Could not initialize CRNNRecognizer: {exc}")
                self._model_loaded = False
                self.backend = "baseline"
        else:
            self.backend = "baseline"
            self._model_loaded = False

    def extract(self, page: Any, **kwargs: Any) -> PageExtraction:
        page_number = kwargs.get("page_number", 1)
        orig_width = kwargs.get("orig_width")
        orig_height = kwargs.get("orig_height")

        if not isinstance(page, np.ndarray):
            raise TypeError(f"CNNOCRExtractor expects numpy.ndarray image, got {type(page)}")

        img_height, img_width = page.shape[:2]
        width = float(orig_width if orig_width is not None else img_width)
        height = float(orig_height if orig_height is not None else img_height)

        # Scale factor from pixel image to PDF point space
        scale_x = width / img_width if img_width > 0 else 1.0
        scale_y = height / img_height if img_height > 0 else 1.0

        if self._model_loaded and self.backend == "cnn_ocr" and self._recognizer is not None:
            return self._extract_with_crnn(page, page_number, width, height, scale_x, scale_y)
        else:
            return self._extract_with_baseline(page, page_number, width, height, scale_x, scale_y)

    def _extract_with_crnn(
        self,
        image: np.ndarray,
        page_number: int,
        page_width: float,
        page_height: float,
        scale_x: float,
        scale_y: float,
    ) -> PageExtraction:
        """
        Segment lines and run CRNN inference on line crops.
        """
        # 1. Line contour detection
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        dilated = cv2.dilate(binary, kernel, iterations=2)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > 10 and h > 6:
                boxes.append((x, y, w, h))

        boxes.sort(key=lambda b: (b[1], b[0]))

        if not boxes:
            return PageExtraction(
                page_number=page_number,
                width=page_width,
                height=page_height,
                method="cnn_ocr",
                confidence=1.0,
                raw_text="",
                blocks=[],
                words=[],
                chars=[],
            )

        # 2. Extract crops with small padding
        img_h, img_w = image.shape[:2]
        line_crops = []
        for bx, by, bw, bh in boxes:
            pad = 2
            y0 = max(0, by - pad)
            y1 = min(img_h, by + bh + pad)
            x0 = max(0, bx - pad)
            x1 = min(img_w, bx + bw + pad)
            crop = image[y0:y1, x0:x1]
            line_crops.append(crop)

        # 3. Batch prediction through CRNN
        predictions = self._recognizer.predict_batch(line_crops)

        blocks = []
        words = []
        confs = []

        for idx, ((bx, by, bw, bh), pred) in enumerate(zip(boxes, predictions)):
            x0 = round(bx * scale_x, 2)
            top = round(by * scale_y, 2)
            x1 = round((bx + bw) * scale_x, 2)
            bottom = round((by + bh) * scale_y, 2)

            text = pred["text"].strip()
            conf = pred["confidence"]
            confs.append(conf)

            blocks.append({
                "text": text if text else f"[Line {idx+1}]",
                "bbox": [x0, top, x1, bottom],
                "avg_font_size": round(bh * scale_y * 0.75, 2),
                "fontname": "CRNN_OCR",
                "word_count": len(text.split()),
                "confidence": conf,
            })

            # Generate approximate word bboxes along line
            raw_tokens = text.split()
            if raw_tokens:
                char_width = (x1 - x0) / max(1, len(text))
                cur_x = x0
                for w_idx, token in enumerate(raw_tokens):
                    w_len = len(token)
                    wx1 = cur_x + (w_len * char_width)
                    words.append({
                        "text": token,
                        "x0": round(cur_x, 2),
                        "top": top,
                        "x1": round(wx1, 2),
                        "bottom": bottom,
                        "fontname": "CRNN_OCR",
                        "size": round(bh * scale_y * 0.75, 2),
                    })
                    cur_x = wx1 + char_width

        raw_text = "\n".join(b["text"] for b in blocks)
        page_conf = round(float(np.mean(confs)), 4) if confs else 0.90

        return PageExtraction(
            page_number=page_number,
            width=page_width,
            height=page_height,
            method="cnn_ocr",
            confidence=page_conf,
            raw_text=raw_text,
            blocks=blocks,
            words=words,
            chars=[],
            tables=[],
            images=[],
            metadata={
                "backend": "cnn_ocr",
                "model_version": self._recognizer.model_version,
                "detected_regions": len(blocks),
            },
        )

    def _extract_with_baseline(
        self,
        image: np.ndarray,
        page_number: int,
        page_width: float,
        page_height: float,
        scale_x: float,
        scale_y: float,
    ) -> PageExtraction:
        """
        Fallback heuristic text line segmentation for scanned pages.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        dilated = cv2.dilate(binary, kernel, iterations=2)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > 10 and h > 6:
                boxes.append((x, y, w, h))

        boxes.sort(key=lambda b: (b[1], b[0]))

        blocks = []
        words = []

        for idx, (bx, by, bw, bh) in enumerate(boxes):
            x0 = round(bx * scale_x, 2)
            top = round(by * scale_y, 2)
            x1 = round((bx + bw) * scale_x, 2)
            bottom = round((by + bh) * scale_y, 2)
            region_text = f"[Scanned text line {idx + 1}]"

            blocks.append({
                "text": region_text,
                "bbox": [x0, top, x1, bottom],
                "avg_font_size": round(bh * scale_y * 0.75, 2),
                "fontname": "BaselineOCRFont",
                "word_count": len(region_text.split()),
                "confidence": 0.85,
            })

            line_roi = binary[by : by + bh, bx : bx + bw]
            word_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 2))
            word_dilated = cv2.dilate(line_roi, word_kernel, iterations=1)
            word_cnts, _ = cv2.findContours(word_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            word_boxes = [cv2.boundingRect(c) for c in word_cnts if cv2.boundingRect(c)[2] > 4]
            word_boxes.sort(key=lambda wb: wb[0])

            for w_idx, (wx, wy, ww, wh) in enumerate(word_boxes):
                wx0 = round((bx + wx) * scale_x, 2)
                wtop = round((by + wy) * scale_y, 2)
                wx1 = round((bx + wx + ww) * scale_x, 2)
                wbottom = round((by + wy + wh) * scale_y, 2)
                words.append({
                    "text": f"word_{w_idx + 1}",
                    "x0": wx0,
                    "top": wtop,
                    "x1": wx1,
                    "bottom": wbottom,
                    "fontname": "BaselineOCRFont",
                    "size": round(wh * scale_y * 0.75, 2),
                })

        raw_text = "\n".join(b["text"] for b in blocks)

        return PageExtraction(
            page_number=page_number,
            width=page_width,
            height=page_height,
            method="baseline_ocr",
            confidence=0.85,
            raw_text=raw_text,
            blocks=blocks,
            words=words,
            chars=[],
            tables=[],
            images=[],
            metadata={
                "backend": self.backend,
                "detected_regions": len(blocks),
                "detected_words": len(words),
            },
        )
