import string
from dataclasses import dataclass
from pathlib import Path
import pdfplumber
import pymupdf
from app.core.config import settings
from app.core.logging import logger
from app.schemas.document import PDFProfile
from app.schemas.page import PageRoute


@dataclass
class PageQualityReport:
    page_number: int
    width: float
    height: float
    route: PageRoute
    text_layer_available: bool
    text_character_count: int
    word_count: int
    image_count: int
    printable_ratio: float
    whitespace_ratio: float
    garbage_ratio: float
    text_quality_score: float
    should_use_text_layer: bool


class PDFPageAnalyzer:
    """
    Evaluates the quality of a PDF page's digital text layer, determines its routing
    (DIGITAL_TEXT, POOR_TEXT_LAYER, SCANNED_OR_IMAGE_ONLY), and profiles PDF documents.
    """

    def __init__(self, quality_threshold: float = 0.70):
        self.quality_threshold = quality_threshold or settings.TEXT_QUALITY_THRESHOLD
        self.printable_set = set(string.printable)

    def analyze(self, page: pdfplumber.page.Page, page_number: int) -> PageQualityReport:
        width = float(page.width)
        height = float(page.height)

        raw_text = page.extract_text() or ""
        char_count = len(raw_text)
        image_count = len(getattr(page, "images", []))

        # Basic presence check
        if char_count == 0:
            return PageQualityReport(
                page_number=page_number,
                width=width,
                height=height,
                route=PageRoute.SCANNED_OR_IMAGE_ONLY,
                text_layer_available=False,
                text_character_count=0,
                word_count=0,
                image_count=image_count,
                printable_ratio=0.0,
                whitespace_ratio=0.0,
                garbage_ratio=1.0,
                text_quality_score=0.0,
                should_use_text_layer=False,
            )

        # 1. Printable character ratio
        printable_chars = sum(1 for c in raw_text if c in self.printable_set)
        printable_ratio = printable_chars / char_count if char_count > 0 else 0.0

        # 2. Word count and valid word ratio
        raw_words = [w for w in raw_text.split() if w.strip()]
        word_count = len(raw_words)

        # 3. Excessive garbage / replacement characters detection (e.g. \ufffd, control chars)
        garbage_chars = sum(1 for c in raw_text if ord(c) < 32 and c not in "\n\r\t")
        garbage_chars += sum(1 for c in raw_text if ord(c) == 0xFFFD)
        garbage_ratio = garbage_chars / char_count if char_count > 0 else 0.0

        # 4. Whitespace anomaly detection
        whitespace_count = sum(1 for c in raw_text if c.isspace())
        whitespace_ratio = whitespace_count / char_count if char_count > 0 else 0.0

        # 5. Average word length check
        avg_word_len = (
            sum(len(w) for w in raw_words) / word_count if word_count > 0 else 0.0
        )

        # Composite quality score calculation:
        score = 1.0

        # Penalty for non-printable characters
        if printable_ratio < 0.95:
            score -= (0.95 - printable_ratio) * 2.0

        # Heavy penalty for garbage characters
        if garbage_ratio > 0.01:
            score -= min(1.0, garbage_ratio * 10.0)

        # Penalty for abnormal average word length (e.g. 1-letter words due to bad kerning or huge strings)
        if avg_word_len < 2.2 or avg_word_len > 25.0:
            score -= 0.35

        # Penalty if character count is suspiciously low while images dominate the page
        if char_count < 150 and image_count > 0:
            score -= 0.30

        # Penalty for extreme whitespace ratio (standard book pages are typically 0.12 - 0.28)
        if whitespace_ratio > 0.40 or (char_count > 50 and whitespace_ratio < 0.05):
            score -= 0.25

        score = max(0.0, min(1.0, round(score, 3)))
        should_use = score >= self.quality_threshold

        # Determine route
        if should_use:
            route = PageRoute.DIGITAL_TEXT
        elif char_count > 0:
            route = PageRoute.POOR_TEXT_LAYER
        else:
            route = PageRoute.SCANNED_OR_IMAGE_ONLY

        return PageQualityReport(
            page_number=page_number,
            width=width,
            height=height,
            route=route,
            text_layer_available=True,
            text_character_count=char_count,
            word_count=word_count,
            image_count=image_count,
            printable_ratio=round(printable_ratio, 3),
            whitespace_ratio=round(whitespace_ratio, 3),
            garbage_ratio=round(garbage_ratio, 3),
            text_quality_score=score,
            should_use_text_layer=should_use,
        )

    def profile_document(self, pdf_path: str | Path) -> PDFProfile:
        """
        Scan and profile an arbitrary PDF document without performing full OCR extraction.
        """
        p_path = Path(pdf_path)
        file_size = p_path.stat().st_size if p_path.exists() else 0

        with pdfplumber.open(str(p_path)) as pdf:
            page_count = len(pdf.pages)
            dimensions = []
            text_layer_pages = 0
            poor_text_pages = 0
            image_pages = 0
            zero_text_pages = 0
            total_chars = 0
            total_words = 0

            for idx, page in enumerate(pdf.pages):
                dims = {"width": float(page.width), "height": float(page.height)}
                dimensions.append(dims)

                try:
                    report = self.analyze(page, idx + 1)
                    total_chars += report.text_character_count
                    total_words += report.word_count

                    if report.route == PageRoute.DIGITAL_TEXT:
                        text_layer_pages += 1
                    elif report.route == PageRoute.POOR_TEXT_LAYER:
                        poor_text_pages += 1
                    else:
                        zero_text_pages += 1

                    if report.image_count > 0:
                        image_pages += 1
                except Exception:
                    poor_text_pages += 1

            avg_chars = round(total_chars / page_count, 2) if page_count > 0 else 0.0
            avg_words = round(total_words / page_count, 2) if page_count > 0 else 0.0

            return PDFProfile(
                file_size=file_size,
                page_count=page_count,
                page_dimensions=dimensions,
                text_layer_pages=text_layer_pages,
                poor_text_pages=poor_text_pages,
                image_pages=image_pages,
                zero_text_pages=zero_text_pages,
                average_characters_per_page=avg_chars,
                average_words_per_page=avg_words,
            )
