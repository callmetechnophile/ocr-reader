from abc import ABC, abstractmethod
from typing import Any
from app.schemas.page import PageExtraction


class OCRExtractor(ABC):
    """Abstract interface for page text and region extractors."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the extractor (e.g. 'pdfplumber', 'cnn_ocr', 'baseline_ocr')."""
        pass

    @abstractmethod
    async def extract(self, page: Any, **kwargs: Any) -> PageExtraction:
        """
        Extract text, words, characters, tables, and bounding boxes from a page.

        Args:
            page: Either a pdfplumber Page object, or a rendered image (numpy ndarray),
                  depending on the extractor implementation.
            **kwargs: Additional parameters such as page_number, document_id, etc.

        Returns:
            PageExtraction containing un-normalized extracted elements.
        """
        pass


BaseExtractor = OCRExtractor
