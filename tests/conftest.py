from pathlib import Path
from typing import Generator
import cv2
import numpy as np
import pymupdf
import pytest
from app.core.config import settings
from app.storage.object_store import LocalFileSystemStore


@pytest.fixture
def temp_storage(tmp_path: Path) -> Generator[LocalFileSystemStore, None, None]:
    original_path = settings.STORAGE_PATH
    settings.STORAGE_PATH = str(tmp_path)
    store = LocalFileSystemStore(tmp_path)
    yield store
    settings.STORAGE_PATH = original_path


@pytest.fixture
def digital_pdf_path(tmp_path: Path) -> Path:
    """Create a multi-page digital PDF with clean text, headings, tables, and page numbers."""
    pdf_path = tmp_path / "digital_textbook.pdf"
    doc = pymupdf.open()

    # Page 1: Chapter 1 Digital Page
    page1 = doc.new_page(width=612, height=792)
    # Header
    page1.insert_text(pymupdf.Point(72, 36), "CHAPTER 1: INTRODUCTION TO SEMICONDUCTORS", fontsize=10)
    # Heading
    page1.insert_text(pymupdf.Point(72, 90), "1.1 MOSFET Fundamentals", fontsize=18)
    # Body text
    body_text = (
        "The metal-oxide-semiconductor field-effect transistor (MOSFET) is the primary "
        "building block of modern integrated circuits. In this chapter, we discuss the "
        "conduction mechanism and I-V characteristics."
    )
    page1.insert_textbox(pymupdf.Rect(72, 110, 540, 200), body_text, fontsize=11)

    # Subheading
    page1.insert_text(pymupdf.Point(72, 230), "1.1.1 Drain Current Equation", fontsize=14)
    # Equation
    page1.insert_text(pymupdf.Point(120, 260), "I_D = 0.5 * mu * C_ox * (W/L) * (V_GS - V_th)^2", fontsize=12)

    # Footer & Page Number
    page1.insert_text(pymupdf.Point(72, 756), "Semiconductor Devices and Circuits", fontsize=9)
    page1.insert_text(pymupdf.Point(540, 756), "1", fontsize=10)

    # Page 2: Two-column layout page
    page2 = doc.new_page(width=612, height=792)
    page2.insert_text(pymupdf.Point(72, 36), "CHAPTER 1: INTRODUCTION TO SEMICONDUCTORS", fontsize=10)
    page2.insert_text(pymupdf.Point(72, 80), "1.2 Two-Column Circuit Analysis", fontsize=16)

    # Left column
    col1_text = (
        "Left Column Text: Analysis of the subthreshold slope indicates exponential "
        "dependence on gate voltage. Leakage currents must be strictly managed in low power CMOS."
    )
    page2.insert_textbox(pymupdf.Rect(72, 110, 280, 400), col1_text, fontsize=10)

    # Right column
    col2_text = (
        "Right Column Text: High frequency operation requires minimizing parasitic capacitances "
        "between gate and drain terminals. S-parameter measurements confirm the model."
    )
    page2.insert_textbox(pymupdf.Rect(332, 110, 540, 400), col2_text, fontsize=10)
    page2.insert_text(pymupdf.Point(540, 756), "2", fontsize=10)

    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.fixture
def scanned_pdf_path(tmp_path: Path) -> Path:
    """Create a scanned image-only PDF with no embedded digital text layer."""
    pdf_path = tmp_path / "scanned_textbook.pdf"
    doc = pymupdf.open()

    page = doc.new_page(width=612, height=792)
    # Create synthetic image with text-like dark bars/boxes
    img_array = np.ones((792, 612, 3), dtype=np.uint8) * 255
    # Draw simulated lines
    img_array[100:130, 72:400] = 0
    img_array[160:180, 72:540] = 50
    img_array[200:220, 72:540] = 50
    img_array[240:260, 72:540] = 50

    # Encode as PNG and insert into page
    _, img_encoded = cv2.imencode(".png", img_array)
    page.insert_image(page.rect, stream=img_encoded.tobytes())

    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.fixture
def corrupted_text_pdf_path(tmp_path: Path) -> Path:
    """Create a PDF with corrupted/anomalous text (e.g. single-letter spaced text + low word length)."""
    pdf_path = tmp_path / "corrupted_text.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    # Spaced out 1-letter tokens (low avg word length anomaly)
    line = "a b c d e f g h i j k l m n o p q r s t u v w x y z " * 2
    for y_offset in range(100, 300, 30):
        page.insert_text(pymupdf.Point(72, y_offset), line, fontsize=10)

    # Insert a dummy image to simulate image-heavy scan with bad OCR text
    img_array = np.ones((400, 400, 3), dtype=np.uint8) * 200
    _, img_encoded = cv2.imencode(".png", img_array)
    page.insert_image(pymupdf.Rect(72, 350, 472, 750), stream=img_encoded.tobytes())

    doc.save(str(pdf_path))
    doc.close()
    return pdf_path
