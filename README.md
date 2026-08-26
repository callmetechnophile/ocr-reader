# Textbook OCR Microservice & Benchmark V0

A high-performance, document-agnostic **Textbook OCR & Document Ingestion System** built with FastAPI, PyMuPDF, pdfplumber, OpenCV, and PyTorch (CRNN OCR).

---

## 1. Architecture Overview

The system accepts arbitrary textbook PDFs, adaptively profiles each page, routes to high-fidelity digital extraction (`pdfplumber`) or scanned/poor-text OCR (`CRNN Recognizer`), detects document structure (chapters and sections), and generates token-bounded semantic chunks:

```text
                     ARBITRARY PDF TEXTBOOK
                               │
                               ▼
               Stable Content-Based Document ID
                 (doc_ + sha256[:16] isolation)
                               │
                               ▼
                        PDF Analyzer
                 (Quality Score & Page Profiler)
                               │
                ┌──────────────┴──────────────┐
                ▼                             ▼
          Digital Text                   Poor / Scanned
                │                             │
                ▼                             ▼
           pdfplumber                   Page Renderer
                                              │
                                              ▼
                                       Layout Detector
                                              │
                                              ▼
                                         Text Regions
                                              │
                                              ▼
                                           CRNN OCR
                └──────────────┬──────────────┘
                               ▼
                      Normalized Page JSON
                               │
                               ▼
                       Chapter Detection
                               │
                               ▼
                       Section Detection
                               │
                               ▼
                            Chunking
                    (300–800 tokens + overlap)
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
               Chapter JSON          Chunk JSONL
                    │                     │
                    └──────────┬──────────┘
                               ▼
                  Manifest, Report, Performance
```

---

## 2. Directory Layout per Document

For every processed document, all artifacts are saved under an isolated directory `data/processed/{document_id}/`:

```text
data/processed/{document_id}/
├── metadata.json           # File metadata, SHA-256, page count, PDF title
├── profile.json            # High-level PDF quality profiling breakdown
├── manifest.json           # Dynamic index of all pages, chapters, sections, and chunks
├── job.json                # Atomic local job status for background progress tracking
├── report.json             # Execution summary, routing/extraction breakdown
├── performance.json        # Detailed timing benchmarks and average page duration
├── errors.jsonl            # Isolated per-page failure logs (if any)
│
├── pages/                  # Normalized per-page JSON (bounding boxes, reading order, provenance)
│   ├── 0001.json
│   ├── 0002.json
│   └── ...
│
├── chapters/               # Chapter boundaries and nested sections
│   ├── ch_001.json
│   ├── ch_002.json
│   └── ...
│
├── chunks/                 # Structure-aware chunks (300-800 tokens, 1 chunk per JSONL line)
│   ├── ch_001.jsonl
│   ├── ch_002.jsonl
│   └── ...
│
└── debug/                  # Visual overlays with bounding boxes & labels (when --debug is active)
    ├── page_0001.png
    ├── page_0001_text.txt
    └── ...
```

---

## 3. Quickstart & Benchmark CLI

The CLI is completely document-agnostic and supports user-configurable input, output, debug, and TOON directories across Windows, Linux, and macOS.

### Generic Usage Examples

```bash
# 1. Basic Document Processing
python scripts/test_book.py "<PDF_PATH>"

# 2. Generate Canonical TOON File in Custom Directory
python scripts/test_book.py "<PDF_PATH>" --toon "<TOON_OUTPUT_DIRECTORY>"

# 3. Generate TOON File and Visual Debug Overlays
python scripts/test_book.py "<PDF_PATH>" --toon "<TOON_OUTPUT_DIRECTORY>" --debug

# 4. Custom Processed Output Directory
python scripts/test_book.py "<PDF_PATH>" --output "<PROCESSED_OUTPUT_DIRECTORY>"

# 5. Full Custom Configuration (Input, Processed, TOON, Debug, Overwrite Force)
python scripts/test_book.py \
    "<PDF_PATH>" \
    --output "<PROCESSED_OUTPUT_DIRECTORY>" \
    --toon "<TOON_OUTPUT_DIRECTORY>" \
    --debug \
    --debug-output "<DEBUG_OUTPUT_DIRECTORY>" \
    --force

# 6. Specific Page Range
python scripts/test_book.py "<PDF_PATH>" --start-page 1 --end-page 25
```

### TOON Naming & Overwrite Protection
- The `.toon` filename is dynamically derived from the PDF stem: `<pdf_stem>.toon` (e.g., `Engineering Mathematics.pdf` $\rightarrow$ `Engineering Mathematics.toon`, `book.final.v2.pdf` $\rightarrow$ `book.final.v2.toon`).
- If `<TOON_OUTPUT_DIRECTORY>/<pdf_stem>.toon` already exists, the file is safely preserved unless `--force` is provided.
- Target directories are automatically created if they do not exist.

---

## 4. REST API Endpoints

Start the FastAPI service:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

- `POST /v1/documents`: Upload an arbitrary PDF. Returns content-derived `document_id` and queues processing.
- `GET /v1/documents/{document_id}`: Query live job progress, processed/failed page counts, and status (`queued`, `processing`, `completed`, `completed_with_errors`, `failed`).
- `GET /v1/documents/{document_id}/manifest`: Retrieve the completed document manifest.
- `GET /v1/documents/{document_id}/pages/{page_number}`: Retrieve normalized JSON for a single page.
- `GET /health` & `GET /ready`: System health and readiness probes.

---

## 5. Automated Tests

```bash
pytest -v
```
