# Textbook OCR Evaluation & Benchmark Harness

The `evaluation` harness is a decoupled, document-agnostic benchmarking and structural validation system for the Textbook OCR Microservice.

It is maintained separately in `evaluation_branch` so that evaluation dependencies and benchmarking logic do not bloat the production OCR runtime.

---

## 1. Branch Architecture

```text
main (Production OCR Core)
│
├── app/ (OCR service, PyMuPDF, pdfplumber, Layout CNN, CRNN, Chunking, TOON)
├── models/
├── scripts/
└── requirements.txt

evaluation (Evaluation & Benchmarks)
│
├── evaluation/
│   ├── metrics/ (OCR, Layout, Structure, Reading Order, Chunk, TOON)
│   ├── schemas/ (Ground Truth models)
│   ├── runners/ (CLI evaluate_document)
│   ├── reports/ (JSON and HTML report generators)
│   ├── datasets/
│   └── results/
└── requirements-evaluation.txt
```

---

## 2. Dependency Isolation

- **Core Runtime Dependencies** (`requirements.txt`):
  Keeps the core microservice lightweight without requiring evaluation packages.
  ```bash
  pip install -r requirements.txt
  ```

- **Evaluation Dependencies** (`requirements-evaluation.txt`):
  Installs core runtime packages plus isolated evaluation libraries:
  - `python-Levenshtein`: CER, WER, and normalized edit distance.
  - `apted`: Tree edit distance for document structural hierarchy trees.
  - `zss`: Zhang-Shasha tree edit distance algorithms.
  - `lxml`: Fast XML/HTML parsing for structured document comparison.
  - `Polygon3`: Optional polygon geometry calculations.
  ```bash
  pip install -r requirements-evaluation.txt
  ```

---

## 3. Dynamic Document Discovery

The evaluator makes zero assumptions about:
- Book title / filename
- Number of pages
- Number of chapters
- Number of sections
- Number of chunks

It automatically inspects any processed document directory:
```bash
python evaluation/runners/evaluate_document.py data/processed/<document_id>
```

It dynamically discovers:
- `manifest.json`
- `metadata.json`
- `report.json`
- `pages/*.json`
- `chapters/*.json`
- `chunks/*.jsonl`
- `*.toon` files

---

## 4. Evaluation Modes

### Mode 1: System Validation (Without Ground Truth)
Executed automatically when `--ground-truth` is not supplied.
Evaluates:
- Output completeness and provenance integrity
- Heading hierarchy and numbering consistency (Arabic/Roman/custom)
- Parent-child integrity and page boundary validation
- Reading order monotonicity & duplicate detection
- Chunk distribution, token sizes, orphan chunks, and empty chunks
- TOON format serialization and round-trip structural reconstruction
- Manifest consistency

*Note: In Mode 1, accuracy scores are never fabricated (`cer`, `wer`, `map_50` are reported as `null` or `"not_available"`).*

### Mode 2: Ground-Truth Evaluation (With Ground Truth)
Executed when `--ground-truth <path>` is supplied.
Evaluates:
- **OCR**: CER, WER, Normalized Edit Distance, Character & Word Accuracy.
- **Layout**: IoU, Precision, Recall, F1, and mAP across 10 document classes (`BODY`, `HEADING`, `SUBHEADING`, `EQUATION`, `TABLE`, `FIGURE`, `CAPTION`, `HEADER`, `FOOTER`, `PAGE_NUMBER`).
- **Structure**: APTED / ZSS Tree Edit Distance, Normalized Tree Similarity, Chapter Boundary Precision/Recall/F1.
- **Reading Order**: Pairwise ordering accuracy (Kendall's tau style) and exact page sequence matches.
- **Chunking**: Boundary Precision, Recall, F1, and Section Preservation Score.

---

## 5. CLI Usage & Examples

### Evaluate a Processed Document (System Validation)
```bash
python evaluation/runners/evaluate_document.py data/processed/doc_24a54049990766ee
```

### Evaluate with Ground Truth Dataset
```bash
python evaluation/runners/evaluate_document.py \
    data/processed/doc_24a54049990766ee \
    --ground-truth evaluation/datasets/sample_gt.json \
    --output data/evaluation_results/doc_24a54049990766ee \
    --html
```

---

## 6. Output Artifacts

Evaluation results are saved into `data/evaluation_results/<document_id>/`:
- `evaluation.json`: Complete unified benchmark summary
- `ocr_metrics.json`: OCR metrics breakdown
- `layout_metrics.json`: Per-class layout detection metrics
- `structure_metrics.json`: Tree edit distance and hierarchy consistency
- `reading_order_metrics.json`: Monotonicity and pairwise ordering scores
- `chunk_metrics.json`: Chunk distribution and boundary matching
- `toon_metrics.json`: TOON format validation and round-trip statistics
- `report.html`: Visual HTML report with responsive cards and metrics

---

## 7. Future Git Merge Workflow

When ready to merge the evaluation subsystem into `main`:

```bash
# 1. Switch to main branch
git checkout main

# 2. Merge evaluation branch without modifying core runtime files
git merge evaluation

# 3. Verify core tests still pass without evaluation requirements
pytest -v
```
