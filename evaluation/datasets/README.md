# Benchmark Datasets

This directory holds ground-truth dataset fixtures for document intelligence and OCR evaluation.

## Ground Truth Schema Format

Ground-truth files should follow the JSON schema structure defined in `evaluation/schemas/ground_truth.py`:

```json
{
  "document_id": "doc_sample",
  "filename": "sample_textbook.pdf",
  "page_count": 4,
  "pages": [
    {
      "page_number": 1,
      "width": 612.0,
      "height": 792.0,
      "text": "Full ground truth page text...",
      "regions": [
        {
          "region_id": "r001",
          "type": "HEADING",
          "bbox": [72.0, 72.0, 500.0, 100.0],
          "text": "1. Foundations of Computing",
          "reading_order": 1
        }
      ]
    }
  ],
  "chapters": [
    {
      "chapter_id": "ch_001",
      "number": 1,
      "title": "Foundations of Computing",
      "page_start": 1,
      "page_end": 2,
      "sections": [
        {
          "section_id": "ch_001_s001",
          "title": "1.1 Architecture Basics",
          "page_start": 1,
          "page_end": 1
        }
      ]
    }
  ],
  "chunks": [
    {
      "chunk_id": "chk_001",
      "chapter_id": "ch_001",
      "section_id": "ch_001_s001",
      "page_start": 1,
      "page_end": 1,
      "text": "Ground truth text content for chunk..."
    }
  ]
}
```
