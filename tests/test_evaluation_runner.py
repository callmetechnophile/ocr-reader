import json
from pathlib import Path
import pytest
from evaluation.runners.evaluate_document import (
    discover_processed_document,
    load_ground_truth,
    run_evaluation,
)


def test_discover_processed_document(tmp_path: Path):
    doc_dir = tmp_path / "doc_test_123"
    doc_dir.mkdir(parents=True)
    (doc_dir / "pages").mkdir()
    (doc_dir / "chapters").mkdir()
    (doc_dir / "chunks").mkdir()

    with open(doc_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump({"document_id": "doc_test_123", "filename": "sample.pdf", "page_count": 1}, f)

    with open(doc_dir / "pages" / "0001.json", "w", encoding="utf-8") as f:
        json.dump({"page_number": 1, "regions": []}, f)

    with open(doc_dir / "chapters" / "ch_001.json", "w", encoding="utf-8") as f:
        json.dump({"chapter_id": "ch_001", "title": "Chapter 1", "sections": []}, f)

    with open(doc_dir / "chunks" / "ch_001.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"chunk_id": "c1", "chapter_id": "ch_001", "text": "Sample text", "token_count": 50}) + "\n")

    discovered = discover_processed_document(doc_dir)
    assert discovered["document_id"] == "doc_test_123"
    assert discovered["filename"] == "sample.pdf"
    assert len(discovered["pages"]) == 1
    assert len(discovered["chapters"]) == 1
    assert len(discovered["chunks"]) == 1


def test_run_evaluation_mode_1_and_mode_2(tmp_path: Path):
    # Setup processed document directory
    doc_dir = tmp_path / "processed" / "doc_abc"
    doc_dir.mkdir(parents=True)
    (doc_dir / "pages").mkdir()
    (doc_dir / "chapters").mkdir()
    (doc_dir / "chunks").mkdir()

    with open(doc_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump({"document_id": "doc_abc", "filename": "book.pdf", "page_count": 1}, f)

    with open(doc_dir / "pages" / "0001.json", "w", encoding="utf-8") as f:
        json.dump({
            "page_number": 1,
            "regions": [
                {"type": "HEADING", "bbox": [10, 10, 100, 30], "text": "Intro to Robotics", "reading_order": 1}
            ]
        }, f)

    with open(doc_dir / "chapters" / "ch_001.json", "w", encoding="utf-8") as f:
        json.dump({"chapter_id": "ch_001", "title": "Intro to Robotics", "page_start": 1, "page_end": 1, "sections": []}, f)

    with open(doc_dir / "chunks" / "ch_001.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"chunk_id": "c1", "chapter_id": "ch_001", "text": "Intro to Robotics", "token_count": 30, "page_start": 1, "page_end": 1}) + "\n")

    out_mode1 = tmp_path / "eval_mode1"
    # Mode 1: System Validation
    res1 = run_evaluation(doc_dir, ground_truth_path=None, output_dir=out_mode1, generate_html=True)
    assert res1["evaluation_mode"] == "SYSTEM_VALIDATION"
    assert res1["ground_truth_available"] is False
    assert (out_mode1 / "evaluation.json").exists()
    assert (out_mode1 / "report.html").exists()
    assert (out_mode1 / "ocr_metrics.json").exists()

    # Mode 2: Ground Truth
    gt_file = tmp_path / "ground_truth.json"
    gt_payload = {
        "document_id": "doc_abc",
        "pages": [
            {
                "page_number": 1,
                "text": "Intro to Robotics",
                "regions": [
                    {"region_id": "r1", "type": "HEADING", "bbox": [10, 10, 100, 30], "text": "Intro to Robotics"}
                ]
            }
        ],
        "chapters": [
            {"chapter_id": "ch_001", "title": "Intro to Robotics", "page_start": 1, "page_end": 1, "sections": []}
        ],
        "chunks": [
            {"chunk_id": "c1", "chapter_id": "ch_001", "text": "Intro to Robotics", "page_start": 1, "page_end": 1}
        ]
    }
    with open(gt_file, "w", encoding="utf-8") as f:
        json.dump(gt_payload, f)

    out_mode2 = tmp_path / "eval_mode2"
    res2 = run_evaluation(doc_dir, ground_truth_path=gt_file, output_dir=out_mode2, generate_html=True)
    assert res2["evaluation_mode"] == "GROUND_TRUTH_EVALUATION"
    assert res2["ground_truth_available"] is True
    assert res2["ocr_metrics"]["cer"] == 0.0
    assert res2["layout_metrics"]["overall_f1"] == 1.0
    assert res2["structure_metrics"]["normalized_tree_similarity"] == 1.0
