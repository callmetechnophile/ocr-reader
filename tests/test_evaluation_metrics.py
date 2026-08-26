import pytest
from evaluation.metrics import (
    ChunkEvaluator,
    LayoutEvaluator,
    OCREvaluator,
    ReadingOrderEvaluator,
    StructureEvaluator,
    ToonEvaluator,
)
from evaluation.metrics.ocr_metrics import calculate_cer, calculate_normalized_edit_distance, calculate_wer
from evaluation.metrics.layout_metrics import calculate_bbox_iou
from evaluation.metrics.reading_order_metrics import calculate_pairwise_ordering_accuracy
from evaluation.schemas.ground_truth import (
    ChapterGT,
    ChunkGT,
    DocumentGT,
    PageGT,
    RegionGT,
    SectionGT,
)


def test_ocr_metrics_calculations():
    ref = "Machine Learning Foundations"
    hyp = "Machine Learning Fundations"  # 1 substitution

    cer = calculate_cer(ref, hyp)
    assert 0.0 < cer < 0.1
    wer = calculate_wer(ref, hyp)
    assert wer == pytest.approx(1.0 / 3.0, rel=1e-3)
    ned = calculate_normalized_edit_distance(ref, hyp)
    assert 0.0 < ned < 0.1

    # Exact match
    assert calculate_cer(ref, ref) == 0.0
    assert calculate_wer(ref, ref) == 0.0
    assert calculate_normalized_edit_distance(ref, ref) == 0.0


def test_ocr_evaluator_without_and_with_gt():
    evaluator = OCREvaluator()
    pred_pages = [
        {"page_number": 1, "regions": [{"text": "Hello world"}]}
    ]

    # Without GT
    res_no_gt = evaluator.evaluate(pred_pages, None)
    assert res_no_gt["status"] == "not_available"
    assert res_no_gt["cer"] is None

    # With GT
    gt = DocumentGT(
        pages=[
            PageGT(page_number=1, text="Hello world")
        ]
    )
    res_gt = evaluator.evaluate(pred_pages, gt)
    assert res_gt["status"] == "completed"
    assert res_gt["cer"] == 0.0
    assert res_gt["character_accuracy"] == 1.0


def test_layout_evaluator_iou_and_matching():
    boxA = [10.0, 10.0, 50.0, 50.0]
    boxB = [10.0, 10.0, 50.0, 50.0]
    assert calculate_bbox_iou(boxA, boxB) == 1.0

    boxC = [30.0, 10.0, 70.0, 50.0]
    iou = calculate_bbox_iou(boxA, boxC)
    assert 0.3 < iou < 0.4

    evaluator = LayoutEvaluator(iou_threshold=0.5)
    pred_pages = [
        {
            "page_number": 1,
            "regions": [
                {"type": "HEADING", "bbox": [10.0, 10.0, 50.0, 50.0]},
                {"type": "BODY", "bbox": [10.0, 60.0, 50.0, 100.0]},
            ],
        }
    ]

    gt = DocumentGT(
        pages=[
            PageGT(
                page_number=1,
                regions=[
                    RegionGT(region_id="r1", type="HEADING", bbox=[10.0, 10.0, 50.0, 50.0]),
                    RegionGT(region_id="r2", type="BODY", bbox=[10.0, 60.0, 50.0, 100.0]),
                ],
            )
        ]
    )

    res = evaluator.evaluate(pred_pages, gt)
    assert res["status"] == "completed"
    assert res["overall_precision"] == 1.0
    assert res["overall_recall"] == 1.0
    assert res["overall_f1"] == 1.0


def test_structure_evaluator_tree_similarity():
    evaluator = StructureEvaluator()
    pred_chapters = [
        {
            "chapter_id": "ch_001",
            "title": "Chapter 1",
            "page_start": 1,
            "page_end": 2,
            "sections": [{"section_id": "s1", "title": "Sec 1.1"}],
        }
    ]
    pred_pages = [{"page_number": 1}, {"page_number": 2}]
    pred_chunks = [{"chunk_id": "c1", "chapter_id": "ch_001", "page_start": 1, "page_end": 1}]

    # Without GT
    res_no_gt = evaluator.evaluate(pred_chapters, pred_pages, pred_chunks, None)
    assert res_no_gt["parent_child_integrity"] == 1.0
    assert res_no_gt["page_boundary_integrity"] == 1.0

    # With GT
    gt = DocumentGT(
        chapters=[
            ChapterGT(
                chapter_id="ch_001",
                title="Chapter 1",
                page_start=1,
                page_end=2,
                sections=[SectionGT(section_id="s1", title="Sec 1.1", page_start=1, page_end=1)],
            )
        ]
    )
    res_gt = evaluator.evaluate(pred_chapters, pred_pages, pred_chunks, gt)
    assert res_gt["status"] == "completed_mode_2_ground_truth"
    assert res_gt["normalized_tree_similarity"] == 1.0
    assert res_gt["chapter_boundary_f1"] == 1.0


def test_reading_order_pairwise_accuracy():
    pred_order = ["A", "B", "C", "D"]
    gt_order = ["A", "B", "C", "D"]
    assert calculate_pairwise_ordering_accuracy(pred_order, gt_order) == 1.0

    inverted_order = ["D", "C", "B", "A"]
    assert calculate_pairwise_ordering_accuracy(inverted_order, gt_order) == 0.0

    evaluator = ReadingOrderEvaluator()
    pred_pages = [
        {
            "page_number": 1,
            "regions": [
                {"text": "A", "reading_order": 1, "bbox": [10, 10, 50, 20]},
                {"text": "B", "reading_order": 2, "bbox": [10, 30, 50, 40]},
            ],
        }
    ]
    gt = DocumentGT(
        pages=[
            PageGT(
                page_number=1,
                regions=[
                    RegionGT(region_id="r1", text="A", bbox=[10, 10, 50, 20]),
                    RegionGT(region_id="r2", text="B", bbox=[10, 30, 50, 40]),
                ],
            )
        ]
    )
    res = evaluator.evaluate(pred_pages, gt)
    assert res["pairwise_ordering_accuracy"] == 1.0
    assert res["exact_order_match_pages"] == 1


def test_chunk_evaluator():
    evaluator = ChunkEvaluator()
    pred_chunks = [
        {"chunk_id": "c1", "chapter_id": "ch_001", "section_id": "s1", "token_count": 100, "page_start": 1, "page_end": 1, "text": "Chunk text 1"},
        {"chunk_id": "c2", "chapter_id": "ch_001", "section_id": "s1", "token_count": 150, "page_start": 2, "page_end": 2, "text": "Chunk text 2"},
    ]
    pred_chapters = [{"chapter_id": "ch_001"}]

    res_no_gt = evaluator.evaluate(pred_chunks, pred_chapters, max_page=2, ground_truth=None)
    assert res_no_gt["total_chunks"] == 2
    assert res_no_gt["average_token_size"] == 125.0
    assert res_no_gt["chunk_integrity_score"] == 1.0

    gt = DocumentGT(
        chunks=[
            ChunkGT(chunk_id="c1", chapter_id="ch_001", section_id="s1", page_start=1, page_end=1, text="Chunk text 1"),
            ChunkGT(chunk_id="c2", chapter_id="ch_001", section_id="s1", page_start=2, page_end=2, text="Chunk text 2"),
        ]
    )
    res_gt = evaluator.evaluate(pred_chunks, pred_chapters, max_page=2, ground_truth=gt)
    assert res_gt["boundary_f1"] == 1.0


def test_toon_evaluator_valid_and_missing(tmp_path):
    evaluator = ToonEvaluator()

    # Missing file
    res_missing = evaluator.evaluate(tmp_path / "nonexistent.toon", "doc_123")
    assert res_missing["status"] == "not_available"
    assert res_missing["toon_file_found"] is False

    # Valid TOON
    toon_file = tmp_path / "sample.toon"
    toon_content = {
        "format": "TOON_V1",
        "document_id": "doc_123",
        "pages": [{"page_number": 1, "regions": []}],
        "structure": {"chapters": [{"chapter_id": "ch_001", "title": "Chapter 1", "sections": []}]},
        "chunks": [{"chunk_id": "chk_001", "text": "Sample text"}],
    }
    import json
    with open(toon_file, "w", encoding="utf-8") as f:
        json.dump(toon_content, f)

    res_valid = evaluator.evaluate(toon_file, "doc_123")
    assert res_valid["valid"] is True
    assert res_valid["round_trip_validation"] is True
