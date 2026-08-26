from typing import Any, Optional
import Levenshtein
from evaluation.schemas.ground_truth import DocumentGT


def calculate_cer(reference: str, hypothesis: str) -> float:
    """
    Calculate Character Error Rate (CER) using Levenshtein distance:
    CER = Levenshtein_distance(ref, hyp) / max(1, len(ref))
    """
    if not reference and not hypothesis:
        return 0.0
    if not reference:
        return 1.0
    dist = Levenshtein.distance(reference, hypothesis)
    return round(float(dist) / float(len(reference)), 4)


def calculate_wer(reference: str, hypothesis: str) -> float:
    """
    Calculate Word Error Rate (WER) using word-level Levenshtein distance:
    WER = word_level_edit_distance(ref_words, hyp_words) / max(1, len(ref_words))
    """
    ref_words = reference.strip().split()
    hyp_words = hypothesis.strip().split()
    if not ref_words and not hyp_words:
        return 0.0
    if not ref_words:
        return 1.0

    dist = Levenshtein.distance(ref_words, hyp_words)
    return round(float(dist) / float(len(ref_words)), 4)


def calculate_normalized_edit_distance(reference: str, hypothesis: str) -> float:
    """
    Normalized Levenshtein Edit Distance in range [0.0, 1.0]:
    dist / max(len(ref), len(hyp), 1)
    """
    max_len = max(len(reference), len(hypothesis), 1)
    dist = Levenshtein.distance(reference, hypothesis)
    return round(float(dist) / float(max_len), 4)


class OCREvaluator:
    """
    Evaluates OCR accuracy against ground truth using CER, WER, and Normalized Edit Distance.
    Returns status='not_available' when ground truth is missing.
    """

    def evaluate(
        self,
        predicted_pages: list[dict[str, Any]],
        ground_truth: Optional[DocumentGT] = None,
    ) -> dict[str, Any]:
        if ground_truth is None or not ground_truth.pages:
            return {
                "status": "not_available",
                "ground_truth_available": False,
                "cer": None,
                "wer": None,
                "normalized_edit_distance": None,
                "character_accuracy": None,
                "word_accuracy": None,
                "total_evaluated_pages": 0,
                "page_metrics": [],
            }

        gt_page_map = {p.page_number: p for p in ground_truth.pages}
        page_metrics = []
        total_ref_chars = 0
        total_char_dist = 0
        total_ref_words = 0
        total_word_dist = 0

        for pred_page in predicted_pages:
            p_num = pred_page.get("page_number", 1)
            if p_num not in gt_page_map:
                continue

            gt_page = gt_page_map[p_num]
            # Construct text from GT page
            if gt_page.text:
                ref_text = gt_page.text
            else:
                ref_text = " ".join([r.text for r in gt_page.regions if r.text])

            # Construct text from predicted page regions
            regions = pred_page.get("regions", [])
            pred_text = " ".join([r.get("text", "") for r in regions if r.get("text")])

            p_cer = calculate_cer(ref_text, pred_text)
            p_wer = calculate_wer(ref_text, pred_text)
            p_ned = calculate_normalized_edit_distance(ref_text, pred_text)

            char_dist = Levenshtein.distance(ref_text, pred_text)
            ref_words = ref_text.strip().split()
            hyp_words = pred_text.strip().split()
            word_dist = Levenshtein.distance(ref_words, hyp_words)

            total_ref_chars += len(ref_text)
            total_char_dist += char_dist
            total_ref_words += len(ref_words)
            total_word_dist += word_dist

            page_metrics.append({
                "page_number": p_num,
                "cer": p_cer,
                "wer": p_wer,
                "normalized_edit_distance": p_ned,
                "ref_char_count": len(ref_text),
                "hyp_char_count": len(pred_text),
            })

        macro_cer = (
            round(sum(m["cer"] for m in page_metrics) / len(page_metrics), 4)
            if page_metrics
            else 0.0
        )
        macro_wer = (
            round(sum(m["wer"] for m in page_metrics) / len(page_metrics), 4)
            if page_metrics
            else 0.0
        )
        micro_cer = (
            round(float(total_char_dist) / float(max(1, total_ref_chars)), 4)
            if total_ref_chars > 0
            else 0.0
        )
        micro_wer = (
            round(float(total_word_dist) / float(max(1, total_ref_words)), 4)
            if total_ref_words > 0
            else 0.0
        )

        char_acc = round(max(0.0, 1.0 - micro_cer), 4)
        word_acc = round(max(0.0, 1.0 - micro_wer), 4)

        return {
            "status": "completed",
            "ground_truth_available": True,
            "cer": micro_cer,
            "wer": micro_wer,
            "macro_cer": macro_cer,
            "macro_wer": macro_wer,
            "character_accuracy": char_acc,
            "word_accuracy": word_acc,
            "total_evaluated_pages": len(page_metrics),
            "page_metrics": page_metrics,
        }
