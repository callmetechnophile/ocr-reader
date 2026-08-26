from typing import Any, Optional
from evaluation.schemas.ground_truth import DocumentGT


def calculate_pairwise_ordering_accuracy(pred_order: list[str], gt_order: list[str]) -> float:
    """
    Calculate pairwise ordering consistency between predicted and ground-truth element sequences.
    Fraction of pairs (A, B) where order(A) < order(B) agrees in both sequences.
    """
    common_items = [x for x in pred_order if x in gt_order]
    if len(common_items) < 2:
        return 1.0

    gt_pos = {item: idx for idx, item in enumerate(gt_order)}
    pred_pos = {item: idx for idx, item in enumerate(common_items)}

    total_pairs = 0
    concordant_pairs = 0

    items = list(common_items)
    n = len(items)
    for i in range(n):
        for j in range(i + 1, n):
            item_a = items[i]
            item_b = items[j]
            total_pairs += 1

            pred_before = pred_pos[item_a] < pred_pos[item_b]
            gt_before = gt_pos[item_a] < gt_pos[item_b]

            if pred_before == gt_before:
                concordant_pairs += 1

    return round(concordant_pairs / max(1, total_pairs), 4)


class ReadingOrderEvaluator:
    """
    Evaluates reading order quality:
    - Internal monotonicity & inversion heuristics (Mode 1: System Validation)
    - Pairwise ordering accuracy & exact sequence match (Mode 2: GT Evaluation)
    """

    def evaluate(
        self,
        predicted_pages: list[dict[str, Any]],
        ground_truth: Optional[DocumentGT] = None,
    ) -> dict[str, Any]:
        duplicate_indices_count = 0
        non_monotonic_pages = 0
        total_pages = len(predicted_pages)
        likely_inversions = 0

        for p in predicted_pages:
            regions = p.get("regions", [])
            orders = [r.get("reading_order", 0) for r in regions]

            if len(orders) != len(set(orders)):
                duplicate_indices_count += 1
            if orders != sorted(orders):
                non_monotonic_pages += 1

            # Heuristic checks on consecutive regions: heading followed by content vs figure/caption
            for i in range(len(regions) - 1):
                r1 = regions[i]
                r2 = regions[i + 1]
                b1 = r1.get("bbox", [0, 0, 0, 0])
                b2 = r2.get("bbox", [0, 0, 0, 0])
                # If vertical inversion (lower region appears before higher region in single column)
                if b1[0] < 300 and b2[0] < 300 and b1[1] > b2[3] + 50:
                    likely_inversions += 1

        consistency_score = round(
            max(0.0, 1.0 - ((non_monotonic_pages + duplicate_indices_count) / max(1, total_pages * 2))),
            4,
        )

        res: dict[str, Any] = {
            "total_evaluated_pages": total_pages,
            "duplicate_order_pages": duplicate_indices_count,
            "non_monotonic_pages": non_monotonic_pages,
            "likely_inversions_count": likely_inversions,
            "reading_order_consistency_score": consistency_score,
        }

        if ground_truth is None or not ground_truth.pages:
            res.update({
                "status": "completed_mode_1_system_validation",
                "ground_truth_available": False,
                "ordering_accuracy": None,
                "pairwise_ordering_accuracy": None,
            })
            return res

        # Mode 2: Compare against Ground Truth reading orders
        gt_page_map = {p.page_number: p for p in ground_truth.pages}
        pairwise_scores = []
        exact_matches = 0
        evaluated_gt_pages = 0

        for p in predicted_pages:
            p_num = p.get("page_number", 1)
            if p_num not in gt_page_map:
                continue

            evaluated_gt_pages += 1
            gt_p = gt_page_map[p_num]
            pred_order_texts = [r.get("text", "").strip() for r in p.get("regions", []) if r.get("text")]
            gt_order_texts = [r.text.strip() for r in gt_p.regions if r.text]

            score = calculate_pairwise_ordering_accuracy(pred_order_texts, gt_order_texts)
            pairwise_scores.append(score)

            if pred_order_texts == gt_order_texts:
                exact_matches += 1

        mean_pairwise = (
            round(sum(pairwise_scores) / len(pairwise_scores), 4) if pairwise_scores else 1.0
        )
        exact_acc = round(exact_matches / max(1, evaluated_gt_pages), 4)

        res.update({
            "status": "completed_mode_2_ground_truth",
            "ground_truth_available": True,
            "ordering_accuracy": exact_acc,
            "pairwise_ordering_accuracy": mean_pairwise,
            "exact_order_match_pages": exact_matches,
            "evaluated_gt_pages": evaluated_gt_pages,
        })
        return res
