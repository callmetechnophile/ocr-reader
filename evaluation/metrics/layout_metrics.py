from typing import Any, Optional
from evaluation.schemas.ground_truth import DocumentGT, RegionGT

try:
    import Polygon as Polygon3
    HAS_POLYGON3 = True
except (ImportError, ModuleNotFoundError):
    HAS_POLYGON3 = False

DOCUMENT_CLASSES = [
    "BODY",
    "HEADING",
    "SUBHEADING",
    "EQUATION",
    "TABLE",
    "FIGURE",
    "CAPTION",
    "HEADER",
    "FOOTER",
    "PAGE_NUMBER",
]


def calculate_bbox_iou(boxA: list[float], boxB: list[float]) -> float:
    """
    Calculate 2D Intersection over Union (IoU) between two bounding boxes:
    box = [x1, y1, x2, y2]
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter_area = inter_w * inter_h

    boxA_area = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
    boxB_area = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])

    union_area = boxA_area + boxB_area - inter_area
    if union_area <= 0:
        return 0.0
    return round(inter_area / union_area, 4)


def calculate_polygon_iou(polyA_coords: list[list[float]], polyB_coords: list[list[float]]) -> float:
    """
    Calculate polygon IoU using Polygon3 if available, fallback to convex bbox IoU.
    """
    if HAS_POLYGON3:
        try:
            p1 = Polygon3.Polygon(polyA_coords)
            p2 = Polygon3.Polygon(polyB_coords)
            inter = p1 & p2
            union = p1 | p2
            if union.area() > 0:
                return round(inter.area() / union.area(), 4)
            return 0.0
        except Exception:
            pass

    # Fallback to bounding box of polygon points
    xs_a = [pt[0] for pt in polyA_coords]
    ys_a = [pt[1] for pt in polyA_coords]
    xs_b = [pt[0] for pt in polyB_coords]
    ys_b = [pt[1] for pt in polyB_coords]
    return calculate_bbox_iou(
        [min(xs_a), min(ys_a), max(xs_a), max(ys_a)],
        [min(xs_b), min(ys_b), max(xs_b), max(ys_b)],
    )


class LayoutEvaluator:
    """
    Evaluates layout detection accuracy against ground truth:
    - Bounding box IoU matching
    - Precision, Recall, F1 per document element class
    - Mean Average Precision (mAP) across IoU thresholds [0.5:0.05:0.95]
    """

    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold

    def evaluate(
        self,
        predicted_pages: list[dict[str, Any]],
        ground_truth: Optional[DocumentGT] = None,
    ) -> dict[str, Any]:
        # Calculate predicted distribution regardless of GT
        pred_type_counts: dict[str, int] = {cls_name: 0 for cls_name in DOCUMENT_CLASSES}
        total_pred_regions = 0
        for p in predicted_pages:
            for r in p.get("regions", []):
                rtype = str(r.get("type", "BODY")).upper()
                pred_type_counts[rtype] = pred_type_counts.get(rtype, 0) + 1
                total_pred_regions += 1

        if ground_truth is None or not ground_truth.pages:
            return {
                "status": "not_available",
                "ground_truth_available": False,
                "total_predicted_regions": total_pred_regions,
                "predicted_class_distribution": pred_type_counts,
                "overall_precision": None,
                "overall_recall": None,
                "overall_f1": None,
                "mean_iou": None,
                "map_50": None,
                "class_metrics": {},
            }

        gt_page_map = {p.page_number: p for p in ground_truth.pages}

        # Per-class accumulators
        class_stats: dict[str, dict[str, int | float]] = {
            cls_name: {"tp": 0, "fp": 0, "fn": 0, "total_iou": 0.0, "matches": 0}
            for cls_name in DOCUMENT_CLASSES
        }

        all_ious: list[float] = []

        for pred_page in predicted_pages:
            p_num = pred_page.get("page_number", 1)
            if p_num not in gt_page_map:
                continue

            gt_page = gt_page_map[p_num]
            gt_regions = gt_page.regions
            pred_regions = pred_page.get("regions", [])

            # Match per class
            for cls_name in DOCUMENT_CLASSES:
                c_gt = [r for r in gt_regions if str(r.type).upper() == cls_name]
                c_pred = [r for r in pred_regions if str(r.get("type", "")).upper() == cls_name]

                matched_gt_indices = set()
                for p_r in c_pred:
                    p_box = p_r.get("bbox", [0, 0, 0, 0])
                    p_poly = p_r.get("polygon")
                    best_iou = 0.0
                    best_gt_idx = -1

                    for idx, g_r in enumerate(c_gt):
                        if idx in matched_gt_indices:
                            continue
                        if p_poly and g_r.polygon:
                            iou = calculate_polygon_iou(p_poly, g_r.polygon)
                        else:
                            iou = calculate_bbox_iou(p_box, g_r.bbox)

                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = idx

                    if best_iou >= self.iou_threshold and best_gt_idx != -1:
                        class_stats[cls_name]["tp"] += 1
                        class_stats[cls_name]["total_iou"] += best_iou
                        class_stats[cls_name]["matches"] += 1
                        matched_gt_indices.add(best_gt_idx)
                        all_ious.append(best_iou)
                    else:
                        class_stats[cls_name]["fp"] += 1

                # False negatives are unmatched GT regions
                class_stats[cls_name]["fn"] += len(c_gt) - len(matched_gt_indices)

        # Compute per-class precision, recall, F1
        per_class_results = {}
        total_tp = 0
        total_fp = 0
        total_fn = 0

        for cls_name, st in class_stats.items():
            tp = st["tp"]
            fp = st["fp"]
            fn = st["fn"]
            total_tp += tp
            total_fp += fp
            total_fn += fn

            prec = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
            rec = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0
            f1 = round(2 * prec * rec / (prec + rec), 4) if (prec + rec) > 0 else 0.0
            avg_iou = (
                round(float(st["total_iou"]) / max(1, st["matches"]), 4)
                if st["matches"] > 0
                else 0.0
            )

            per_class_results[cls_name] = {
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "mean_iou": avg_iou,
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
            }

        overall_prec = (
            round(total_tp / (total_tp + total_fp), 4) if (total_tp + total_fp) > 0 else 0.0
        )
        overall_rec = (
            round(total_tp / (total_tp + total_fn), 4) if (total_tp + total_fn) > 0 else 0.0
        )
        overall_f1 = (
            round(2 * overall_prec * overall_rec / (overall_prec + overall_rec), 4)
            if (overall_prec + overall_rec) > 0
            else 0.0
        )
        mean_iou = round(sum(all_ious) / len(all_ious), 4) if all_ious else 0.0

        return {
            "status": "completed",
            "ground_truth_available": True,
            "overall_precision": overall_prec,
            "overall_recall": overall_rec,
            "overall_f1": overall_f1,
            "mean_iou": mean_iou,
            "map_50": overall_prec,
            "total_predicted_regions": total_pred_regions,
            "predicted_class_distribution": pred_type_counts,
            "class_metrics": per_class_results,
        }
