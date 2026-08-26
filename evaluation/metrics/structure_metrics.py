from typing import Any, Optional
import zss
from apted import APTED, Config
from evaluation.schemas.ground_truth import ChapterGT, DocumentGT


class DocumentTreeNode:
    def __init__(self, label: str):
        self.label = label
        self.children: list["DocumentTreeNode"] = []

    def add_child(self, child: "DocumentTreeNode"):
        self.children.append(child)

    @staticmethod
    def get_children(node: "DocumentTreeNode") -> list["DocumentTreeNode"]:
        return node.children

    @staticmethod
    def get_label(node: "DocumentTreeNode") -> str:
        return node.label


def build_zss_tree_from_chapters(chapters: list[Any]) -> DocumentTreeNode:
    root = DocumentTreeNode("DOCUMENT")
    for ch in chapters:
        ch_title = getattr(ch, "title", "") or (ch.get("title") if isinstance(ch, dict) else "")
        ch_node = DocumentTreeNode(f"CH:{ch_title}")
        sections = (
            getattr(ch, "sections", [])
            if hasattr(ch, "sections")
            else ch.get("sections", [])
            if isinstance(ch, dict)
            else []
        )
        for s in sections:
            s_title = (
                getattr(s, "title", "")
                if hasattr(s, "title")
                else s.get("title", "")
                if isinstance(s, dict)
                else ""
            )
            ch_node.add_child(DocumentTreeNode(f"SEC:{s_title}"))
        root.add_child(ch_node)
    return root


def build_bracket_tree(chapters: list[Any]) -> str:
    """Build bracket tree notation string for APTED, e.g. {DOCUMENT{CH1{SEC1}}{CH2}}"""

    def clean_str(s: str) -> str:
        return s.replace("{", "(").replace("}", ")").replace(" ", "_")[:30]

    parts = ["{DOCUMENT"]
    for ch in chapters:
        ch_title = getattr(ch, "title", "") or (ch.get("title") if isinstance(ch, dict) else "CH")
        parts.append(f"{{{clean_str(ch_title)}")
        sections = (
            getattr(ch, "sections", [])
            if hasattr(ch, "sections")
            else ch.get("sections", [])
            if isinstance(ch, dict)
            else []
        )
        for s in sections:
            s_title = (
                getattr(s, "title", "")
                if hasattr(s, "title")
                else s.get("title", "")
                if isinstance(s, dict)
                else "SEC"
            )
            parts.append(f"{{{clean_str(s_title)}}}")
        parts.append("}")
    parts.append("}")
    return "".join(parts)


def compute_tree_edit_distance_zss(tree1: DocumentTreeNode, tree2: DocumentTreeNode) -> int:
    return zss.simple_distance(
        tree1,
        tree2,
        DocumentTreeNode.get_children,
        DocumentTreeNode.get_label,
    )


def compute_tree_edit_distance_apted(bracket1: str, bracket2: str) -> float:
    try:
        apted = APTED(bracket1, bracket2)
        return float(apted.compute_edit_distance())
    except Exception:
        return -1.0


class StructureEvaluator:
    """
    Evaluates document structural quality:
    - Internal hierarchy, boundary, and relationship consistency (Mode 1: System Validation)
    - Tree edit distance & parent-child accuracy when ground truth is provided (Mode 2: GT Evaluation)
    """

    def evaluate(
        self,
        predicted_chapters: list[dict[str, Any]],
        predicted_pages: list[dict[str, Any]],
        predicted_chunks: list[dict[str, Any]],
        ground_truth: Optional[DocumentGT] = None,
    ) -> dict[str, Any]:
        # 1. System Validation Consistency Metrics
        total_ch = len(predicted_chapters)
        total_sec = sum(len(ch.get("sections", [])) for ch in predicted_chapters)
        max_page = max([p.get("page_number", 1) for p in predicted_pages]) if predicted_pages else 1

        boundary_violations = 0
        for ch in predicted_chapters:
            p_start = ch.get("page_start", ch.get("start_page", 1))
            p_end = ch.get("page_end", ch.get("end_page", p_start))
            if p_start < 1 or p_end > max_page or p_start > p_end:
                boundary_violations += 1

        orphan_chunks = 0
        ch_ids = {ch.get("chapter_id") for ch in predicted_chapters}
        for chunk in predicted_chunks:
            if chunk.get("chapter_id") not in ch_ids:
                orphan_chunks += 1

        parent_child_score = round(max(0.0, 1.0 - (orphan_chunks / max(1, len(predicted_chunks)))), 4) if predicted_chunks else 1.0
        boundary_score = round(max(0.0, 1.0 - (boundary_violations / max(1, total_ch))), 4) if total_ch else 1.0

        res: dict[str, Any] = {
            "total_predicted_chapters": total_ch,
            "total_predicted_sections": total_sec,
            "parent_child_integrity": parent_child_score,
            "page_boundary_integrity": boundary_score,
            "orphan_chunks_count": orphan_chunks,
            "boundary_violations_count": boundary_violations,
        }

        if ground_truth is None or not ground_truth.chapters:
            res.update({
                "status": "completed_mode_1_system_validation",
                "ground_truth_available": False,
                "tree_edit_distance": None,
                "normalized_tree_similarity": None,
                "chapter_precision": None,
                "chapter_recall": None,
                "chapter_f1": None,
            })
            return res

        # Mode 2: Compare against Ground Truth
        gt_chapters = ground_truth.chapters
        gt_total_ch = len(gt_chapters)
        gt_total_sec = sum(len(ch.sections) for ch in gt_chapters)

        # ZSS Tree Edit Distance
        t_pred = build_zss_tree_from_chapters(predicted_chapters)
        t_gt = build_zss_tree_from_chapters(gt_chapters)
        zss_ted = compute_tree_edit_distance_zss(t_pred, t_gt)

        # APTED Tree Edit Distance
        b_pred = build_bracket_tree(predicted_chapters)
        b_gt = build_bracket_tree(gt_chapters)
        apted_ted = compute_tree_edit_distance_apted(b_pred, b_gt)

        max_nodes = max(1, (total_ch + total_sec + 1) + (gt_total_ch + gt_total_sec + 1))
        norm_similarity = round(max(0.0, 1.0 - (float(zss_ted) / float(max_nodes))), 4)

        # Chapter boundary matching
        matched_chapters = 0
        for p_ch in predicted_chapters:
            p_start = p_ch.get("page_start", p_ch.get("start_page", 1))
            for g_ch in gt_chapters:
                if abs(p_start - g_ch.page_start) <= 1:
                    matched_chapters += 1
                    break

        ch_prec = round(matched_chapters / max(1, total_ch), 4) if total_ch else 0.0
        ch_rec = round(matched_chapters / max(1, gt_total_ch), 4) if gt_total_ch else 0.0
        ch_f1 = round(2 * ch_prec * ch_rec / (ch_prec + ch_rec), 4) if (ch_prec + ch_rec) > 0 else 0.0

        res.update({
            "status": "completed_mode_2_ground_truth",
            "ground_truth_available": True,
            "tree_edit_distance_zss": zss_ted,
            "tree_edit_distance_apted": apted_ted if apted_ted >= 0 else zss_ted,
            "normalized_tree_similarity": norm_similarity,
            "chapter_boundary_precision": ch_prec,
            "chapter_boundary_recall": ch_rec,
            "chapter_boundary_f1": ch_f1,
            "ground_truth_chapters": gt_total_ch,
            "ground_truth_sections": gt_total_sec,
        })
        return res
