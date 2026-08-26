import statistics
from typing import Any, Optional
from evaluation.schemas.ground_truth import DocumentGT


class ChunkEvaluator:
    """
    Evaluates chunk generation quality:
    - Distribution, token sizes, missing references, duplicates (Mode 1: System Validation)
    - Boundary precision, recall, F1, and section preservation (Mode 2: GT Evaluation)
    """

    def evaluate(
        self,
        predicted_chunks: list[dict[str, Any]],
        predicted_chapters: list[dict[str, Any]],
        max_page: int = 1,
        ground_truth: Optional[DocumentGT] = None,
    ) -> dict[str, Any]:
        total_chunks = len(predicted_chunks)
        token_counts = [c.get("token_count", 0) for c in predicted_chunks]

        avg_size = round(sum(token_counts) / max(1, total_chunks), 2) if token_counts else 0.0
        median_size = round(statistics.median(token_counts), 2) if token_counts else 0.0
        p95_size = (
            round(statistics.quantiles(token_counts, n=20)[18], 2)
            if len(token_counts) >= 20
            else (max(token_counts) if token_counts else 0.0)
        )

        empty_chunks = sum(1 for t in token_counts if t == 0)
        valid_ch_ids = {ch.get("chapter_id") for ch in predicted_chapters}

        orphan_chunks = 0
        missing_sections = 0
        invalid_page_ranges = 0
        texts_seen = set()
        duplicate_chunks = 0

        for c in predicted_chunks:
            ch_id = c.get("chapter_id")
            if not ch_id or ch_id not in valid_ch_ids:
                orphan_chunks += 1
            if not c.get("section_id"):
                missing_sections += 1

            p_start = c.get("page_start", 1)
            p_end = c.get("page_end", p_start)
            if p_start < 1 or p_end > max_page or p_start > p_end:
                invalid_page_ranges += 1

            t_hash = c.get("text", "").strip()
            if t_hash:
                if t_hash in texts_seen:
                    duplicate_chunks += 1
                texts_seen.add(t_hash)

        integrity_score = round(
            max(0.0, 1.0 - ((orphan_chunks + empty_chunks + invalid_page_ranges) / max(1, total_chunks))),
            4,
        ) if total_chunks else 1.0

        res: dict[str, Any] = {
            "total_chunks": total_chunks,
            "average_token_size": avg_size,
            "median_token_size": median_size,
            "p95_token_size": p95_size,
            "empty_chunks_count": empty_chunks,
            "orphan_chunks_count": orphan_chunks,
            "missing_section_refs": missing_sections,
            "invalid_page_ranges": invalid_page_ranges,
            "duplicate_chunks_count": duplicate_chunks,
            "chunk_integrity_score": integrity_score,
        }

        if ground_truth is None or not ground_truth.chunks:
            res.update({
                "status": "completed_mode_1_system_validation",
                "ground_truth_available": False,
                "boundary_precision": None,
                "boundary_recall": None,
                "boundary_f1": None,
                "section_preservation_score": None,
            })
            return res

        # Mode 2: Ground Truth Chunk Evaluation
        gt_chunks = ground_truth.chunks
        gt_total = len(gt_chunks)

        pred_boundaries = set((c.get("page_start", 1), c.get("page_end", 1)) for c in predicted_chunks)
        gt_boundaries = set((c.page_start, c.page_end) for c in gt_chunks)

        tp = len(pred_boundaries & gt_boundaries)
        fp = len(pred_boundaries - gt_boundaries)
        fn = len(gt_boundaries - pred_boundaries)

        prec = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
        rec = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0
        f1 = round(2 * prec * rec / (prec + rec), 4) if (prec + rec) > 0 else 0.0

        # Section preservation: chunks that don't split sections unexpectedly
        section_preservation = round(1.0 - (orphan_chunks / max(1, total_chunks)), 4) if total_chunks else 1.0

        res.update({
            "status": "completed_mode_2_ground_truth",
            "ground_truth_available": True,
            "ground_truth_chunks": gt_total,
            "boundary_precision": prec,
            "boundary_recall": rec,
            "boundary_f1": f1,
            "section_preservation_score": section_preservation,
        })
        return res
