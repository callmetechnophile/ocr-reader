from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any, Optional
from app.schemas.document import ChapterInfo, DocumentManifest
from app.schemas.page import PageSchema
from app.schemas.region import Region, RegionType


@dataclass
class StructuralValidationReport:
    document_id: str
    metrics: dict[str, float]
    heading_hierarchy: dict[str, Any]
    chapter_numbering: dict[str, Any]
    section_numbering: dict[str, Any]
    parent_child_integrity: dict[str, Any]
    page_boundary_integrity: dict[str, Any]
    reading_order: dict[str, Any]
    duplicate_headings: dict[str, Any]
    empty_chapters: dict[str, Any]
    cross_boundary_sections: list[dict[str, Any]]
    chunk_alignment: dict[str, Any]
    detected_issues: list[dict[str, Any]]
    recommendations: list[str]


def _section_to_dict(s: Any) -> dict[str, Any]:
    if hasattr(s, "model_dump"):
        return s.model_dump()
    if isinstance(s, dict):
        return dict(s)
    return {
        "section_id": getattr(s, "section_id", "sec_000"),
        "title": getattr(s, "title", ""),
        "page_start": getattr(s, "page_start", 1),
        "page_end": getattr(s, "page_end", 1),
    }


class DocumentStructuralValidator:
    """
    Document-agnostic structural validation engine.
    Evaluates consistency, hierarchy, page boundaries, parent-child relationships,
    and reading order without making assumptions about absolute chapter or section counts.
    """

    def validate(
        self,
        document_id: str,
        pages: list[PageSchema],
        chapters: list[ChapterInfo],
        chunks: list[dict[str, Any]],
        manifest: Optional[DocumentManifest] = None,
    ) -> StructuralValidationReport:
        # Build lookup maps
        page_map = {p.page_number: p for p in pages}
        chapter_map = {ch.chapter_id: ch for ch in chapters}
        section_list: list[dict[str, Any]] = []
        for ch in chapters:
            for s in ch.sections:
                s_copy = _section_to_dict(s)
                s_copy["parent_chapter_id"] = ch.chapter_id
                s_copy["parent_chapter_number"] = ch.number
                section_list.append(s_copy)

        issues: list[dict[str, Any]] = []

        # 1. Heading Hierarchy Consistency
        hierarchy_res = self._validate_heading_hierarchy(pages, chapters)
        issues.extend(hierarchy_res.get("issues", []))

        # 2. Chapter Numbering Consistency
        ch_num_res = self._validate_chapter_numbering(chapters)
        issues.extend(ch_num_res.get("issues", []))

        # 3. Section Numbering Consistency
        sec_num_res = self._validate_section_numbering(chapters)
        issues.extend(sec_num_res.get("issues", []))

        # 4. Parent-Child Relationships
        pc_res = self._validate_parent_child_relationships(chapters, chunks)
        issues.extend(pc_res.get("issues", []))

        # 5. Page Boundary Consistency
        boundary_res = self._validate_page_boundaries(pages, chapters, chunks)
        issues.extend(boundary_res.get("issues", []))

        # 6. Reading Order
        ro_res = self._validate_reading_order(pages)
        issues.extend(ro_res.get("issues", []))

        # 7. Duplicate Headings Detection
        dup_res = self._validate_duplicate_headings(pages)
        issues.extend(dup_res.get("issues", []))

        # 8. Empty / Near-Empty Chapters
        empty_res = self._validate_empty_chapters(chapters, chunks, pages)
        issues.extend(empty_res.get("issues", []))

        # 9. Sections Crossing Chapter Boundaries
        cross_res = self._validate_cross_boundary_sections(chapters, pages)
        issues.extend([{"category": "cross_boundary_section", **c} for c in cross_res if c["status"] == "INVALID"])

        # 10. Chunk-to-Section Alignment
        chunk_res = self._validate_chunk_alignment(chunks, chapter_map, section_list, page_map)
        issues.extend(chunk_res.get("issues", []))

        # Compute category metrics [0.0 - 1.0] where 1.0 is highest integrity
        metrics = {
            "heading_hierarchy": hierarchy_res.get("score", 1.0),
            "chapter_numbering": ch_num_res.get("score", 1.0),
            "section_numbering": sec_num_res.get("score", 1.0),
            "parent_child_integrity": pc_res.get("score", 1.0),
            "page_boundary_integrity": boundary_res.get("score", 1.0),
            "reading_order": ro_res.get("score", 1.0),
            "duplicate_heading_rate": dup_res.get("duplicate_rate", 0.0),
            "empty_chapter_rate": empty_res.get("empty_rate", 0.0),
            "cross_boundary_section_rate": len([c for c in cross_res if c["status"] == "INVALID"]) / max(1, len(section_list)),
            "chunk_alignment": chunk_res.get("score", 1.0),
        }

        # Recommendations based on issues
        recommendations = self._generate_recommendations(issues, metrics)

        return StructuralValidationReport(
            document_id=document_id,
            metrics=metrics,
            heading_hierarchy=hierarchy_res,
            chapter_numbering=ch_num_res,
            section_numbering=sec_num_res,
            parent_child_integrity=pc_res,
            page_boundary_integrity=boundary_res,
            reading_order=ro_res,
            duplicate_headings=dup_res,
            empty_chapters=empty_res,
            cross_boundary_sections=cross_res,
            chunk_alignment=chunk_res,
            detected_issues=issues,
            recommendations=recommendations,
        )

    def _validate_heading_hierarchy(self, pages: list[PageSchema], chapters: list[ChapterInfo]) -> dict[str, Any]:
        issues = []
        headings_count = 0
        subheadings_count = 0
        for p in pages:
            for r in p.regions:
                if r.type == RegionType.HEADING:
                    headings_count += 1
                elif r.type == RegionType.SUBHEADING:
                    subheadings_count += 1

        score = 1.0
        if len(chapters) > 0 and headings_count == 0:
            issues.append({
                "severity": "WARNING",
                "category": "heading_hierarchy",
                "message": "Chapters detected but zero HEADING regions classified on pages.",
            })
            score = 0.85

        return {
            "score": score,
            "heading_count": headings_count,
            "subheading_count": subheadings_count,
            "chapters_detected": len(chapters),
            "issues": issues,
        }

    def _validate_chapter_numbering(self, chapters: list[ChapterInfo]) -> dict[str, Any]:
        issues = []
        if not chapters:
            return {"scheme": "not_detected", "score": 1.0, "issues": []}

        numbers = [ch.number for ch in chapters if ch.number]
        if not numbers:
            return {"scheme": "not_detected", "score": 1.0, "issues": []}

        # Check for numeric or roman numbering
        numeric_vals = []
        for n in numbers:
            try:
                numeric_vals.append(int(n))
            except ValueError:
                pass

        duplicates = set([x for x in numbers if numbers.count(x) > 1])
        if duplicates:
            issues.append({
                "severity": "WARNING",
                "category": "chapter_numbering",
                "message": f"Duplicate chapter numbers found: {list(duplicates)}",
            })

        score = 1.0
        if duplicates:
            score -= 0.15

        if len(numeric_vals) >= 3:
            # Check for large gaps
            sorted_nums = sorted(numeric_vals)
            gaps = []
            for i in range(len(sorted_nums) - 1):
                diff = sorted_nums[i + 1] - sorted_nums[i]
                if diff > 2:
                    gaps.append((sorted_nums[i], sorted_nums[i + 1]))
            if gaps:
                issues.append({
                    "severity": "INFO",
                    "category": "chapter_numbering",
                    "message": f"Chapter numbering has gaps: {gaps}",
                })

        return {
            "scheme": "numbered" if numeric_vals else "custom",
            "score": max(0.0, score),
            "duplicates": list(duplicates),
            "total_numbered_chapters": len(numbers),
            "issues": issues,
        }

    def _validate_section_numbering(self, chapters: list[ChapterInfo]) -> dict[str, Any]:
        issues = []
        all_sections = [_section_to_dict(s) for ch in chapters for s in ch.sections]
        if not all_sections:
            return {"scheme": "not_detected", "score": 1.0, "issues": []}

        sec_ids = [s.get("section_id") for s in all_sections if s.get("section_id")]
        duplicates = set([x for x in sec_ids if sec_ids.count(x) > 1])

        unrelated_sections = []
        for ch in chapters:
            if not ch.number:
                continue
            for raw_s in ch.sections:
                s = _section_to_dict(raw_s)
                title = s.get("title", "")
                # If section title starts with numeric like "4.2" and chapter is "1"
                m = re.match(r"^(\d+)\.\d+", title.strip())
                if m:
                    sec_prefix = m.group(1)
                    if str(ch.number) != sec_prefix and len(chapters) > 1:
                        unrelated_sections.append({
                            "section": title,
                            "chapter": ch.title,
                            "chapter_num": ch.number,
                            "section_prefix": sec_prefix,
                        })

        score = 1.0
        if duplicates:
            issues.append({
                "severity": "WARNING",
                "category": "section_numbering",
                "message": f"Duplicate section IDs found: {list(duplicates)}",
            })
            score -= 0.15

        if unrelated_sections:
            issues.append({
                "severity": "INFO",
                "category": "section_numbering",
                "message": f"{len(unrelated_sections)} section numbers do not match parent chapter prefix",
                "examples": unrelated_sections[:3],
            })
            score -= 0.10

        return {
            "total_sections": len(all_sections),
            "duplicates": list(duplicates),
            "unrelated_prefix_count": len(unrelated_sections),
            "score": max(0.0, score),
            "issues": issues,
        }

    def _validate_parent_child_relationships(
        self, chapters: list[ChapterInfo], chunks: list[dict[str, Any]]
    ) -> dict[str, Any]:
        issues = []
        orphan_chunks = 0
        invalid_parent_refs = 0
        chapter_ids = {ch.chapter_id for ch in chapters}

        for chunk in chunks:
            ch_id = chunk.get("chapter_id")
            if not ch_id:
                orphan_chunks += 1
            elif ch_id not in chapter_ids:
                invalid_parent_refs += 1

        score = 1.0
        if orphan_chunks > 0:
            issues.append({
                "severity": "WARNING",
                "category": "parent_child_integrity",
                "message": f"Found {orphan_chunks} chunks without a parent chapter ID.",
            })
            score -= min(0.3, orphan_chunks / max(1, len(chunks)))

        if invalid_parent_refs > 0:
            issues.append({
                "severity": "ERROR",
                "category": "parent_child_integrity",
                "message": f"Found {invalid_parent_refs} chunks referencing non-existent chapter IDs.",
            })
            score -= min(0.4, invalid_parent_refs / max(1, len(chunks)))

        return {
            "score": max(0.0, round(score, 3)),
            "orphan_chunks": orphan_chunks,
            "invalid_parent_refs": invalid_parent_refs,
            "issues": issues,
        }

    def _validate_page_boundaries(
        self, pages: list[PageSchema], chapters: list[ChapterInfo], chunks: list[dict[str, Any]]
    ) -> dict[str, Any]:
        issues = []
        max_page = max([p.page_number for p in pages]) if pages else 0
        boundary_violations = 0

        for ch in chapters:
            ch_start = getattr(ch, "page_start", getattr(ch, "start_page", 1))
            ch_end = getattr(ch, "page_end", getattr(ch, "end_page", ch_start))
            if ch_start < 1 or ch_end > max_page or ch_start > ch_end:
                boundary_violations += 1
                issues.append({
                    "severity": "ERROR",
                    "category": "page_boundary",
                    "message": f"Invalid chapter page range [{ch_start}, {ch_end}] for '{ch.title}'",
                })

        for chunk in chunks:
            p_start = chunk.get("page_start", 1)
            p_end = chunk.get("page_end", p_start)
            if p_start < 1 or p_end > max_page or p_start > p_end:
                boundary_violations += 1

        score = 1.0 - min(1.0, boundary_violations / max(1, len(chapters) + len(chunks)))
        return {
            "score": round(score, 3),
            "boundary_violations": boundary_violations,
            "max_page": max_page,
            "issues": issues,
        }

    def _validate_reading_order(self, pages: list[PageSchema]) -> dict[str, Any]:
        issues = []
        duplicate_indices = 0
        non_monotonic_pages = 0

        for p in pages:
            orders = [r.reading_order for r in p.regions]
            if len(orders) != len(set(orders)):
                duplicate_indices += 1

            if orders != sorted(orders):
                non_monotonic_pages += 1

        score = 1.0
        if duplicate_indices > 0:
            score -= 0.1
            issues.append({
                "severity": "INFO",
                "category": "reading_order",
                "message": f"Duplicate reading order indices found on {duplicate_indices} pages.",
            })

        return {
            "score": round(max(0.0, score), 3),
            "duplicate_order_pages": duplicate_indices,
            "non_monotonic_pages": non_monotonic_pages,
            "issues": issues,
        }

    def _validate_duplicate_headings(self, pages: list[PageSchema]) -> dict[str, Any]:
        issues = []
        headings: dict[str, list[int]] = {}
        for p in pages:
            for r in p.regions:
                if r.type in (RegionType.HEADING, RegionType.SUBHEADING):
                    t = r.text.strip().lower()
                    if len(t) > 5:
                        headings.setdefault(t, []).append(p.page_number)

        legitimate = 0
        likely_errors = 0
        for text, p_nums in headings.items():
            if len(p_nums) > 1:
                # If repeated on adjacent pages or > 5 times (likely running header classified as heading)
                if len(p_nums) > 5:
                    likely_errors += 1
                else:
                    legitimate += 1

        rate = likely_errors / max(1, len(headings))
        return {
            "duplicate_rate": round(rate, 4),
            "legitimate_duplicates": legitimate,
            "likely_duplicate_errors": likely_errors,
            "issues": issues,
        }

    def _validate_empty_chapters(
        self, chapters: list[ChapterInfo], chunks: list[dict[str, Any]], pages: list[PageSchema]
    ) -> dict[str, Any]:
        issues = []
        classification = {}
        empty_count = 0

        for ch in chapters:
            ch_chunks = [c for c in chunks if c.get("chapter_id") == ch.chapter_id]
            if len(ch_chunks) == 0:
                classification[ch.chapter_id] = "EMPTY"
                empty_count += 1
            elif len(ch_chunks) == 1 and ch_chunks[0].get("token_count", 0) < 50:
                classification[ch.chapter_id] = "LOW_CONTENT"
            else:
                classification[ch.chapter_id] = "NORMAL"

        rate = empty_count / max(1, len(chapters)) if chapters else 0.0
        return {
            "empty_rate": round(rate, 4),
            "empty_chapter_count": empty_count,
            "classifications": classification,
            "issues": issues,
        }

    def _validate_cross_boundary_sections(
        self, chapters: list[ChapterInfo], pages: list[PageSchema]
    ) -> list[dict[str, Any]]:
        results = []
        for ch in chapters:
            ch_start = getattr(ch, "page_start", getattr(ch, "start_page", 1))
            ch_end = getattr(ch, "page_end", getattr(ch, "end_page", ch_start))
            for raw_s in ch.sections:
                s = _section_to_dict(raw_s)
                p_start = s.get("page_start", ch_start)
                p_end = s.get("page_end", ch_end)

                status = "VALID"
                if p_start < ch_start or p_end > ch_end:
                    # If slight mismatch across single page transition, questionable, else invalid
                    if p_end > ch_end + 1 or p_start < ch_start - 1:
                        status = "INVALID"
                    else:
                        status = "QUESTIONABLE"

                results.append({
                    "section_id": s.get("section_id"),
                    "section_title": s.get("title"),
                    "parent_chapter": ch.title,
                    "section_page_start": p_start,
                    "section_page_end": p_end,
                    "chapter_page_start": ch_start,
                    "chapter_page_end": ch_end,
                    "status": status,
                })
        return results

    def _validate_chunk_alignment(
        self,
        chunks: list[dict[str, Any]],
        chapter_map: dict[str, ChapterInfo],
        section_list: list[dict[str, Any]],
        page_map: dict[int, PageSchema],
    ) -> dict[str, Any]:
        issues = []
        misaligned_chunks = 0

        for chunk in chunks:
            ch_id = chunk.get("chapter_id")
            p_start = chunk.get("page_start", 1)
            p_end = chunk.get("page_end", p_start)

            if ch_id and ch_id in chapter_map:
                ch = chapter_map[ch_id]
                ch_start = getattr(ch, "page_start", getattr(ch, "start_page", 1))
                ch_end = getattr(ch, "page_end", getattr(ch, "end_page", ch_start))
                if p_start < ch_start or p_end > ch_end:
                    misaligned_chunks += 1

        score = 1.0 - (misaligned_chunks / max(1, len(chunks)))
        return {
            "score": round(max(0.0, score), 3),
            "total_chunks": len(chunks),
            "misaligned_chunks": misaligned_chunks,
            "issues": issues,
        }

    def _generate_recommendations(self, issues: list[dict[str, Any]], metrics: dict[str, float]) -> list[str]:
        recs = []
        if metrics.get("parent_child_integrity", 1.0) < 0.9:
            recs.append("Review orphan chunks or sections that lack valid chapter association.")
        if metrics.get("cross_boundary_section_rate", 0.0) > 0.05:
            recs.append("Check chapter boundary heuristics as multiple sections extend past their parent chapter.")
        if metrics.get("empty_chapter_rate", 0.0) > 0.2:
            recs.append("Multiple empty chapters detected; verify if title/preface pages were classified as chapters.")
        if not recs:
            recs.append("Document structure is internally consistent and verified.")
        return recs

    def write_audit_reports(self, audit_dir: str | Path, report: StructuralValidationReport) -> None:
        """Write all JSON and Markdown audit artifacts to the audit directory."""
        a_dir = Path(audit_dir)
        a_dir.mkdir(parents=True, exist_ok=True)

        # 1. structural_validation.json
        val_data = {
            "document_id": report.document_id,
            "metrics": report.metrics,
            "heading_hierarchy": report.heading_hierarchy,
            "chapter_numbering": report.chapter_numbering,
            "section_numbering": report.section_numbering,
            "parent_child_integrity": report.parent_child_integrity,
            "page_boundary_integrity": report.page_boundary_integrity,
            "reading_order": report.reading_order,
            "duplicate_headings": report.duplicate_headings,
            "empty_chapters": report.empty_chapters,
            "cross_boundary_sections": report.cross_boundary_sections,
            "chunk_alignment": report.chunk_alignment,
            "detected_issues": report.detected_issues,
            "recommendations": report.recommendations,
        }
        with open(a_dir / "structural_validation.json", "w", encoding="utf-8") as f:
            json.dump(val_data, f, indent=2)

        # 2. chapter_audit.json
        with open(a_dir / "chapter_audit.json", "w", encoding="utf-8") as f:
            json.dump({
                "numbering": report.chapter_numbering,
                "empty_chapters": report.empty_chapters,
            }, f, indent=2)

        # 3. section_audit.json
        with open(a_dir / "section_audit.json", "w", encoding="utf-8") as f:
            json.dump({
                "numbering": report.section_numbering,
                "cross_boundary_sections": report.cross_boundary_sections,
            }, f, indent=2)

        # 4. chunk_audit.json
        with open(a_dir / "chunk_audit.json", "w", encoding="utf-8") as f:
            json.dump(report.chunk_alignment, f, indent=2)

        # 5. reading_order_audit.json
        with open(a_dir / "reading_order_audit.json", "w", encoding="utf-8") as f:
            json.dump(report.reading_order, f, indent=2)

        # 6. structural_validation.md
        md_lines = [
            f"# Structural Validation Report — {report.document_id}",
            "",
            "## Structural Summary",
            f"- **Parent-Child Integrity**: {report.metrics['parent_child_integrity']:.2f}",
            f"- **Page Boundary Integrity**: {report.metrics['page_boundary_integrity']:.2f}",
            f"- **Reading Order Score**: {report.metrics['reading_order']:.2f}",
            f"- **Chunk Alignment Score**: {report.metrics['chunk_alignment']:.2f}",
            "",
            "## Heading Hierarchy",
            f"- Headings: {report.heading_hierarchy.get('heading_count', 0)}",
            f"- Subheadings: {report.heading_hierarchy.get('subheading_count', 0)}",
            "",
            "## Chapter Numbering",
            f"- Scheme: `{report.chapter_numbering.get('scheme', 'not_detected')}`",
            f"- Numbered Chapters: {report.chapter_numbering.get('total_numbered_chapters', 0)}",
            "",
            "## Section Numbering",
            f"- Total Sections: {report.section_numbering.get('total_sections', 0)}",
            f"- Duplicate Section IDs: {len(report.section_numbering.get('duplicates', []))}",
            "",
            "## Parent-Child Relationships",
            f"- Orphan Chunks: {report.parent_child_integrity.get('orphan_chunks', 0)}",
            f"- Invalid Parent References: {report.parent_child_integrity.get('invalid_parent_refs', 0)}",
            "",
            "## Page Boundaries",
            f"- Total Boundary Violations: {report.page_boundary_integrity.get('boundary_violations', 0)}",
            "",
            "## Reading Order",
            f"- Non-monotonic Pages: {report.reading_order.get('non_monotonic_pages', 0)}",
            "",
            "## Duplicate Headings",
            f"- Likely Duplicate Errors: {report.duplicate_headings.get('likely_duplicate_errors', 0)}",
            "",
            "## Empty / Near-Empty Chapters",
            f"- Empty Chapters: {report.empty_chapters.get('empty_chapter_count', 0)}",
            "",
            "## Cross-Boundary Sections",
            f"- Total Checked: {len(report.cross_boundary_sections)}",
            "",
            "## Chunk Alignment",
            f"- Total Chunks: {report.chunk_alignment.get('total_chunks', 0)}",
            f"- Misaligned Chunks: {report.chunk_alignment.get('misaligned_chunks', 0)}",
            "",
            "## Detected Issues",
        ]
        if report.detected_issues:
            for iss in report.detected_issues:
                md_lines.append(f"- **[{iss.get('severity', 'INFO')}]** {iss.get('message', '')}")
        else:
            md_lines.append("- No structural issues detected.")

        md_lines.extend(["", "## Recommendations"])
        for rec in report.recommendations:
            md_lines.append(f"- {rec}")
        md_lines.append("")

        with open(a_dir / "structural_validation.md", "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))
