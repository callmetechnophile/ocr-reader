import re
from typing import Sequence
from app.schemas.document import Chapter, Section
from app.schemas.page import PageSchema
from app.schemas.region import RegionType


class SectionDetector:
    """
    Detects hierarchical sections (e.g. 1.1, 1.2, 2.3.1) and subheadings within chapters.
    """

    SECTION_NUMBERED_PATTERN = re.compile(
        r"^(\d+\s*\.\s*\d+(?:\s*\.\s*\d+)?)\s*(.*)$"
    )

    def detect_sections(
        self,
        chapters: Sequence[Chapter],
        pages: Sequence[PageSchema],
    ) -> list[Chapter]:
        page_dict = {p.page_number: p for p in pages}
        updated_chapters: list[Chapter] = []

        for ch in chapters:
            section_candidates: list[dict] = []
            for p_num in range(ch.page_start, ch.page_end + 1):
                page = page_dict.get(p_num)
                if not page:
                    continue

                for region in page.regions:
                    text = region.text.strip()
                    if not text:
                        continue

                    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
                    for line in lines:
                        if len(line) > 120:
                            continue

                        match = self.SECTION_NUMBERED_PATTERN.match(line)
                        if match:
                            raw_num = match.group(1)
                            sec_num_str = re.sub(r"\s+", "", raw_num)
                            sec_title = match.group(2).strip()

                            ch_prefix = f"{ch.number}."
                            if sec_num_str.startswith(ch_prefix) or len(chapters) == 1:
                                section_candidates.append({
                                    "sec_num": sec_num_str,
                                    "title": line,
                                    "page_number": p_num,
                                })
                        elif region.type == RegionType.SUBHEADING and len(line.split()) < 10:
                            section_candidates.append({
                                "sec_num": f"{ch.number}.{len(section_candidates)+1}",
                                "title": line,
                                "page_number": p_num,
                            })

            # Deduplicate by title/page
            deduped: list[dict] = []
            seen = set()
            for cand in section_candidates:
                key = (cand["title"], cand["page_number"])
                if key not in seen:
                    deduped.append(cand)
                    seen.add(key)

            # Build Section objects
            sections: list[Section] = []
            for s_idx, cand in enumerate(deduped):
                sec_id = f"{ch.chapter_id}_s{s_idx+1:03d}"
                s_start = cand["page_number"]
                if s_idx + 1 < len(deduped):
                    s_end = max(s_start, deduped[s_idx + 1]["page_number"] - 1)
                else:
                    s_end = ch.page_end

                sections.append(
                    Section(
                        section_id=sec_id,
                        title=cand["title"],
                        page_start=s_start,
                        page_end=s_end,
                    )
                )

            ch_dict = ch.model_dump()
            ch_dict["sections"] = sections
            updated_chapters.append(Chapter(**ch_dict))

        return updated_chapters
