import re
from typing import Sequence
from app.schemas.document import Chapter
from app.schemas.page import PageSchema
from app.schemas.region import RegionType


class ChapterDetector:
    """
    Detects chapter boundaries and metadata from extracted textbook pages.
    Evaluates heading classifications, font sizes, numbering, and keyword patterns.
    """

    CHAPTER_EXPLICIT_PATTERN = re.compile(
        r"^(?:c\s*h\s*a\s*p\s*t\s*e\s*r|p\s*a\s*r\s*t)\s*(\d+|[ivxlcdm]+)(?:[\.:\s-]+(.*))?$", re.IGNORECASE
    )
    NUMBERED_HEADING_PATTERN = re.compile(
        r"^(\d+)\s+([A-Z][A-Za-z0-9\s,\.:\-_/]+)$"
    )

    def detect_chapters(
        self,
        pages: Sequence[PageSchema],
        document_id: str,
    ) -> list[Chapter]:
        """
        Scan all pages sequentially and identify chapter boundaries.
        """
        if not pages:
            return []

        sorted_pages = sorted(pages, key=lambda p: p.page_number)
        candidates: list[dict] = []

        for page in sorted_pages:
            page_found = False
            # Check regions on page
            for region in page.regions:
                if page_found:
                    break
                text = region.text.strip()
                if not text:
                    continue

                lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
                for line in lines:
                    if len(line) > 120:
                        continue

                    # Rule 1: Explicit "Chapter N" or "Chapter IV"
                    match_exp = self.CHAPTER_EXPLICIT_PATTERN.match(line)
                    if match_exp:
                        raw_num = match_exp.group(1)
                        raw_title = match_exp.group(2) or ""
                        num = self._parse_number(raw_num)
                        title = raw_title.strip() if raw_title.strip() else f"Chapter {num}"

                        conf = 0.96 if region.type in (RegionType.HEADING, RegionType.SUBHEADING) else 0.88
                        candidates.append({
                            "number": num,
                            "title": title,
                            "page_number": page.page_number,
                            "confidence": conf,
                            "method": "pattern_explicit",
                        })
                        page_found = True
                        break

                    # Rule 2: Top-level numbered heading (e.g., "1 Getting Started")
                    if region.type == RegionType.HEADING:
                        match_num = self.NUMBERED_HEADING_PATTERN.match(line)
                        if match_num and not line.lower().startswith("section"):
                            num_str = match_num.group(1)
                            title_str = match_num.group(2).strip()
                            num = int(num_str)
                            if 1 <= num <= 99:
                                candidates.append({
                                    "number": num,
                                    "title": title_str,
                                    "page_number": page.page_number,
                                    "confidence": 0.85,
                                    "method": "pattern_numbered_heading",
                                })
                                page_found = True
                                break

        if not candidates:
            # Fallback: treat entire document as Chapter 1
            min_page = sorted_pages[0].page_number
            max_page = sorted_pages[-1].page_number
            source_p = [p.page_id for p in sorted_pages]
            return [
                Chapter(
                    chapter_id="ch_001",
                    number=1,
                    title="Introduction / Main Content",
                    page_start=min_page,
                    page_end=max_page,
                    sections=[],
                    source_pages=source_p,
                    confidence=0.50,
                    detection_method="fallback_single_chapter",
                )
            ]

        # Deduplicate and sort candidates by page_number
        candidates.sort(key=lambda c: c["page_number"])
        deduped: list[dict] = []
        seen_pages = set()
        for cand in candidates:
            if cand["page_number"] not in seen_pages:
                deduped.append(cand)
                seen_pages.add(cand["page_number"])

        chapters: list[Chapter] = []
        max_page_num = sorted_pages[-1].page_number

        for i, cand in enumerate(deduped):
            ch_num = cand.get("number", i + 1)
            ch_id = f"ch_{i+1:03d}"
            p_start = cand["page_number"]
            # Page end is the page before next chapter or end of document
            if i + 1 < len(deduped):
                p_end = max(p_start, deduped[i + 1]["page_number"] - 1)
            else:
                p_end = max_page_num

            # Collect source page IDs in range
            source_pages = [
                p.page_id for p in sorted_pages if p_start <= p.page_number <= p_end
            ]

            chapters.append(
                Chapter(
                    chapter_id=ch_id,
                    number=ch_num,
                    title=cand["title"],
                    page_start=p_start,
                    page_end=p_end,
                    sections=[],
                    source_pages=source_pages,
                    confidence=cand["confidence"],
                    detection_method=cand["method"],
                )
            )

        return chapters

    def _parse_number(self, val: str) -> int:
        val_clean = val.strip().lower()
        if val_clean.isdigit():
            return int(val_clean)
        # Roman numerals conversion
        roman_map = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}
        total = 0
        prev = 0
        for char in reversed(val_clean):
            curr = roman_map.get(char, 0)
            if curr >= prev:
                total += curr
            else:
                total -= curr
            prev = curr
        return total if total > 0 else 1
