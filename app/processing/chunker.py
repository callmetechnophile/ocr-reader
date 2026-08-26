from typing import Sequence
from app.schemas.document import Chapter, Chunk
from app.schemas.page import PageSchema
from app.schemas.region import RegionType


class DocumentChunker:
    """
    Structure-aware document chunker.
    Organizes text regions by Chapter and Section, grouping paragraph blocks
    into token-bounded chunks (300-800 tokens) with configurable token overlap.
    """

    def __init__(
        self,
        min_tokens: int = 300,
        max_tokens: int = 800,
        overlap_tokens: int = 50,
    ):
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def chunk_document(
        self,
        chapters: Sequence[Chapter],
        pages: Sequence[PageSchema],
    ) -> dict[str, list[Chunk]]:
        """
        Produce structured chunks organized by chapter ID.

        Returns:
            Dict mapping chapter_id -> list of Chunk objects.
        """
        page_dict = {p.page_number: p for p in pages}
        chapter_chunks: dict[str, list[Chunk]] = {}

        for ch in chapters:
            chunks: list[Chunk] = []
            chunk_idx = 1

            # 1. Collect all body/table/caption text units within this chapter
            units: list[dict] = []
            for p_num in range(ch.page_start, ch.page_end + 1):
                page = page_dict.get(p_num)
                if not page:
                    continue

                # Find current section if any
                active_sec_id = None
                for sec in ch.sections:
                    if sec.page_start <= p_num <= sec.page_end:
                        active_sec_id = sec.section_id
                        break

                for r in page.regions:
                    # Exclude running headers and footers/page numbers from chunk bodies
                    if r.type in (RegionType.HEADER, RegionType.FOOTER, RegionType.PAGE_NUMBER):
                        continue
                    text = r.text.strip()
                    if not text:
                        continue

                    token_count = len(text.split())
                    units.append({
                        "region_id": r.region_id,
                        "page_number": p_num,
                        "section_id": active_sec_id,
                        "text": text,
                        "token_count": token_count,
                    })

            if not units:
                chapter_chunks[ch.chapter_id] = []
                continue

            # 2. Group units into token-bounded chunks
            current_texts: list[str] = []
            current_regions: list[str] = []
            current_pages: list[int] = []
            current_tokens = 0
            current_sec_id = units[0]["section_id"]

            for unit in units:
                unit_tokens = unit["token_count"]

                # If adding unit exceeds max_tokens or section changed and we have >= min_tokens
                should_split = False
                if current_tokens + unit_tokens > self.max_tokens and current_tokens >= self.min_tokens:
                    should_split = True
                elif unit["section_id"] != current_sec_id and current_tokens >= self.min_tokens:
                    should_split = True

                if should_split:
                    # Emit chunk
                    chunk_id = f"{ch.chapter_id}_chk_{chunk_idx:04d}"
                    joined_text = "\n\n".join(current_texts)
                    chunks.append(
                        Chunk(
                            chunk_id=chunk_id,
                            chapter_id=ch.chapter_id,
                            section_id=current_sec_id,
                            page_start=min(current_pages),
                            page_end=max(current_pages),
                            source_regions=list(current_regions),
                            token_count=current_tokens,
                            text=joined_text,
                        )
                    )
                    chunk_idx += 1

                    # Handle overlap
                    if self.overlap_tokens > 0 and len(current_texts) > 1:
                        # Keep the last unit for overlap
                        last_unit = current_texts[-1]
                        last_region = current_regions[-1]
                        last_page = current_pages[-1]
                        current_texts = [last_unit]
                        current_regions = [last_region]
                        current_pages = [last_page]
                        current_tokens = len(last_unit.split())
                    else:
                        current_texts = []
                        current_regions = []
                        current_pages = []
                        current_tokens = 0

                current_texts.append(unit["text"])
                current_regions.append(unit["region_id"])
                current_pages.append(unit["page_number"])
                current_tokens += unit_tokens
                current_sec_id = unit["section_id"]

            # Emit final trailing chunk if non-empty
            if current_texts:
                chunk_id = f"{ch.chapter_id}_chk_{chunk_idx:04d}"
                joined_text = "\n\n".join(current_texts)
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        chapter_id=ch.chapter_id,
                        section_id=current_sec_id,
                        page_start=min(current_pages),
                        page_end=max(current_pages),
                        source_regions=list(current_regions),
                        token_count=current_tokens,
                        text=joined_text,
                    )
                )

            chapter_chunks[ch.chapter_id] = chunks

        return chapter_chunks
