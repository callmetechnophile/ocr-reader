from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Optional
from app.schemas.document import ChapterInfo, DocumentManifest, DocumentMetadata
from app.schemas.page import PageSchema


class ToonWriter:
    """
    Serializes the complete parsed textbook representation into a single canonical .toon file
    for efficient downstream LLM/Transformer Tutor and Retrieval ingestion.
    Preserves full hierarchical structure, provenance, confidence, and tokens without binary bloat.
    """

    def write(
        self,
        document_id: str,
        filename: str,
        manifest: DocumentManifest,
        pages: list[PageSchema],
        chapters: list[ChapterInfo],
        chunks: list[dict[str, Any]],
        output_path: str | Path,
        pipeline_version: str = "0.1.0",
    ) -> Path:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        # Count total sections across chapters
        total_sections = sum(len(ch.sections) for ch in chapters)

        # Build clean JSON serializable representation of pages and regions
        serialized_pages = []
        for p in pages:
            p_dict = p.model_dump()
            serialized_pages.append(p_dict)

        # Build clean chapters
        serialized_chapters = []
        for ch in chapters:
            raw_sections = [
                s.model_dump() if hasattr(s, "model_dump") else s
                for s in ch.sections
            ]
            ch_dict = {
                "chapter_id": ch.chapter_id,
                "number": ch.number,
                "title": ch.title,
                "page_start": getattr(ch, "page_start", getattr(ch, "start_page", 1)),
                "page_end": getattr(ch, "page_end", getattr(ch, "end_page", 1)),
                "sections": raw_sections,
            }
            serialized_chapters.append(ch_dict)

        meta_dict = {}
        if manifest.metadata:
            meta_dict = (
                manifest.metadata.model_dump()
                if hasattr(manifest.metadata, "model_dump")
                else dict(manifest.metadata)
            )

        toon_payload = {
            "format": "TOON_V1",
            "document_id": document_id,
            "filename": filename,
            "pipeline_version": pipeline_version,
            "metadata": meta_dict,
            "summary": {
                "page_count": manifest.page_count,
                "processed_pages": manifest.processed_pages,
                "failed_pages": manifest.failed_pages,
                "chapters_count": len(chapters),
                "sections_count": total_sections,
                "chunks_count": len(chunks),
            },
            "structure": {
                "chapters": serialized_chapters,
            },
            "chunks": chunks,
            "pages": serialized_pages,
        }

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(toon_payload, f, indent=2, ensure_ascii=False)

        return out_file


class ToonValidator:
    """
    Validates a generated .toon file against document manifest and canonical counts.
    """

    @staticmethod
    def validate(
        toon_path: str | Path,
        expected_document_id: str,
        manifest: DocumentManifest,
        output_validation_path: Optional[str | Path] = None,
    ) -> dict[str, Any]:
        p = Path(toon_path)
        if not p.exists():
            res = {
                "path": str(p),
                "valid": False,
                "error": "File does not exist",
            }
            if output_validation_path:
                with open(output_validation_path, "w", encoding="utf-8") as vf:
                    json.dump(res, vf, indent=2)
            return res

        file_size = p.stat().st_size
        if file_size == 0:
            res = {
                "path": str(p),
                "valid": False,
                "error": "File is empty",
            }
            if output_validation_path:
                with open(output_validation_path, "w", encoding="utf-8") as vf:
                    json.dump(res, vf, indent=2)
            return res

        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)

        doc_id_match = (data.get("document_id") == expected_document_id)
        summary = data.get("summary", {})
        pages_count = summary.get("page_count", len(data.get("pages", [])))
        chapters_count = summary.get("chapters_count", len(data.get("structure", {}).get("chapters", [])))
        sections_count = summary.get("sections_count", 0)
        chunks_count = summary.get("chunks_count", len(data.get("chunks", [])))

        is_valid = (
            doc_id_match
            and pages_count == manifest.page_count
            and chapters_count == manifest.chapters
            and sections_count == manifest.sections
            and chunks_count == manifest.chunks
        )

        val_result = {
            "path": str(p),
            "valid": is_valid,
            "document_id": expected_document_id,
            "pages": pages_count,
            "chapters": chapters_count,
            "sections": sections_count,
            "chunks": chunks_count,
            "file_size_bytes": file_size,
        }

        if output_validation_path:
            with open(output_validation_path, "w", encoding="utf-8") as vf:
                json.dump(val_result, vf, indent=2)

        return val_result
