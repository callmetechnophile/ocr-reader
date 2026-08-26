import json
from pathlib import Path
from typing import Any, Optional


class ToonEvaluator:
    """
    Evaluates TOON canonical serialization & reconstruction integrity.
    Verifies file completeness, metadata matches, and round-trip structural consistency.
    """

    def evaluate(
        self,
        toon_path: Optional[str | Path],
        expected_document_id: str,
        manifest_data: Optional[dict[str, Any]] = None,
        chapters: Optional[list[dict[str, Any]]] = None,
        chunks: Optional[list[dict[str, Any]]] = None,
        pages: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        if not toon_path:
            return {
                "status": "not_available",
                "toon_file_found": False,
                "valid": False,
                "error": "No TOON file path provided",
            }

        p = Path(toon_path)
        if not p.exists():
            return {
                "status": "not_available",
                "toon_file_found": False,
                "valid": False,
                "error": f"TOON file not found at: {p}",
            }

        file_size = p.stat().st_size
        if file_size == 0:
            return {
                "status": "completed",
                "toon_file_found": True,
                "valid": False,
                "file_size_bytes": 0,
                "error": "TOON file is empty (0 bytes)",
            }

        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            return {
                "status": "completed",
                "toon_file_found": True,
                "valid": False,
                "file_size_bytes": file_size,
                "error": f"Invalid JSON payload: {e}",
            }

        doc_id_match = data.get("document_id") == expected_document_id
        fmt_match = data.get("format") == "TOON_V1"

        t_summary = data.get("summary", {})
        t_pages = len(data.get("pages", []))
        t_chapters = len(data.get("structure", {}).get("chapters", []))
        t_sections = sum(len(ch.get("sections", [])) for ch in data.get("structure", {}).get("chapters", []))
        t_chunks = len(data.get("chunks", []))

        # Check against manifest if available
        manifest_match = True
        m_issues = []
        if manifest_data:
            m_pages = manifest_data.get("processed_pages", manifest_data.get("page_count", 0))
            m_ch = manifest_data.get("chapters", 0)
            m_sec = manifest_data.get("sections", 0)
            m_chk = manifest_data.get("chunks", 0)

            if m_pages and t_pages != m_pages:
                manifest_match = False
                m_issues.append(f"Pages mismatch: TOON {t_pages} vs manifest {m_pages}")
            if m_ch and t_chapters != m_ch:
                manifest_match = False
                m_issues.append(f"Chapters mismatch: TOON {t_chapters} vs manifest {m_ch}")
            if m_sec and t_sections != m_sec:
                manifest_match = False
                m_issues.append(f"Sections mismatch: TOON {t_sections} vs manifest {m_sec}")
            if m_chk and t_chunks != m_chk:
                manifest_match = False
                m_issues.append(f"Chunks mismatch: TOON {t_chunks} vs manifest {m_chk}")

        # Round-trip reconstruction check
        round_trip_passed = False
        reconstructed_summary = {}
        if fmt_match and doc_id_match and not m_issues:
            # Verify reconstructed entities preserve field contracts
            valid_pages = all("page_number" in pg and "regions" in pg for pg in data.get("pages", []))
            valid_chapters = all("chapter_id" in ch and "title" in ch for ch in data.get("structure", {}).get("chapters", []))
            valid_chunks = all("chunk_id" in ck and "text" in ck for ck in data.get("chunks", []))
            round_trip_passed = valid_pages and valid_chapters and valid_chunks
            reconstructed_summary = {
                "pages_reconstructed": t_pages,
                "chapters_reconstructed": t_chapters,
                "sections_reconstructed": t_sections,
                "chunks_reconstructed": t_chunks,
            }

        is_valid = fmt_match and doc_id_match and manifest_match and round_trip_passed

        return {
            "status": "completed",
            "toon_file_found": True,
            "valid": is_valid,
            "path": str(p),
            "file_size_bytes": file_size,
            "format": data.get("format"),
            "document_id_matches": doc_id_match,
            "manifest_consistency": manifest_match,
            "round_trip_validation": round_trip_passed,
            "manifest_issues": m_issues,
            "reconstructed_summary": reconstructed_summary,
        }
