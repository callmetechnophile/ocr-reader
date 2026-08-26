import hashlib
import inspect
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Sequence
import pdfplumber
import pymupdf
from app.core.config import settings
from app.core.logging import log_event, logger
from app.extractors.base import BaseExtractor
from app.extractors.cnn_ocr_extractor import CNNOCRExtractor
from app.extractors.pdfplumber_extractor import PDFPlumberExtractor
from app.pipeline.analyzer import PDFPageAnalyzer
from app.pipeline.debug_visualizer import DebugVisualizer
from app.pipeline.normalizer import LayoutNormalizer
from app.pipeline.renderer import PDFPageRenderer
from app.formats.toon_writer import ToonValidator, ToonWriter
from app.processing.chunker import DocumentChunker
from app.schemas.document import (
    Chapter,
    Chunk,
    DocumentManifest,
    DocumentMetadata,
    DocumentStatus,
    LocalJobState,
    PDFProfile,
    ProcessingReport,
)
from app.schemas.page import PageRoute, PageSchema
from app.storage.object_store import LocalFileSystemStore, ObjectStore
from app.structure.chapter_detector import ChapterDetector
from app.structure.section_detector import SectionDetector
from app.structure.validator import DocumentStructuralValidator


def calculate_file_sha256(file_path: str | Path) -> str:
    """Computes SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


class DocumentPipelineOrchestrator:
    """
    Document-agnostic 11-stage textbook ingestion pipeline orchestrator.
    Processes arbitrary PDF textbooks into normalized page JSON, structured
    chapters, sections, chunks, manifests, and reports.
    """

    def __init__(
        self,
        object_store: Optional[ObjectStore] = None,
        digital_extractor: Optional[BaseExtractor] = None,
        ocr_extractor: Optional[BaseExtractor] = None,
        analyzer: Optional[PDFPageAnalyzer] = None,
        normalizer: Optional[LayoutNormalizer] = None,
        chapter_detector: Optional[ChapterDetector] = None,
        section_detector: Optional[SectionDetector] = None,
        chunker: Optional[DocumentChunker] = None,
        debug_visualizer: Optional[DebugVisualizer] = None,
        renderer: Optional[PDFPageRenderer] = None,
        validator: Optional[DocumentStructuralValidator] = None,
        toon_writer: Optional[ToonWriter] = None,
    ):
        self.storage = object_store or LocalFileSystemStore(base_path="./data")
        self.digital_extractor = digital_extractor or PDFPlumberExtractor()
        self.ocr_extractor = ocr_extractor or CNNOCRExtractor()
        self.analyzer = analyzer or PDFPageAnalyzer()
        self.normalizer = normalizer or LayoutNormalizer()
        self.chapter_detector = chapter_detector or ChapterDetector()
        self.section_detector = section_detector or SectionDetector()
        self.chunker = chunker or DocumentChunker()
        self.debug_visualizer = debug_visualizer or DebugVisualizer()
        self.renderer = renderer or PDFPageRenderer()
        self.validator = validator or DocumentStructuralValidator()
        self.toon_writer = toon_writer or ToonWriter()

    async def process_document(
        self,
        pdf_path: str | Path,
        output_dir: Optional[str | Path] = None,
        document_id: Optional[str] = None,
        start_page: Optional[int] = None,
        end_page: Optional[int] = None,
        debug: bool = False,
        debug_output_dir: Optional[str | Path] = None,
        force: bool = False,
        generate_toon: bool = False,
        toon_output_dir: Optional[str | Path] = None,
        progress_callback: Optional[Callable[..., None]] = None,
    ) -> DocumentManifest:
        p_path = Path(pdf_path).resolve()
        if not p_path.exists():
            raise FileNotFoundError(f"PDF file not found at: {p_path}")

        t0_total = time.perf_counter()
        file_size = p_path.stat().st_size
        file_sha256 = calculate_file_sha256(p_path)

        # 1. Automatic Document ID derived from SHA256 content hash
        doc_id = document_id or f"doc_{file_sha256[:16]}"
        filename = p_path.name

        # Resolve output directory
        base_out_dir = Path(output_dir).resolve() if output_dir else (Path("./data/processed") / doc_id).resolve()
        base_out_dir.mkdir(parents=True, exist_ok=True)
        pages_dir = base_out_dir / "pages"
        pages_dir.mkdir(exist_ok=True)
        chapters_dir = base_out_dir / "chapters"
        chapters_dir.mkdir(exist_ok=True)
        chunks_dir = base_out_dir / "chunks"
        chunks_dir.mkdir(exist_ok=True)
        debug_dir = Path(debug_output_dir).resolve() if debug_output_dir else (base_out_dir / "debug")
        if debug:
            debug_dir.mkdir(parents=True, exist_ok=True)

        log_event(logger, f"Starting document pipeline processing for {doc_id}", document_id=doc_id, stage="pipeline_start")

        # Atomic initial Job State
        self._write_job_state(
            base_out_dir,
            LocalJobState(
                document_id=doc_id,
                filename=filename,
                status=DocumentStatus.PROCESSING,
                progress=0.0,
                pages_total=0,
                pages_processed=0,
            ),
        )

        timings = {
            "pdf_loading_ms": 0.0,
            "profiling_ms": 0.0,
            "page_analysis_ms": 0.0,
            "extraction_ms": 0.0,
            "rendering_ms": 0.0,
            "ocr_ms": 0.0,
            "structure_extraction_ms": 0.0,
            "chunking_ms": 0.0,
            "total_processing_ms": 0.0,
            "average_time_per_page_ms": 0.0,
        }

        # Step 1: Validate & open PDF
        t0 = time.perf_counter()
        try:
            with pymupdf.open(str(p_path)) as doc_mupdf:
                total_doc_pages = len(doc_mupdf)
                pdf_title = doc_mupdf.metadata.get("title") if doc_mupdf.metadata else None
        except Exception as e:
            self._write_job_state(
                base_out_dir,
                LocalJobState(
                    document_id=doc_id,
                    filename=filename,
                    status=DocumentStatus.FAILED,
                    error=f"Corrupted or invalid PDF: {str(e)}",
                ),
            )
            raise ValueError(f"Invalid PDF file: {e}") from e
        timings["pdf_loading_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        # Step 2: PDF Profiling
        t0 = time.perf_counter()
        pdf_profile = self.analyzer.profile_document(p_path)
        timings["profiling_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        self._save_json(base_out_dir / "profile.json", pdf_profile.model_dump())

        # Step 3: Page Range resolution
        actual_start = max(1, start_page or 1)
        actual_end = min(total_doc_pages, end_page or total_doc_pages)
        target_page_numbers = list(range(actual_start, actual_end + 1))
        pages_to_process_count = len(target_page_numbers)
        is_partial_run = (actual_start > 1) or (actual_end < total_doc_pages)

        # Write Metadata
        metadata = DocumentMetadata(
            document_id=doc_id,
            filename=filename,
            sha256=file_sha256,
            file_size=file_size,
            page_count=total_doc_pages,
            created_at=datetime.now(timezone.utc).isoformat(),
            pipeline_version="ocr_v0",
            pdf_title=pdf_title,
            status=DocumentStatus.PROCESSING,
        )
        self._save_json(base_out_dir / "metadata.json", metadata.model_dump())

        # Step 4, 5, 6: Page Processing Loop
        processed_page_schemas: list[PageSchema] = []
        routing_counts = {"digital_text": 0, "poor_text": 0, "scanned": 0}
        extraction_counts = {"pdfplumber": 0, "baseline_ocr": 0, "cnn_ocr": 0}
        errors_logged: list[dict] = []
        errors_file = base_out_dir / "errors.jsonl"

        with pdfplumber.open(str(p_path)) as plumber_pdf:
            for idx, page_num in enumerate(target_page_numbers):
                page_key = f"{page_num:04d}.json"
                page_file = pages_dir / page_key

                # Resumability check: if page exists and force is False, skip
                storage_page_key = f"processed/{doc_id}/pages/{page_key}"
                is_cached = page_file.exists() or (await self.storage.exists(storage_page_key))
                if not force and is_cached:
                    try:
                        if page_file.exists():
                            with open(page_file, "r", encoding="utf-8") as pf:
                                existing_data = json.load(pf)
                        else:
                            existing_data = await self.storage.get_json(storage_page_key)

                        cached_schema = PageSchema(**existing_data)
                        processed_page_schemas.append(cached_schema)

                        # Track stats
                        if cached_schema.route == PageRoute.DIGITAL_TEXT:
                            routing_counts["digital_text"] += 1
                        elif cached_schema.route == PageRoute.POOR_TEXT_LAYER:
                            routing_counts["poor_text"] += 1
                        else:
                            routing_counts["scanned"] += 1

                        meth = cached_schema.extraction_method
                        extraction_counts[meth] = extraction_counts.get(meth, 0) + 1

                        # Save to local file if only in storage
                        if not page_file.exists():
                            self._save_json(page_file, cached_schema.model_dump())
                        # Save to self.storage for backwards compatibility
                        await self.storage.put_json(storage_page_key, cached_schema.model_dump())
                        continue
                    except Exception:
                        pass  # If corrupt, reprocess

                # Process page
                t_page_start = time.perf_counter()
                try:
                    plumber_page = plumber_pdf.pages[page_num - 1]

                    # Analyze quality
                    t_an = time.perf_counter()
                    quality_report = self.analyzer.analyze(plumber_page, page_num)
                    timings["page_analysis_ms"] += (time.perf_counter() - t_an) * 1000

                    # Route stats
                    if quality_report.route == PageRoute.DIGITAL_TEXT:
                        routing_counts["digital_text"] += 1
                    elif quality_report.route == PageRoute.POOR_TEXT_LAYER:
                        routing_counts["poor_text"] += 1
                    else:
                        routing_counts["scanned"] += 1

                    # Extract
                    t_ext = time.perf_counter()
                    if quality_report.should_use_text_layer:
                        res = self.digital_extractor.extract(
                            plumber_page, page_num=page_num, pdf_path=p_path
                        )
                        extraction = await res if inspect.isawaitable(res) else res
                        timings["extraction_ms"] += (time.perf_counter() - t_ext) * 1000
                    else:
                        rendered = self.renderer.render_page(str(p_path), page_num - 1)
                        res = self.ocr_extractor.extract(
                            rendered.processed_image, page_num=page_num, pdf_path=p_path
                        )
                        extraction = await res if inspect.isawaitable(res) else res
                        timings["ocr_ms"] += (time.perf_counter() - t_ext) * 1000

                    meth = extraction.method
                    extraction_counts[meth] = extraction_counts.get(meth, 0) + 1

                    # Normalize into PageSchema
                    page_schema = self.normalizer.normalize_page(
                        extraction=extraction,
                        document_id=doc_id,
                        page_number=page_num,
                        quality_score=quality_report.text_quality_score,
                    )
                    page_schema.route = quality_report.route
                    processed_page_schemas.append(page_schema)

                    # Persist page JSON locally and to storage
                    self._save_json(page_file, page_schema.model_dump())
                    await self.storage.put_json(
                        f"processed/{doc_id}/pages/{page_key}",
                        page_schema.model_dump(),
                    )

                    # Debug render (e.g. representative pages: page 1, 50, 100, or first/middle/last)
                    if debug:
                        should_render = (
                            page_num in (1, 50, 100, 200, 500)
                            or idx == 0
                            or idx == pages_to_process_count - 1
                            or idx == pages_to_process_count // 2
                        )
                        if should_render:
                            self.debug_visualizer.render_debug_page(
                                p_path, page_schema, debug_dir
                            )

                except Exception as e:
                    log_event(logger, f"Page extraction failed on page {page_num}", document_id=doc_id, page_number=page_num, error=str(e), level=logging.ERROR)
                    err_entry = {
                        "page": page_num,
                        "stage": "extraction",
                        "error": str(e),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    errors_logged.append(err_entry)
                    with open(errors_file, "a", encoding="utf-8") as ef:
                        ef.write(json.dumps(err_entry) + "\n")

                # Progress update
                progress = round(((idx + 1) / pages_to_process_count) * 100.0, 1)
                self._write_job_state(
                    base_out_dir,
                    LocalJobState(
                        document_id=doc_id,
                        filename=filename,
                        status=DocumentStatus.PROCESSING,
                        progress=progress,
                        pages_total=pages_to_process_count,
                        pages_processed=len(processed_page_schemas),
                        pages_failed=len(errors_logged),
                        current_page=page_num,
                    ),
                )
                if progress_callback:
                    try:
                        progress_callback(progress, idx + 1, pages_to_process_count)
                    except TypeError:
                        progress_callback(len(processed_page_schemas), len(errors_logged), pages_to_process_count, page_num)

        # Step 7: Chapter Detection
        t0 = time.perf_counter()
        raw_chapters = self.chapter_detector.detect_chapters(processed_page_schemas, doc_id)
        # Step 8: Section Detection
        chapters = self.section_detector.detect_sections(raw_chapters, processed_page_schemas)
        timings["structure_extraction_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        # Save Chapters JSON
        total_sections = 0
        for ch in chapters:
            total_sections += len(ch.sections)
            ch_file = chapters_dir / f"{ch.chapter_id}.json"
            self._save_json(ch_file, ch.model_dump())

        # Step 9: Chunk Generation
        t0 = time.perf_counter()
        chapter_chunks = self.chunker.chunk_document(chapters, processed_page_schemas)
        timings["chunking_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        total_chunks = 0
        all_chunks = []
        for ch_id, chks in chapter_chunks.items():
            total_chunks += len(chks)
            chunk_file = chunks_dir / f"{ch_id}.jsonl"
            with open(chunk_file, "w", encoding="utf-8") as cf:
                for c in chks:
                    c_dict = c.model_dump()
                    all_chunks.append(c_dict)
                    cf.write(json.dumps(c_dict) + "\n")

        # Step 10: Manifest Generation
        paths_dict = {
            "pages": "pages/",
            "chapters": "chapters/",
            "chunks": "chunks/",
        }
        if debug:
            paths_dict["debug"] = "debug/"

        # Determine TOON configuration
        toon_is_enabled = bool(generate_toon or toon_output_dir is not None)
        toon_info: dict[str, Any] = {"enabled": toon_is_enabled, "path": None}
        toon_out = None
        if toon_is_enabled:
            target_toon_dir = Path(toon_output_dir).resolve() if toon_output_dir else base_out_dir
            target_toon_dir.mkdir(parents=True, exist_ok=True)
            toon_out = target_toon_dir / f"{p_path.stem}.toon"
            toon_info["path"] = str(toon_out)

        manifest = DocumentManifest(
            document_id=doc_id,
            filename=filename,
            page_count=total_doc_pages,
            processed_pages=len(processed_page_schemas),
            failed_pages=len(errors_logged),
            chapters=len(chapters),
            sections=total_sections,
            chunks=total_chunks,
            paths=paths_dict,
            pages=[f"pages/{p.page_number:04d}.json" for p in processed_page_schemas],
            partial_run=is_partial_run,
            page_range=[actual_start, actual_end] if is_partial_run else None,
            metadata={
                "sha256": file_sha256,
                "file_size": file_size,
                "pdf_title": pdf_title,
                "pages_processed": len(processed_page_schemas),
                "pages_failed": len(errors_logged),
                "errors": errors_logged,
            },
            toon=toon_info,
        )
        self._save_json(base_out_dir / "manifest.json", manifest.model_dump())
        await self.storage.put_json(f"processed/{doc_id}/manifest.json", manifest.model_dump())

        # Step 11: Document-Agnostic Structural Validation & Audit
        t0_audit = time.perf_counter()
        audit_dir = base_out_dir / "audit"
        audit_report = self.validator.validate(
            document_id=doc_id,
            pages=processed_page_schemas,
            chapters=chapters,
            chunks=all_chunks,
            manifest=manifest,
        )
        self.validator.write_audit_reports(audit_dir=audit_dir, report=audit_report)
        timings["structural_validation_ms"] = round((time.perf_counter() - t0_audit) * 1000, 2)

        # Step 12: Optional TOON serialization & validation
        toon_size_bytes = None
        toon_duration_s = None
        if toon_is_enabled and toon_out is not None:
            t0_toon = time.perf_counter()
            should_write_toon = True
            if toon_out.exists() and not force:
                log_event(
                    logger,
                    f"TOON output already exists at {toon_out}. Preserving file (use --force to overwrite).",
                    document_id=doc_id,
                    stage="toon_export",
                )
                should_write_toon = False

            if should_write_toon:
                self.toon_writer.write(
                    document_id=doc_id,
                    filename=filename,
                    manifest=manifest,
                    pages=processed_page_schemas,
                    chapters=chapters,
                    chunks=all_chunks,
                    output_path=toon_out,
                )
                try:
                    ToonValidator.validate(
                        toon_path=toon_out,
                        expected_document_id=doc_id,
                        manifest=manifest,
                        output_validation_path=audit_dir / "toon_validation.json",
                    )
                except Exception as e:
                    log_event(logger, f"TOON validation warning: {e}", document_id=doc_id, stage="toon_export", level=logging.WARNING)
            toon_duration_s = time.perf_counter() - t0_toon
            timings["toon_export_ms"] = round(toon_duration_s * 1000, 2)
            if toon_out.exists():
                toon_size_bytes = toon_out.stat().st_size

        # Step 13: Performance & Processing Reports
        total_duration_s = time.perf_counter() - t0_total
        timings["total_processing_ms"] = round(total_duration_s * 1000, 2)
        avg_page_ms = (
            round(timings["total_processing_ms"] / max(1, len(processed_page_schemas)), 2)
        )
        timings["average_time_per_page_ms"] = avg_page_ms
        self._save_json(base_out_dir / "performance.json", timings)

        final_status = (
            DocumentStatus.COMPLETED_WITH_ERRORS if errors_logged else DocumentStatus.COMPLETED
        )
        now_iso = datetime.now(timezone.utc).isoformat()

        report = ProcessingReport(
            document_id=doc_id,
            filename=filename,
            processing={
                "started_at": metadata.created_at,
                "completed_at": now_iso,
                "duration_seconds": round(total_duration_s, 2),
                "status": final_status.value,
            },
            pages={
                "total": total_doc_pages,
                "processed": len(processed_page_schemas),
                "failed": len(errors_logged),
            },
            routing=routing_counts,
            extraction=extraction_counts,
            structure={
                "chapters": len(chapters),
                "sections": total_sections,
                "chunks": total_chunks,
            },
            performance={
                "total_seconds": round(total_duration_s, 2),
                "average_page_ms": avg_page_ms,
            },
            toon_enabled=toon_is_enabled,
            toon_output_path=str(toon_out) if toon_out else None,
            toon_size_bytes=toon_size_bytes,
            toon_generation_time=round(toon_duration_s, 4) if toon_duration_s is not None else None,
        )
        self._save_json(base_out_dir / "report.json", report.model_dump())

        # Update metadata to completed
        metadata.status = final_status
        metadata.completed_at = now_iso
        self._save_json(base_out_dir / "metadata.json", metadata.model_dump())

        # Final atomic Job State
        self._write_job_state(
            base_out_dir,
            LocalJobState(
                document_id=doc_id,
                filename=filename,
                status=final_status,
                progress=100.0,
                pages_total=pages_to_process_count,
                pages_processed=len(processed_page_schemas),
                pages_failed=len(errors_logged),
            ),
        )

        log_event(
            logger,
            f"Document pipeline completed for {doc_id}",
            document_id=doc_id,
            status=final_status.value,
            duration_ms=round(total_duration_s * 1000, 2),
        )

        return manifest

    def _save_json(self, path: Path, data: dict[str, Any]) -> None:
        """Atomic write JSON file."""
        tmp_path = path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        tmp_path.replace(path)

    def _write_job_state(self, base_dir: Path, state: LocalJobState) -> None:
        """Atomic write of local job state."""
        job_file = base_dir / "job.json"
        self._save_json(job_file, state.model_dump())
