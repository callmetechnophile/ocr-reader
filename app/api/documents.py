import hashlib
import json
from pathlib import Path
from typing import Annotated, Optional
import aiofiles
import pymupdf
from fastapi import APIRouter, File, HTTPException, UploadFile, status
from app.core.config import settings
from app.core.logging import log_event, logger
from app.schemas.document import (
    DocumentCreateResponse,
    DocumentManifest,
    DocumentStatus,
    DocumentStatusResponse,
    LocalJobState,
)
from app.schemas.page import PageSchema
from app.storage.object_store import get_object_store
from app.workers.tasks import task_worker

router = APIRouter(prefix="/v1/documents", tags=["Documents"])


@router.post(
    "",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=DocumentCreateResponse,
    summary="Upload textbook PDF for asynchronous OCR processing",
)
async def upload_document(
    file: Annotated[UploadFile, File(description="PDF textbook file")],
) -> DocumentCreateResponse:
    # 1. Validate file presence
    if not file or not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file uploaded",
        )

    # 2. Validate PDF extension
    filename = file.filename.strip()
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file extension for '{filename}'. Only .pdf files are accepted.",
        )

    # 3. Read header to validate magic bytes and compute initial size
    content_chunk = await file.read(1024)
    if not content_chunk:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty",
        )

    if not content_chunk.startswith(b"%PDF"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid PDF file format. Header magic bytes '%PDF' missing.",
        )

    # Temporary staging path to compute content hash
    hasher = hashlib.sha256()
    hasher.update(content_chunk)
    temp_raw = settings.temp_storage_path / f"upload_{file.filename}"
    settings.temp_storage_path.mkdir(parents=True, exist_ok=True)
    settings.raw_storage_path.mkdir(parents=True, exist_ok=True)

    total_size = len(content_chunk)
    async with aiofiles.open(temp_raw, "wb") as out_file:
        await out_file.write(content_chunk)
        while chunk := await file.read(1024 * 1024):  # 1MB chunks
            total_size += len(chunk)
            if total_size > settings.max_file_size_bytes:
                await out_file.close()
                if temp_raw.exists():
                    temp_raw.unlink()
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"File exceeds maximum allowed size of {settings.MAX_FILE_SIZE_MB}MB",
                )
            hasher.update(chunk)
            await out_file.write(chunk)

    file_sha256 = hasher.hexdigest()
    # Content-based stable document ID
    document_id = f"doc_{file_sha256[:16]}"
    final_raw_path = settings.raw_storage_path / f"{document_id}.pdf"

    if temp_raw.exists():
        temp_raw.replace(final_raw_path)

    # 4. Validate readability and page count
    try:
        with pymupdf.open(str(final_raw_path)) as doc:
            page_count = len(doc)
            if page_count == 0:
                if final_raw_path.exists():
                    final_raw_path.unlink()
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="PDF contains 0 pages",
                )
    except HTTPException:
        raise
    except Exception as exc:
        if final_raw_path.exists():
            final_raw_path.unlink()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Corrupted or unreadable PDF: {exc}",
        )

    # Dispatch to background task worker
    await task_worker.enqueue(document_id=document_id, pdf_path=final_raw_path)

    log_event(
        logger,
        "Uploaded textbook PDF accepted for processing",
        document_id=document_id,
        stage="upload_accept",
    )

    return DocumentCreateResponse(
        document_id=document_id,
        filename=filename,
        status=DocumentStatus.QUEUED,
    )


@router.get(
    "/{document_id}",
    response_model=DocumentStatusResponse,
    summary="Get document processing status and progress",
)
async def get_document_status(document_id: str) -> DocumentStatusResponse:
    # 1. Check local job.json file first
    job_file = Path(f"./data/processed/{document_id}/job.json")
    if job_file.exists():
        try:
            with open(job_file, "r", encoding="utf-8") as jf:
                job_data = json.load(jf)
            return DocumentStatusResponse(
                document_id=document_id,
                filename=job_data.get("filename"),
                status=DocumentStatus(job_data.get("status", "processing")),
                progress=float(job_data.get("progress", 0.0)),
                pages_processed=int(job_data.get("pages_processed", 0)),
                pages_failed=int(job_data.get("pages_failed", 0)),
                total_pages=int(job_data.get("pages_total", 0)),
                current_page=job_data.get("current_page"),
                error=job_data.get("error"),
            )
        except Exception:
            pass

    # 2. Check in-memory task worker
    task_state = await task_worker.get_state(document_id)
    if task_state is not None:
        return task_state.to_status_response()

    # 3. Check storage for existing manifest
    storage = get_object_store()
    manifest_key = f"processed/{document_id}/manifest.json"
    manifest_exists = await storage.exists(manifest_key)

    if manifest_exists:
        manifest_data = await storage.get_json(manifest_key)
        total_p = manifest_data.get("page_count", 0)
        proc_p = manifest_data.get("processed_pages", len(manifest_data.get("pages", [])))
        fail_p = manifest_data.get("failed_pages", 0)
        status_val = DocumentStatus.COMPLETED if fail_p == 0 else DocumentStatus.COMPLETED_WITH_ERRORS
        return DocumentStatusResponse(
            document_id=document_id,
            filename=manifest_data.get("filename"),
            status=status_val,
            progress=100.0,
            pages_processed=proc_p,
            pages_failed=fail_p,
            total_pages=total_p,
            current_page=total_p,
            error=None,
        )

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Document with ID '{document_id}' not found.",
    )


@router.get(
    "/{document_id}/manifest",
    response_model=DocumentManifest,
    summary="Get completed document manifest and page listing",
)
async def get_document_manifest(document_id: str) -> DocumentManifest:
    # 1. Check local processed directory
    local_manifest = Path(f"./data/processed/{document_id}/manifest.json")
    if local_manifest.exists():
        with open(local_manifest, "r", encoding="utf-8") as f:
            return DocumentManifest(**json.load(f))

    storage = get_object_store()
    manifest_key = f"processed/{document_id}/manifest.json"

    if not await storage.exists(manifest_key):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Manifest for document '{document_id}' not found or processing incomplete.",
        )

    manifest_data = await storage.get_json(manifest_key)
    return DocumentManifest(**manifest_data)


@router.get(
    "/{document_id}/pages/{page_number}",
    response_model=PageSchema,
    summary="Get single normalized page JSON with bounding boxes and reading order",
)
async def get_document_page(document_id: str, page_number: int) -> PageSchema:
    # 1. Check local processed directory
    local_page = Path(f"./data/processed/{document_id}/pages/{page_number:04d}.json")
    if local_page.exists():
        with open(local_page, "r", encoding="utf-8") as f:
            return PageSchema(**json.load(f))

    storage = get_object_store()
    page_key = f"processed/{document_id}/pages/{page_number:04d}.json"

    if not await storage.exists(page_key):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Page {page_number} for document '{document_id}' not found.",
        )

    page_data = await storage.get_json(page_key)
    return PageSchema(**page_data)
