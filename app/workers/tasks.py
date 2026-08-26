import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from app.core.config import settings
from app.core.logging import log_event, logger
from app.pipeline.orchestrator import DocumentPipelineOrchestrator
from app.schemas.document import DocumentStatus, DocumentStatusResponse
from app.storage.object_store import get_object_store


class TaskState:
    def __init__(self, document_id: str, total_pages: int = 0):
        self.document_id = document_id
        self.status = DocumentStatus.QUEUED
        self.pages_total = total_pages
        self.pages_processed = 0
        self.pages_failed = 0
        self.current_page: Optional[int] = None
        self.error: Optional[str] = None
        self.created_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)

    @property
    def progress_percentage(self) -> float:
        if self.pages_total <= 0:
            return 0.0
        return round((self.pages_processed / self.pages_total) * 100.0, 2)

    def to_status_response(self) -> DocumentStatusResponse:
        return DocumentStatusResponse(
            document_id=self.document_id,
            status=self.status,
            progress=self.progress_percentage,
            pages_processed=self.pages_processed,
            pages_failed=self.pages_failed,
            total_pages=self.pages_total,
            current_page=self.current_page,
            error=self.error,
        )


class BackgroundTaskWorker:
    """
    In-process asynchronous task worker for managing background document processing.
    Designed with a decoupled interface ready to be replaced with Celery or Redis queue.
    """

    def __init__(self, orchestrator: Optional[DocumentPipelineOrchestrator] = None):
        self.orchestrator = orchestrator or DocumentPipelineOrchestrator()
        self._tasks: dict[str, TaskState] = {}
        self._running_tasks: dict[str, asyncio.Task[Any]] = {}

    def register_task(self, document_id: str, total_pages: int = 0) -> TaskState:
        state = TaskState(document_id, total_pages)
        self._tasks[document_id] = state
        return state

    def get_task_status(self, document_id: str) -> Optional[DocumentStatusResponse]:
        state = self._tasks.get(document_id)
        if not state:
            return None
        return state.to_status_response()

    async def get_state(self, document_id: str) -> Optional[TaskState]:
        return self._tasks.get(document_id)

    async def enqueue(self, document_id: str, pdf_path: str | Path) -> None:
        await self.enqueue_document_processing(document_id, pdf_path)

    async def enqueue_document_processing(self, document_id: str, file_path: str | Path) -> None:
        state = self._tasks.get(document_id)
        if not state:
            state = self.register_task(document_id)

        task = asyncio.create_task(self._run_processing_job(document_id, file_path))
        self._running_tasks[document_id] = task

    async def _run_processing_job(self, document_id: str, file_path: str | Path) -> None:
        state = self._tasks[document_id]
        state.status = DocumentStatus.PROCESSING
        state.updated_at = datetime.now(timezone.utc)

        def on_progress(processed: int, failed: int, total: int, current: Optional[int]) -> None:
            state.pages_processed = processed
            state.pages_failed = failed
            state.pages_total = total
            state.current_page = current
            state.updated_at = datetime.now(timezone.utc)

        try:
            manifest = await self.orchestrator.process_document(
                pdf_path=file_path,
                document_id=document_id,
                progress_callback=on_progress,
            )
            state.status = DocumentStatus.COMPLETED if state.pages_failed == 0 else DocumentStatus.COMPLETED_WITH_ERRORS
            state.current_page = None
            state.updated_at = datetime.now(timezone.utc)
            log_event(logger, f"Background task completed for {document_id}", document_id=document_id)
        except Exception as exc:
            state.status = DocumentStatus.FAILED
            state.error = str(exc)
            state.updated_at = datetime.now(timezone.utc)
            log_event(logger, f"Background task failed for {document_id}: {exc}", document_id=document_id, error=str(exc))
        finally:
            self._running_tasks.pop(document_id, None)


# Global worker instance
task_worker = BackgroundTaskWorker()
