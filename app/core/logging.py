import logging
import sys
import time
from contextlib import contextmanager
from typing import Any, Generator, Optional
from app.core.config import settings


class StructuredFormatter(logging.Formatter):
    """Formats log records with standard prefix and key=value pairs for observability."""

    def format(self, record: logging.LogRecord) -> str:
        base_msg = record.getMessage()
        extra_fields = []
        for key in ["document_id", "page_number", "stage", "duration_ms", "status", "error"]:
            if hasattr(record, key):
                val = getattr(record, key)
                if val is not None:
                    extra_fields.append(f"{key}={val}")

        extra_str = " " + " ".join(extra_fields) if extra_fields else ""
        return f"{self.formatTime(record, self.datefmt)} [{record.levelname}] {record.name}: {base_msg}{extra_str}"


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("ocr_service")
    logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            StructuredFormatter(datefmt="%Y-%m-%d %H:%M:%S")
        )
        logger.addHandler(handler)

    return logger


logger = setup_logging()


def log_event(
    logger_instance: logging.Logger,
    message: str,
    *,
    document_id: Optional[str] = None,
    page_number: Optional[int] = None,
    stage: Optional[str] = None,
    duration_ms: Optional[float] = None,
    status: Optional[str] = None,
    error: Optional[str] = None,
    level: int = logging.INFO,
) -> None:
    """Log an event with standardized structured telemetry attributes."""
    extra = {
        "document_id": document_id,
        "page_number": page_number,
        "stage": stage,
        "duration_ms": round(duration_ms, 2) if duration_ms is not None else None,
        "status": status,
        "error": error,
    }
    logger_instance.log(level, message, extra=extra)


@contextmanager
def log_stage_timing(
    logger_instance: logging.Logger,
    stage_name: str,
    *,
    document_id: Optional[str] = None,
    page_number: Optional[int] = None,
) -> Generator[dict[str, Any], None, None]:
    """Context manager to measure and log execution duration for a pipeline stage."""
    start_time = time.perf_counter()
    ctx: dict[str, Any] = {"status": "success", "error": None}
    try:
        yield ctx
    except Exception as exc:
        ctx["status"] = "failed"
        ctx["error"] = str(exc)
        duration_ms = (time.perf_counter() - start_time) * 1000
        log_event(
            logger_instance,
            f"Stage {stage_name} failed",
            document_id=document_id,
            page_number=page_number,
            stage=stage_name,
            duration_ms=duration_ms,
            status="failed",
            error=str(exc),
            level=logging.ERROR,
        )
        raise
    else:
        duration_ms = (time.perf_counter() - start_time) * 1000
        log_event(
            logger_instance,
            f"Stage {stage_name} completed",
            document_id=document_id,
            page_number=page_number,
            stage=stage_name,
            duration_ms=duration_ms,
            status=ctx.get("status", "success"),
            error=ctx.get("error"),
            level=logging.INFO,
        )
