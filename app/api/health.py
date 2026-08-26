from pathlib import Path
from typing import Any
from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
from app.core.config import settings

router = APIRouter(tags=["Health"])


@router.get("/health", status_code=status.HTTP_200_OK)
async def health_check() -> dict[str, str]:
    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "environment": settings.APP_ENV,
    }


@router.get("/ready", status_code=status.HTTP_200_OK)
async def readiness_check() -> JSONResponse:
    storage_ok = False
    model_dir_ok = False

    try:
        # Check storage write readiness
        test_file = Path(settings.STORAGE_PATH) / "temp" / ".ready_check"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_text("ok")
        if test_file.exists():
            test_file.unlink()
            storage_ok = True
    except Exception:
        storage_ok = False

    try:
        model_dir = Path(settings.MODEL_DIR)
        model_dir_ok = model_dir.exists()
    except Exception:
        model_dir_ok = False

    is_ready = storage_ok and model_dir_ok
    status_code = status.HTTP_200_OK if is_ready else status.HTTP_503_SERVICE_UNAVAILABLE

    return JSONResponse(
        status_code=status_code,
        content={
            "status": "ready" if is_ready else "unready",
            "storage_ready": storage_ok,
            "models_ready": model_dir_ok,
            "baseline_backend": settings.BASELINE_BACKEND,
        },
    )
