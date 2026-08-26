from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.api.documents import router as documents_router
from app.api.health import router as health_router
from app.core.config import settings
from app.core.logging import logger, setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Startup: Ensure directories and logging are configured
    setup_logging()
    settings.raw_storage_path
    settings.processed_storage_path
    settings.temp_storage_path
    logger.info(
        f"Starting {settings.APP_NAME} in {settings.APP_ENV} mode",
        extra={"stage": "startup", "status": "initialized"},
    )
    yield
    # Shutdown
    logger.info(
        f"Shutting down {settings.APP_NAME}",
        extra={"stage": "shutdown", "status": "terminated"},
    )


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        version="0.1.0",
        description="Production-grade asynchronous OCR microservice for large engineering textbooks.",
        lifespan=lifespan,
    )

    # CORS configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register Routers
    app.include_router(health_router)
    app.include_router(documents_router)

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error(
            f"Unhandled exception on {request.method} {request.url.path}: {exc}",
            extra={"error": str(exc), "stage": "global_exception_handler"},
            exc_info=True,
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error occurred in OCR microservice."},
        )

    return app


app = create_app()
