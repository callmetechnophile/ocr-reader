from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="OCR_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    APP_NAME: str = "Textbook OCR Microservice"
    APP_ENV: str = "development"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    MAX_FILE_SIZE_MB: int = Field(default=250, description="Max allowed PDF upload size in MB")
    RENDER_DPI: int = Field(default=150, description="DPI for rendering PDF pages to image")
    TEXT_QUALITY_THRESHOLD: float = Field(
        default=0.70, description="Minimum quality score to use pdfplumber text layer"
    )
    STORAGE_PATH: str = Field(default="./data", description="Base directory for object storage")
    WORKER_COUNT: int = Field(default=4, description="Background task concurrency limit")
    LOG_LEVEL: str = Field(default="INFO", description="Application log level")
    BASELINE_BACKEND: str = Field(
        default="baseline", description="OCR backend name ('baseline' or 'cnn_ocr')"
    )
    MODEL_DIR: str = Field(default="./models/ocr", description="Directory for OCR weights")
    MODEL_PATH: str = Field(default="./models/ocr/crnn_v1_best.pt", description="Path to CRNN checkpoint")
    MODEL_DEVICE: str = Field(default="cpu", description="Inference device ('cpu' or 'cuda')")
    MODEL_VERSION: str = Field(default="crnn_v1", description="Model version tag")
    VOCAB_PATH: str = Field(default="./models/ocr/vocab.json", description="Path to vocab.json")

    @property
    def max_file_size_bytes(self) -> int:
        return self.MAX_FILE_SIZE_MB * 1024 * 1024

    @property
    def raw_storage_path(self) -> Path:
        p = Path(self.STORAGE_PATH) / "raw"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def processed_storage_path(self) -> Path:
        p = Path(self.STORAGE_PATH) / "processed"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def temp_storage_path(self) -> Path:
        p = Path(self.STORAGE_PATH) / "temp"
        p.mkdir(parents=True, exist_ok=True)
        return p


settings = Settings()
