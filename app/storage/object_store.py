import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional
import aiofiles
from app.core.config import settings
from app.core.logging import logger


class ObjectStore(ABC):
    """Abstract object storage interface ready for local disk or S3/MinIO backend."""

    @abstractmethod
    async def put(self, key: str, data: bytes) -> str:
        """Write raw bytes to the specified key."""
        pass

    @abstractmethod
    async def put_json(self, key: str, data: dict[str, Any] | list[Any]) -> str:
        """Serialize and write JSON data to the specified key."""
        pass

    @abstractmethod
    async def get(self, key: str) -> bytes:
        """Read raw bytes from the specified key."""
        pass

    @abstractmethod
    async def get_json(self, key: str) -> Any:
        """Read and deserialize JSON data from the specified key."""
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if an object exists at the specified key."""
        pass

    @abstractmethod
    async def list_keys(self, prefix: str = "") -> list[str]:
        """List all object keys matching the prefix."""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete object at the specified key."""
        pass


class LocalFileSystemStore(ObjectStore):
    """Production local filesystem implementation of ObjectStore."""

    def __init__(self, base_path: Optional[str | Path] = None):
        self.base_path = Path(base_path or settings.STORAGE_PATH).resolve()
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _resolve_path(self, key: str) -> Path:
        clean_key = key.lstrip("/\\")
        full_path = (self.base_path / clean_key).resolve()
        # Security check: ensure path does not escape base directory
        if not str(full_path).startswith(str(self.base_path)):
            raise ValueError(f"Path traversal detected: {key}")
        return full_path

    async def put(self, key: str, data: bytes) -> str:
        target = self._resolve_path(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(target, "wb") as f:
            await f.write(data)
        return str(target)

    async def put_json(self, key: str, data: dict[str, Any] | list[Any]) -> str:
        target = self._resolve_path(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        json_str = json.dumps(data, indent=2, ensure_ascii=False)
        async with aiofiles.open(target, "w", encoding="utf-8") as f:
            await f.write(json_str)
        return str(target)

    async def get(self, key: str) -> bytes:
        target = self._resolve_path(key)
        if not target.exists():
            raise FileNotFoundError(f"Object not found: {key}")
        async with aiofiles.open(target, "rb") as f:
            return await f.read()

    async def get_json(self, key: str) -> Any:
        target = self._resolve_path(key)
        if not target.exists():
            raise FileNotFoundError(f"Object not found: {key}")
        async with aiofiles.open(target, "r", encoding="utf-8") as f:
            content = await f.read()
            return json.loads(content)

    async def exists(self, key: str) -> bool:
        target = self._resolve_path(key)
        return target.exists()

    async def list_keys(self, prefix: str = "") -> list[str]:
        target_dir = self._resolve_path(prefix)
        if not target_dir.exists():
            return []
        if target_dir.is_file():
            rel = target_dir.relative_to(self.base_path)
            return [str(rel).replace("\\", "/")]

        keys = []
        for p in target_dir.rglob("*"):
            if p.is_file():
                rel = p.relative_to(self.base_path)
                keys.append(str(rel).replace("\\", "/"))
        return sorted(keys)

    async def delete(self, key: str) -> bool:
        target = self._resolve_path(key)
        if target.exists():
            if target.is_file():
                target.unlink()
                return True
        return False


def get_object_store() -> ObjectStore:
    return LocalFileSystemStore(settings.STORAGE_PATH)
