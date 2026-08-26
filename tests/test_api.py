import asyncio
from pathlib import Path
import pytest
from httpx import ASGITransport, AsyncClient
from app.main import app
from app.storage.object_store import LocalFileSystemStore


@pytest.mark.asyncio
async def test_api_upload_and_status_flow(digital_pdf_path: Path, temp_storage: LocalFileSystemStore):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # 1. Upload PDF
        with open(digital_pdf_path, "rb") as f:
            response = await client.post(
                "/v1/documents",
                files={"file": ("textbook.pdf", f, "application/pdf")},
            )

        assert response.status_code == 202
        data = response.json()
        assert "document_id" in data
        assert data["status"] == "queued"
        doc_id = data["document_id"]

        # 2. Poll until completed or timeout
        for _ in range(30):
            status_resp = await client.get(f"/v1/documents/{doc_id}")
            assert status_resp.status_code == 200
            status_data = status_resp.json()
            if status_data["status"] in ("completed", "completed_with_errors"):
                break
            await asyncio.sleep(0.2)

        assert status_data["status"] in ("completed", "completed_with_errors")
        assert status_data["pages_processed"] == 2
        assert status_data["progress"] == 100.0

        # 3. Retrieve manifest
        manifest_resp = await client.get(f"/v1/documents/{doc_id}/manifest")
        assert manifest_resp.status_code == 200
        manifest_data = manifest_resp.json()
        assert manifest_data["document_id"] == doc_id
        assert manifest_data["page_count"] == 2
        assert len(manifest_data["pages"]) == 2

        # 4. Retrieve single page
        page_resp = await client.get(f"/v1/documents/{doc_id}/pages/1")
        assert page_resp.status_code == 200
        page_data = page_resp.json()
        assert page_data["page_number"] == 1
        assert len(page_data["regions"]) > 0


@pytest.mark.asyncio
async def test_api_invalid_file_extension():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/v1/documents",
            files={"file": ("textbook.txt", b"not a pdf", "text/plain")},
        )
        assert response.status_code == 400
        assert "Invalid file extension" in response.json()["detail"]


@pytest.mark.asyncio
async def test_api_invalid_header():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/v1/documents",
            files={"file": ("corrupt.pdf", b"NOT_PDF_HEADER_12345", "application/pdf")},
        )
        assert response.status_code == 400
        assert "Header magic bytes" in response.json()["detail"]


@pytest.mark.asyncio
async def test_api_document_not_found():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/v1/documents/doc_nonexistent")
        assert response.status_code == 404
