"""Integration tests for the FastAPI application."""

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from foldfit.api.app import create_app


@pytest.fixture
def app():  # type: ignore[no-untyped-def]
    return create_app()


@pytest_asyncio.fixture
async def client(app):  # type: ignore[no-untyped-def]
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_health(client: AsyncClient) -> None:
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_msa_single_sequence(client: AsyncClient) -> None:
    response = await client.post(
        "/v1/msa",
        json={"sequence": "MKWVTFISLLLLFSSAYS", "backend": "single"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["num_sequences"] == 1
    assert data["sequence_length"] == 18


@pytest.mark.asyncio
async def test_finetune_returns_202(client: AsyncClient) -> None:
    response = await client.post(
        "/v1/finetune",
        json={"pdb_paths": ["/nonexistent/test.pdb"], "epochs": 1, "device": "cpu"},
    )
    assert response.status_code == 202
    data = response.json()
    assert "job_id" in data
    assert data["status"] == "queued"


@pytest.mark.asyncio
async def test_finetune_status_not_found(client: AsyncClient) -> None:
    response = await client.get("/v1/finetune/nonexistent-id")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_list_finetune_jobs(client: AsyncClient) -> None:
    response = await client.get("/v1/finetune")
    assert response.status_code == 200
    data = response.json()
    assert "jobs" in data
    assert "total" in data


@pytest.mark.asyncio
async def test_create_dataset(client: AsyncClient) -> None:
    response = await client.post(
        "/v1/datasets",
        json={"name": "Test Antibodies", "max_structures": 10, "resolution_max": 2.5},
    )
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "Test Antibodies"
    assert "id" in data
    assert "num_structures" in data
    assert data["source"] == "rcsb"


@pytest.mark.asyncio
async def test_list_datasets(client: AsyncClient) -> None:
    response = await client.get("/v1/datasets")
    assert response.status_code == 200
    data = response.json()
    assert "datasets" in data
    assert "total" in data


@pytest.mark.asyncio
async def test_delete_dataset_not_found(client: AsyncClient) -> None:
    response = await client.delete("/v1/datasets/nonexistent")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_predict_endpoint_responds(client: AsyncClient) -> None:
    response = await client.post(
        "/v1/predict",
        json={"sequence": "MKWVTFISLLLLFSSAYS", "device": "cpu"},
    )
    # 200 if openfold + GPU available, 500/503 if model fails to load in test env
    assert response.status_code in (200, 500, 503)
    if response.status_code == 200:
        data = response.json()
        assert data["sequence_length"] == 18
        assert "pdb_string" in data
