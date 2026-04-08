"""Dataset management endpoints.

CRUD operations for antibody structure datasets sourced from SAbDab.
Datasets are persisted as JSON in data/datasets/ with PDB files alongside.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from foldfit.api.schemas import (
    CreateDatasetRequest,
    DatasetListResponse,
    DatasetResponse,
)
from foldfit.infrastructure.data.dataset_store import DatasetStore
from foldfit.infrastructure.data.sabdab_repository import SabdabRepository

router = APIRouter(prefix="/v1/datasets", tags=["Datasets"])
logger = logging.getLogger(__name__)

_store = DatasetStore()
_repo = SabdabRepository()


@router.post(
    "",
    response_model=DatasetResponse,
    status_code=201,
    summary="Create a new dataset",
    description=(
        "Queries SAbDab for antibody PDB IDs filtered by resolution, "
        "downloads structures from RCSB, and saves everything to disk."
    ),
)
async def create_dataset(body: CreateDatasetRequest) -> DatasetResponse:
    dataset_id = str(uuid.uuid4())[:8]
    pdb_dir = _store.pdb_dir(dataset_id)

    # Query RCSB for antibody PDB IDs with filters
    logger.info(
        f"Creating dataset '{body.name}': type={body.antibody_type}, "
        f"res<={body.resolution_max}A, organism='{body.organism}', "
        f"method='{body.method}', max={body.max_structures}"
    )
    pdb_ids = _repo.query_antibody_pdb_ids(
        resolution_max=body.resolution_max,
        max_results=body.max_structures,
        antibody_type=body.antibody_type,
        organism=body.organism,
        method=body.method,
    )

    downloaded: list[str] = []
    for pdb_id in pdb_ids:
        dest = pdb_dir / f"{pdb_id}.pdb"
        if dest.exists() or _repo.download_pdb(pdb_id, dest):
            downloaded.append(str(dest))

    # Persist metadata
    metadata = {
        "id": dataset_id,
        "name": body.name,
        "num_structures": len(downloaded),
        "resolution_max": body.resolution_max,
        "antibody_type": body.antibody_type,
        "organism": body.organism,
        "method": body.method,
        "source": "rcsb",
        "pdb_paths": downloaded,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
    }
    _store.save(dataset_id, metadata)

    logger.info(f"Dataset '{body.name}' created: {len(downloaded)} structures in {pdb_dir}")
    return DatasetResponse(**metadata)


@router.get(
    "",
    response_model=DatasetListResponse,
    summary="List all datasets",
    description="Returns all datasets stored on disk, ordered by creation date (newest first).",
)
async def list_datasets() -> DatasetListResponse:
    items = _store.list_all()
    datasets = [DatasetResponse(**d) for d in items]
    return DatasetListResponse(datasets=datasets, total=len(datasets))


@router.get(
    "/{dataset_id}",
    response_model=DatasetResponse,
    summary="Get dataset details",
    description="Returns full details for a single dataset including PDB file paths.",
)
async def get_dataset(dataset_id: str) -> DatasetResponse:
    data = _store.load(dataset_id)
    if data is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")
    return DatasetResponse(**data)


@router.delete(
    "/{dataset_id}",
    status_code=204,
    summary="Delete a dataset",
    description="Permanently removes dataset metadata and all downloaded PDB files.",
)
async def delete_dataset(dataset_id: str) -> None:
    if not _store.exists(dataset_id):
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")
    _store.delete(dataset_id)
