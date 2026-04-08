"""Fine-tuning endpoints.

Start, monitor, list, and cancel LoRA fine-tuning jobs.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, BackgroundTasks, HTTPException

from foldfit.api.schemas import (
    FinetuneJobListResponse,
    FinetuneJobResponse,
    FinetuneRequest,
)
from foldfit.infrastructure.data.dataset_store import DatasetStore

router = APIRouter(prefix="/v1/finetune", tags=["Training"])
logger = logging.getLogger(__name__)

# In-memory job store (replace with Redis/DB for production)
_jobs: dict[str, FinetuneJobResponse] = {}
_dataset_store = DatasetStore()


def _run_finetune(job_id: str, body: FinetuneRequest) -> None:
    """Background task that runs the actual fine-tuning."""
    job = _jobs.get(job_id)
    if not job:
        return

    job.status = "running"
    try:
        # Aggressively free GPU memory from any previous predict/train
        import gc
        import torch
        gc.collect()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            allocated = torch.cuda.memory_allocated() / 1024**3
            logger.info(f"GPU before training: {allocated:.1f} GB allocated")
            if allocated > 2.0:
                logger.warning(f"GPU has {allocated:.1f} GB allocated — previous model may not have been freed. Restart backend if OOM.")

        from foldfit.application.finetune_service import FinetuneService
        from foldfit.domain.value_objects import (
            DataConfig,
            FoldfitConfig,
            LoraConfig,
            ModelConfig,
            MsaConfig,
            OutputConfig,
            TrainingConfig,
        )
        from foldfit.infrastructure.checkpoint_store import FileCheckpointStore
        from foldfit.infrastructure.data.msa_provider import MsaProvider
        from foldfit.infrastructure.data.sabdab_repository import SabdabRepository
        from foldfit.infrastructure.openfold.adapter import OpenFoldAdapter
        from foldfit.infrastructure.peft.injector import LoraInjector

        # Resolve dataset_id to PDB paths
        pdb_paths = list(body.pdb_paths)
        if body.dataset_id:
            ds_data = _dataset_store.load(body.dataset_id)
            if ds_data and ds_data.get("pdb_paths"):
                pdb_paths = ds_data["pdb_paths"]
                logger.info(f"Resolved dataset '{body.dataset_id}': {len(pdb_paths)} PDB files")
            else:
                logger.warning(f"Dataset '{body.dataset_id}' not found or empty")

        if not pdb_paths:
            raise ValueError(
                f"No PDB files found. Dataset '{body.dataset_id}' may be empty or not exist. "
                "Create a dataset first in the Data tab."
            )

        # Detect device
        import torch
        device = body.device
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            device = "cpu"
        if device == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"Using GPU: {gpu_name} ({vram:.1f} GB VRAM)")

        config = FoldfitConfig(
            model=ModelConfig(weights_path=body.weights_path, head=body.head, device=device),
            data=DataConfig(pdb_paths=pdb_paths, max_seq_len=body.max_seq_len),
            training=TrainingConfig(
                epochs=body.epochs,
                learning_rate=body.learning_rate,
                lr_lora=body.lr_lora,
                lr_head=body.lr_head,
                scheduler=body.scheduler,
                warmup_steps=body.warmup_steps,
                accumulation_steps=body.accumulation_steps,
                amp=body.amp,
                early_stopping_patience=body.early_stopping_patience,
                grad_clip=body.grad_clip,
            ),
            lora=LoraConfig(
                rank=body.lora_rank,
                alpha=body.lora_alpha,
                dropout=body.lora_dropout,
                target_modules=body.target_modules,
            ),
            msa=MsaConfig(backend=body.msa_backend),
            output=OutputConfig(checkpoint_dir=body.checkpoint_dir),
        )

        msa_config = MsaConfig(backend=body.msa_backend)
        service = FinetuneService(
            model=OpenFoldAdapter(),
            peft=LoraInjector(),
            dataset=SabdabRepository(),
            msa=MsaProvider(msa_config),
            checkpoint=FileCheckpointStore(),
        )
        result = service.run(config)
        job.status = result.status
        job.metrics = result.metrics
        if result.metrics:
            job.train_loss = result.metrics.get("train_loss", 0.0)
            job.val_loss = result.metrics.get("val_loss")
            job.epoch = int(result.metrics.get("epoch", 0))
            job.progress = 100.0 if result.status == "completed" else 0.0

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        logger.error(f"Fine-tuning failed:\n{tb}")
        job.status = "failed"
        job.metrics = {"error": str(e) or tb.split("\n")[-2]}


@router.post(
    "",
    response_model=FinetuneJobResponse,
    status_code=202,
    summary="Start a fine-tuning job",
    description=(
        "Submits a new LoRA fine-tuning job. The job runs in the background. "
        "Use GET /v1/finetune/{job_id} to poll for status and metrics."
    ),
)
async def start_finetune(
    body: FinetuneRequest, background_tasks: BackgroundTasks
) -> FinetuneJobResponse:
    job_id = str(uuid.uuid4())[:8]
    job = FinetuneJobResponse(
        job_id=job_id,
        name=body.name or f"job-{job_id}",
        dataset_id=body.dataset_id,
        status="queued",
        total_epochs=body.epochs,
        lora_rank=body.lora_rank,
        config=body.model_dump(),
        created_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
    )
    _jobs[job_id] = job
    background_tasks.add_task(_run_finetune, job_id, body)
    return job


@router.get(
    "",
    response_model=FinetuneJobListResponse,
    summary="List all fine-tuning jobs",
    description="Returns all submitted jobs ordered by creation date (newest first).",
)
async def list_finetune_jobs() -> FinetuneJobListResponse:
    items = list(_jobs.values())
    items.reverse()
    return FinetuneJobListResponse(jobs=items, total=len(items))


@router.get(
    "/{job_id}",
    response_model=FinetuneJobResponse,
    summary="Get fine-tuning job status",
    description="Returns current status, progress, and training metrics for a job.",
)
async def get_finetune_status(job_id: str) -> FinetuneJobResponse:
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return _jobs[job_id]


@router.delete(
    "/{job_id}",
    status_code=204,
    summary="Delete a fine-tuning job",
    description="Removes a completed or failed job from the list.",
)
async def delete_finetune_job(job_id: str) -> None:
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    del _jobs[job_id]
