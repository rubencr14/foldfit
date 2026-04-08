"""API request/response schemas for all endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# ── Datasets ──────────────────────────────────────────────────────────────────


class CreateDatasetRequest(BaseModel):
    """Request to create a new antibody structure dataset."""

    name: str = Field(min_length=1, max_length=200, description="Human-readable dataset name")
    max_structures: int = Field(default=200, ge=1, le=5000, description="Maximum number of structures to fetch")
    resolution_max: float = Field(default=3.0, gt=0, le=10.0, description="Maximum PDB resolution in angstroms")
    antibody_type: str = Field(
        default="all",
        description="Type filter: all, antibody, nanobody, Fab, scFv, immunoglobulin",
    )
    organism: str = Field(
        default="",
        description="Organism filter (e.g. 'Homo sapiens', 'Mus musculus'). Empty = any.",
    )
    method: str = Field(
        default="",
        description="Experimental method: '' (any), 'X-RAY DIFFRACTION', 'ELECTRON MICROSCOPY', 'SOLUTION NMR'",
    )


class DatasetResponse(BaseModel):
    """A single dataset record."""

    id: str
    name: str
    num_structures: int
    resolution_max: float
    antibody_type: str = "all"
    organism: str = ""
    method: str = ""
    source: str = "rcsb"
    pdb_paths: list[str] = Field(default_factory=list)
    created_at: str


class DatasetListResponse(BaseModel):
    """List of datasets."""

    datasets: list[DatasetResponse]
    total: int


# ── Fine-tuning ───────────────────────────────────────────────────────────────


class FinetuneRequest(BaseModel):
    """Request to start a LoRA fine-tuning job."""

    name: str = Field(default="", description="Job name for identification")
    dataset_id: str = Field(default="", description="Dataset ID to train on")
    weights_path: str = Field(
        default="/home/rubencr/.molfun/weights/openfold/finetuning_ptm_2.pt",
        description="Path to pretrained OpenFold weights",
    )
    head: str = Field(default="structure", description="Prediction head type")
    device: str = Field(default="cuda", description="Compute device")
    pdb_paths: list[str] = Field(default_factory=list, description="Explicit PDB file paths (alternative to dataset_id)")
    max_seq_len: int = Field(default=128, ge=32, le=2048, description="Maximum sequence length (lower = less GPU memory)")
    epochs: int = Field(default=20, ge=1, le=500, description="Number of training epochs")
    learning_rate: float = Field(default=5e-5, gt=0, description="Base learning rate")
    lr_lora: float | None = Field(default=None, description="LoRA-specific learning rate")
    lr_head: float | None = Field(default=None, description="Head-specific learning rate")
    lora_rank: int = Field(default=8, ge=2, le=64, description="LoRA low-rank dimension")
    lora_alpha: float = Field(default=16.0, gt=0, description="LoRA scaling factor")
    lora_dropout: float = Field(default=0.0, ge=0.0, le=0.5, description="LoRA dropout rate")
    target_modules: list[str] = Field(default=["linear_q", "linear_v"], description="Module names to inject LoRA into")
    scheduler: str = Field(default="cosine", description="LR scheduler: cosine, linear, constant")
    warmup_steps: int = Field(default=100, ge=0, description="Linear warmup steps")
    accumulation_steps: int = Field(default=4, ge=1, description="Gradient accumulation steps")
    amp: bool = Field(default=False, description="Enable automatic mixed precision (may cause NaN with gradient checkpointing)")
    early_stopping_patience: int = Field(default=5, ge=0, description="Early stopping patience (0=disabled)")
    grad_clip: float = Field(default=1.0, ge=0, description="Gradient clipping max norm")
    checkpoint_dir: str = Field(default="./checkpoints", description="Directory to save checkpoints")
    msa_backend: str = Field(default="single", description="MSA backend: single, precomputed, colabfold")


class FinetuneJobResponse(BaseModel):
    """A single fine-tuning job."""

    job_id: str
    name: str = ""
    dataset_id: str = ""
    status: str = "pending"
    progress: float = 0.0
    epoch: int = 0
    total_epochs: int = 0
    train_loss: float = 0.0
    val_loss: float | None = None
    lora_rank: int = 8
    config: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    created_at: str = ""


class FinetuneJobListResponse(BaseModel):
    """List of fine-tuning jobs."""

    jobs: list[FinetuneJobResponse]
    total: int


# ── Prediction ────────────────────────────────────────────────────────────────


class PredictRequest(BaseModel):
    """Request to predict a protein structure from sequence."""

    sequence: str = Field(min_length=1, max_length=2048, description="Amino acid sequence")
    adapter_path: str | None = Field(default=None, description="Path to LoRA adapter checkpoint")
    weights_path: str = Field(
        default="/home/rubencr/.molfun/weights/openfold/finetuning_ptm_2.pt",
        description="Path to pretrained OpenFold weights",
    )
    device: str = Field(default="cuda", description="Compute device")


class PredictResponse(BaseModel):
    """Structure prediction result."""

    sequence_length: int
    confidence: list[float] | None = None
    mean_plddt: float | None = None
    pdb_string: str | None = Field(default=None, description="Predicted PDB file content")


# ── MSA ───────────────────────────────────────────────────────────────────────


class MsaRequest(BaseModel):
    """Request to compute a Multiple Sequence Alignment."""

    sequence: str = Field(min_length=1, max_length=2048, description="Query amino acid sequence")
    pdb_id: str = Field(default="query", description="Identifier for caching")
    backend: str = Field(default="single", description="MSA backend: single, precomputed, colabfold")
    msa_dir: str | None = Field(default=None, description="Directory for precomputed .a3m files")


class MsaResponse(BaseModel):
    """MSA computation result."""

    num_sequences: int
    sequence_length: int
    backend: str
