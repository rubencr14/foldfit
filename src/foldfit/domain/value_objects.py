"""Domain value objects: immutable configuration models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class LoraConfig(BaseModel, frozen=True):
    """LoRA adapter configuration.

    Controls rank, scaling, dropout, and which linear layers
    inside the model are replaced with low-rank adapters.
    """

    rank: int = Field(default=8, ge=2, le=64, description="Low-rank dimension")
    alpha: float = Field(default=16.0, gt=0, description="Scaling factor (scaling = alpha / rank)")
    dropout: float = Field(default=0.0, ge=0.0, le=0.5, description="Dropout on LoRA input")
    target_modules: list[str] = Field(
        default=["linear_q", "linear_v"],
        description="Substrings to match against module names for LoRA injection",
    )


class QLoraConfig(LoraConfig, frozen=True):
    """QLoRA: LoRA combined with weight quantization.

    Quantizes the frozen base weights to 4-bit or 8-bit before
    applying LoRA adapters, reducing memory footprint.
    """

    quantization_bits: Literal[4, 8] = Field(default=4, description="Quantization bit width")
    double_quantization: bool = Field(
        default=True, description="Apply double quantization for extra compression"
    )
    quantization_type: str = Field(default="nf4", description="Quantization data type")


class TrainingConfig(BaseModel, frozen=True):
    """Training hyperparameters for fine-tuning."""

    epochs: int = Field(default=20, ge=1, le=500)
    learning_rate: float = Field(default=1e-4, gt=0)
    lr_lora: float | None = Field(default=None, description="LoRA-specific LR (defaults to lr)")
    lr_head: float | None = Field(default=None, description="Head-specific LR (defaults to lr*10)")
    batch_size: int = Field(default=1, ge=1)
    weight_decay: float = Field(default=0.01, ge=0.0)
    warmup_steps: int = Field(default=0, ge=0)
    scheduler: Literal["cosine", "linear", "constant"] = "cosine"
    min_lr: float = Field(default=1e-6, ge=0)
    grad_clip: float = Field(default=1.0, ge=0)
    accumulation_steps: int = Field(default=4, ge=1)
    amp: bool = Field(default=True, description="Automatic mixed precision")
    early_stopping_patience: int = Field(default=0, ge=0, description="0 = disabled")
    ema_decay: float = Field(default=0.0, ge=0.0, lt=1.0, description="0 = disabled")
    gradient_checkpointing: bool = False


class DataConfig(BaseModel, frozen=True):
    """Dataset configuration for antibody structures."""

    sabdab_dir: str | None = Field(default=None, description="Local SAbDab cache directory")
    pdb_paths: list[str] = Field(default_factory=list, description="Explicit PDB file paths")
    max_seq_len: int = Field(default=256, ge=32, le=2048)
    max_structures: int = Field(default=200, ge=1)
    val_frac: float = Field(default=0.1, ge=0.0, lt=1.0)
    test_frac: float = Field(default=0.1, ge=0.0, lt=1.0)
    split_seed: int = 42
    resolution_max: float = Field(default=3.0, gt=0, description="Max PDB resolution in angstroms")


class MsaConfig(BaseModel, frozen=True):
    """MSA computation configuration.

    Backends:
        single: Dummy MSA with query sequence only (no external deps).
        precomputed: Load .a3m files from msa_dir.
        colabfold: Query ColabFold MMseqs2 server (public or self-hosted).
        local: Run MMseqs2/HHblits locally against custom databases (e.g. OAS).
    """

    backend: Literal["precomputed", "colabfold", "local", "single"] = "single"
    msa_dir: str | None = Field(default=None, description="Directory for precomputed .a3m files")
    max_msa_depth: int = Field(default=512, ge=1)

    # ColabFold server (public API or self-hosted)
    colabfold_server: str = Field(
        default="https://api.colabfold.com",
        description="ColabFold server URL. Use your own for no rate limits.",
    )

    # Local MSA tool config
    tool: Literal["mmseqs2", "hhblits", "jackhmmer"] = Field(
        default="mmseqs2",
        description="Local alignment tool (used when backend=local).",
    )
    tool_binary: str | None = Field(
        default=None,
        description="Path to tool binary. Auto-detected from PATH if null.",
    )
    database_paths: list[str] = Field(
        default_factory=list,
        description="Paths to sequence databases (e.g. OAS, UniRef30). Searched in order.",
    )
    n_cpu: int = Field(default=4, ge=1, description="CPUs for local alignment.")


class ModelConfig(BaseModel, frozen=True):
    """Model configuration."""

    weights_path: str | None = Field(default=None, description="Path to pretrained weights")
    head: Literal["structure", "affinity"] = "structure"
    head_config: dict = Field(default_factory=dict)
    device: str = "cuda"


class OutputConfig(BaseModel, frozen=True):
    """Output configuration."""

    checkpoint_dir: str = "./checkpoints"


class FoldfitConfig(BaseModel, frozen=True):
    """Top-level configuration aggregating all sub-configs."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    lora: LoraConfig = Field(default_factory=LoraConfig)
    msa: MsaConfig = Field(default_factory=MsaConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
