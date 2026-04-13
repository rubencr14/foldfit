"""Unified configuration schema for LoRA fine-tuning."""

from dataclasses import asdict
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator

from finetuning.lora.config import LoRAConfig


class TrainingConfig(BaseModel):
    """Training hyperparameters for LoRA fine-tuning."""

    learning_rate: float = Field(1e-4, gt=0, description="Peak learning rate")
    weight_decay: float = Field(0.01, ge=0, description="Weight decay for AdamW")
    max_epochs: int = Field(20, gt=0, description="Maximum training epochs")
    warmup_steps: int = Field(200, ge=0, description="LR warmup steps")
    batch_size: int = Field(1, gt=0, description="Training batch size")
    ema_decay: float = Field(0.999, gt=0, lt=1, description="EMA decay factor")
    gradient_clip_val: float = Field(1.0, gt=0, description="Gradient clipping value")
    scheduler: str = Field("cosine", description="LR scheduler type: 'cosine' or 'linear'")
    num_workers: int = Field(4, ge=0, description="DataLoader workers")
    precision: str = Field("bf16-mixed", description="Training precision")

    @field_validator("scheduler")
    @classmethod
    def validate_scheduler(cls, v: str) -> str:
        if v not in ("cosine", "linear"):
            raise ValueError(f"scheduler must be 'cosine' or 'linear', got '{v}'")
        return v


class DatasetConfig(BaseModel):
    """Dataset configuration for antibody fine-tuning."""

    pdb_ids: list[str] | None = Field(
        None, description="Explicit PDB IDs. If None, fetches from RCSB."
    )
    pdb_ids_file: str | None = Field(
        None, description="Path to a text file with one PDB ID per line."
    )
    search_max_results: int = Field(
        1000, gt=0, description="Max results when searching RCSB"
    )
    max_resolution: float = Field(
        3.0, gt=0, description="Maximum resolution cutoff (Angstroms)"
    )
    require_heavy_chain: bool = Field(
        True, description="Require heavy chain in structures"
    )
    require_light_chain: bool = Field(
        True, description="Require light chain in structures"
    )
    cdr_scheme: str = Field("imgt", description="CDR numbering scheme")
    cache_dir: Path = Field(
        Path("./data/antibody_cache"), description="Cache directory for downloads"
    )
    train_split: float = Field(0.9, gt=0, lt=1, description="Train/val split ratio")
    token_budget: int = Field(640, gt=0, description="Token budget for cropping")

    @field_validator("cdr_scheme")
    @classmethod
    def validate_cdr_scheme(cls, v: str) -> str:
        valid = {"imgt", "chothia", "kabat"}
        if v.lower() not in valid:
            raise ValueError(f"cdr_scheme must be one of {valid}, got '{v}'")
        return v.lower()


class FinetuneConfig(BaseModel):
    """Top-level configuration combining all sub-configs."""

    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    pretrained_checkpoint: Path | None = Field(
        None, description="Path to pretrained OpenFold3 checkpoint"
    )
    output_dir: Path = Field(
        Path("./output/antibody_lora"), description="Output directory"
    )
    seed: int = Field(42, description="Random seed")

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_yaml(cls, path: str | Path) -> "FinetuneConfig":
        """Load configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            Parsed FinetuneConfig instance.
        """
        with open(path) as f:
            data = yaml.safe_load(f)

        # Convert lora dict to LoRAConfig dataclass
        if "lora" in data and isinstance(data["lora"], dict):
            data["lora"] = LoRAConfig(**data["lora"])

        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file."""
        data = self.model_dump()
        # Convert LoRAConfig dataclass to dict
        if isinstance(data.get("lora"), LoRAConfig):
            data["lora"] = asdict(data["lora"])
        # Convert Path objects to strings
        for key in ("pretrained_checkpoint", "output_dir"):
            if data.get(key) is not None:
                data[key] = str(data[key])
        if "dataset" in data and data["dataset"].get("cache_dir"):
            data["dataset"]["cache_dir"] = str(data["dataset"]["cache_dir"])

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
