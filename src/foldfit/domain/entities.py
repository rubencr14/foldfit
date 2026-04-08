"""Domain entities representing core business objects."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import torch
from pydantic import BaseModel, Field

from foldfit.domain.value_objects import FoldfitConfig, LoraConfig


@dataclass
class TrunkOutput:
    """Output from a structure prediction model trunk.

    Attributes:
        single_repr: Per-residue representation [B, L, D].
        pair_repr: Pairwise representation [B, L, L, D] or None.
        structure_coords: Predicted atom coordinates [B, L, 37, 3] or None.
        confidence: Per-residue confidence (pLDDT) [B, L] or None.
        extra: Additional model-specific outputs.
    """

    single_repr: torch.Tensor
    pair_repr: torch.Tensor | None = None
    structure_coords: torch.Tensor | None = None
    confidence: torch.Tensor | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class FinetuneJob(BaseModel):
    """Represents a fine-tuning run."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: str = "pending"
    config: FoldfitConfig = Field(default_factory=FoldfitConfig)
    metrics: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class TrainedModel(BaseModel):
    """Represents a fine-tuned model artifact."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    lora_config: LoraConfig = Field(default_factory=LoraConfig)
    checkpoint_path: str = ""
    metrics: dict[str, Any] = Field(default_factory=dict)
