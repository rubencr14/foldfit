"""Domain interfaces (ports) defining contracts between layers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch.nn as nn
from torch import Tensor

from foldfit.domain.entities import TrunkOutput
from foldfit.domain.value_objects import DataConfig, LoraConfig


class ModelPort(ABC):
    """Port for structure prediction model access."""

    @abstractmethod
    def load(self, weights_path: str | None = None, device: str = "cuda") -> None:
        """Load model weights onto the specified device."""

    @abstractmethod
    def forward(self, batch: dict[str, Tensor]) -> TrunkOutput:
        """Run forward pass and return trunk output."""

    @abstractmethod
    def freeze_trunk(self) -> None:
        """Freeze all trunk parameters (requires_grad = False)."""

    @abstractmethod
    def unfreeze_trunk(self) -> None:
        """Unfreeze all trunk parameters."""

    @abstractmethod
    def get_peft_target_module(self) -> nn.Module:
        """Return the submodule where PEFT adapters should be injected."""

    @abstractmethod
    def get_default_peft_targets(self) -> list[str]:
        """Return default module name substrings for PEFT targeting."""

    @abstractmethod
    def train_mode(self, mode: bool) -> None:
        """Set training mode with model-specific constraints."""

    @abstractmethod
    def get_model(self) -> nn.Module:
        """Return the underlying nn.Module."""


class PeftPort(ABC):
    """Port for parameter-efficient fine-tuning (LoRA/QLoRA)."""

    @abstractmethod
    def apply(self, module: nn.Module, config: LoraConfig) -> None:
        """Inject PEFT adapters into the target module."""

    @abstractmethod
    def merge(self) -> None:
        """Merge adapter weights into base weights for inference."""

    @abstractmethod
    def unmerge(self) -> None:
        """Unmerge adapter weights from base weights."""

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Save adapter weights to disk."""

    @abstractmethod
    def load(self, path: str | Path) -> None:
        """Load adapter weights from disk."""

    @abstractmethod
    def trainable_parameters(self) -> list[nn.Parameter]:
        """Return list of trainable LoRA parameters."""


class DatasetPort(ABC):
    """Port for dataset access."""

    @abstractmethod
    def fetch_structures(self, config: DataConfig) -> list[Path]:
        """Fetch antibody structure file paths."""


class MsaPort(ABC):
    """Port for MSA computation."""

    @abstractmethod
    def get(self, sequence: str, pdb_id: str) -> dict[str, Any]:
        """Compute or retrieve MSA for a sequence.

        Returns dict with keys: msa, deletion_matrix, msa_mask.
        """


class CheckpointPort(ABC):
    """Port for saving/loading training checkpoints."""

    @abstractmethod
    def save(
        self,
        path: str | Path,
        peft: PeftPort,
        head: nn.Module | None,
        training_state: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Save PEFT adapter, head weights, and training state."""

    @abstractmethod
    def load(self, path: str | Path) -> dict[str, Any]:
        """Load checkpoint and return components dict."""
