"""Checkpoint store for saving/loading fine-tuned model artifacts."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from foldfit.domain.interfaces import CheckpointPort, PeftPort

logger = logging.getLogger(__name__)


class FileCheckpointStore(CheckpointPort):
    """File-based checkpoint store.

    Saves LoRA adapters, head weights, and training state to disk.
    Layout:
        checkpoint_dir/
            peft/lora_adapter.pt    - LoRA A/B weights
            head.pt                 - Head module state dict
            training_state.pt       - Optimizer, scheduler, scaler state
            meta.pt                 - Config metadata
    """

    def save(
        self,
        path: str | Path,
        peft: PeftPort,
        head: nn.Module | None,
        training_state: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Save checkpoint to disk.

        Args:
            path: Directory to save checkpoint.
            peft: PEFT port with adapter weights.
            head: Optional head module.
            training_state: Dict with optimizer/scheduler/scaler states.
            metadata: Optional metadata (config, epoch, etc.).
        """
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save PEFT adapter
        peft.save(save_path / "peft")

        # Save head
        if head is not None:
            torch.save(head.state_dict(), save_path / "head.pt")

        # Save training state
        if training_state:
            torch.save(training_state, save_path / "training_state.pt")

        # Save metadata
        if metadata:
            torch.save(metadata, save_path / "meta.pt")

        logger.info(f"Checkpoint saved to {save_path}")

    def load(self, path: str | Path) -> dict[str, Any]:
        """Load checkpoint from disk.

        Args:
            path: Directory containing the checkpoint.

        Returns:
            Dict with keys: 'peft_path', 'head_state', 'training_state', 'metadata'.
        """
        load_path = Path(path)
        result: dict[str, Any] = {}

        peft_path = load_path / "peft"
        if peft_path.exists():
            result["peft_path"] = str(peft_path)

        head_path = load_path / "head.pt"
        if head_path.exists():
            result["head_state"] = torch.load(head_path, weights_only=True)

        state_path = load_path / "training_state.pt"
        if state_path.exists():
            result["training_state"] = torch.load(state_path, weights_only=True)

        meta_path = load_path / "meta.pt"
        if meta_path.exists():
            result["metadata"] = torch.load(meta_path, weights_only=True)

        logger.info(f"Checkpoint loaded from {load_path}")
        return result
