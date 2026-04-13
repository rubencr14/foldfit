"""Exponential Moving Average that only tracks LoRA parameters."""

import copy
import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class LoRAExponentialMovingAverage:
    """EMA that only tracks LoRA parameters (lora_A, lora_B).

    Compared to tracking all model parameters, this uses minimal memory
    since LoRA parameters are a small fraction of the total.

    Args:
        model: Model with LoRA layers applied.
        decay: EMA decay factor. Closer to 1.0 means slower updates.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {}
        self.backup: dict[str, torch.Tensor] = {}

        for name, param in model.named_parameters():
            if self._is_lora_param(name) and param.requires_grad:
                self.shadow[name] = param.data.clone()

        logger.info(f"LoRA EMA initialized with {len(self.shadow)} parameters")

    @staticmethod
    def _is_lora_param(name: str) -> bool:
        return "lora_A" in name or "lora_B" in name

    def update(self, model: nn.Module) -> None:
        """Update the shadow parameters with the current model parameters."""
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].lerp_(param.data, 1.0 - self.decay)

    def apply_shadow(self, model: nn.Module) -> None:
        """Replace LoRA parameters with their EMA values.

        Call restore() afterwards to undo this.
        """
        self.backup.clear()
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module) -> None:
        """Restore LoRA parameters from the backup created by apply_shadow()."""
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup.clear()

    def state_dict(self) -> dict:
        """Serialize the EMA state for checkpointing."""
        return {
            "decay": self.decay,
            "shadow": copy.deepcopy(self.shadow),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Load EMA state from a checkpoint."""
        self.decay = state_dict["decay"]
        self.shadow = state_dict["shadow"]

    def to(self, device: torch.device) -> "LoRAExponentialMovingAverage":
        """Move shadow parameters to the specified device."""
        self.shadow = {k: v.to(device) for k, v in self.shadow.items()}
        return self
