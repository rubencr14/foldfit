"""OpenFold model adapter implementing the ModelPort interface.

Wraps OpenFold's AlphaFold model behind a clean interface for LoRA fine-tuning.

CRITICAL: OpenFold's EvoformerStack requires eval() mode even during training
due to chunked operations. The adapter enforces this constraint.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from foldfit.domain.entities import TrunkOutput
from foldfit.domain.interfaces import ModelPort

try:
    from openfold.config import model_config as openfold_model_config
    from openfold.model.model import AlphaFold
    from openfold.utils.loss import compute_plddt

    HAS_OPENFOLD = True
except ImportError:
    HAS_OPENFOLD = False

logger = logging.getLogger(__name__)


def _require_openfold() -> None:
    if not HAS_OPENFOLD:
        raise ImportError(
            "openfold is required for OpenFoldAdapter. "
            "Install it from: https://github.com/aqlaboratory/openfold"
        )


class OpenFoldAdapter(ModelPort):
    """Adapter wrapping OpenFold's AlphaFold model.

    Args:
        model: Pre-built AlphaFold model, or None to load from weights.
        config: OpenFold model config (ml_collections.ConfigDict).
    """

    DEFAULT_PEFT_TARGETS = ["linear_q", "linear_v"]

    def __init__(
        self,
        model: nn.Module | None = None,
        config: Any = None,
    ) -> None:
        self._model: nn.Module | None = model
        self._config = config
        self._device = "cpu"

    @property
    def model(self) -> nn.Module:
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self._model

    def load(self, weights_path: str | None = None, device: str = "cuda") -> None:
        """Load AlphaFold model from weights.

        Args:
            weights_path: Path to pretrained weights (.pt file).
            device: Device to load the model onto.
        """
        self._device = device

        if self._model is None:
            self._model = self._build_model()

        if weights_path is not None:
            state = torch.load(weights_path, map_location="cpu", weights_only=True)
            self._model.load_state_dict(state, strict=False)
            logger.info(f"Loaded weights from {weights_path}")

        self._model.to(device)
        self._model.eval()
        logger.info(f"OpenFold model loaded on {device}")

    def _build_model(self) -> nn.Module:
        """Build AlphaFold model from config."""
        _require_openfold()

        if self._config is None:
            self._config = openfold_model_config("model_1")

        return AlphaFold(self._config)

    def forward(self, batch: dict[str, torch.Tensor]) -> TrunkOutput:
        """Run forward pass through the AlphaFold model.

        Args:
            batch: Feature dict with OpenFold-format tensors.

        Returns:
            TrunkOutput with representations and predicted structure.
        """
        model = self.model

        with torch.set_grad_enabled(torch.is_grad_enabled()):
            outputs = model(batch)

        # Extract representations
        single_repr = outputs.get("single", outputs.get("sm", {}).get("single"))
        pair_repr = outputs.get("pair")
        coords = outputs.get("final_atom_positions")

        # Compute pLDDT confidence
        confidence = outputs.get("plddt")
        if confidence is None and "sm" in outputs and "single" in outputs["sm"]:
            if HAS_OPENFOLD:
                confidence = compute_plddt(outputs["sm"]["single"])

        return TrunkOutput(
            single_repr=single_repr,
            pair_repr=pair_repr,
            structure_coords=coords,
            confidence=confidence,
            extra={"_raw_outputs": outputs},
        )

    def freeze_trunk(self) -> None:
        """Freeze all model parameters."""
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_trunk(self) -> None:
        """Unfreeze all model parameters."""
        for param in self.model.parameters():
            param.requires_grad = True

    def get_peft_target_module(self) -> nn.Module:
        """Return the Evoformer stack for LoRA injection."""
        return self.model.evoformer

    def get_default_peft_targets(self) -> list[str]:
        """Return default attention projection names for LoRA."""
        return self.DEFAULT_PEFT_TARGETS

    def train_mode(self, mode: bool) -> None:
        """Set training mode.

        CRITICAL: OpenFold must remain in eval() mode even during training
        because EvoformerStack's chunked operations require it.
        Gradients still flow through requires_grad=True parameters.
        """
        if mode:
            self.model.train()
            self.model.eval()  # Force eval for EvoformerStack
        else:
            self.model.eval()

    def get_model(self) -> nn.Module:
        """Return the underlying AlphaFold model."""
        return self.model

    def param_summary(self) -> dict[str, int]:
        """Return parameter counts."""
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}
