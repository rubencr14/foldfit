"""Inference service: load a model with optional LoRA adapters and predict."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from foldfit.domain.entities import TrunkOutput
from foldfit.domain.interfaces import CheckpointPort, ModelPort, PeftPort
from foldfit.domain.value_objects import LoraConfig, ModelConfig

logger = logging.getLogger(__name__)


class InferenceService:
    """Runs inference with optional LoRA adapter loading.

    Args:
        model: Model port for structure prediction.
        peft: PEFT port for adapter loading.
        checkpoint: Checkpoint port for loading saved artifacts.
    """

    def __init__(
        self,
        model: ModelPort,
        peft: PeftPort,
        checkpoint: CheckpointPort,
    ) -> None:
        self._model = model
        self._peft = peft
        self._checkpoint = checkpoint
        self._loaded = False

    def load(
        self,
        model_config: ModelConfig,
        adapter_path: str | None = None,
        lora_config: LoraConfig | None = None,
        merge_adapter: bool = True,
    ) -> None:
        """Load model and optionally apply a LoRA adapter."""
        self._model.load(
            weights_path=model_config.weights_path,
            device=model_config.device,
        )

        if adapter_path is not None:
            if lora_config is None:
                ckpt = self._checkpoint.load(adapter_path)
                meta = ckpt.get("metadata", {})
                config_dict = meta.get("config", {}).get("lora", {})
                lora_config = LoraConfig(**config_dict) if config_dict else LoraConfig()

            target_module = self._model.get_peft_target_module()
            self._peft.apply(target_module, lora_config)

            peft_path = Path(adapter_path) / "peft"
            if peft_path.exists():
                self._peft.load(peft_path)
            else:
                self._peft.load(adapter_path)

            if merge_adapter:
                self._peft.merge()
                logger.info("LoRA adapter merged into base weights")

        self._model.train_mode(False)
        self._loaded = True

    @torch.no_grad()
    def predict(self, batch: dict[str, torch.Tensor]) -> TrunkOutput:
        """Run inference on a pre-built feature batch."""
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self._model.forward(batch)

    @torch.no_grad()
    def predict_from_sequence(
        self,
        sequence: str,
        device: str = "cuda",
        max_seq_len: int = 256,
    ) -> dict[str, Any]:
        """Predict structure from a raw amino acid sequence.

        Args:
            sequence: Amino acid sequence string.
            device: Compute device.
            max_seq_len: Maximum sequence length.

        Returns:
            Dict with 'pdb_string', 'confidence', 'mean_plddt', 'sequence_length'.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        from foldfit.infrastructure.openfold.featurizer import OpenFoldFeaturizer
        from foldfit.infrastructure.openfold.pdb_writer import coords_to_pdb

        # Featurize
        featurizer = OpenFoldFeaturizer(max_seq_len=max_seq_len)
        features = featurizer.from_sequence(sequence)

        if not features:
            raise ValueError(f"Failed to featurize sequence of length {len(sequence)}")

        # Add batch dimension and move to device
        batch = {
            k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v
            for k, v in features.items()
        }

        # Forward pass
        output = self._model.forward(batch)

        # Extract results
        coords = output.structure_coords
        confidence = output.confidence
        seq_len = min(len(sequence), max_seq_len)

        result: dict[str, Any] = {"sequence_length": seq_len}

        if coords is not None:
            coords_np = coords[0, :seq_len].cpu().numpy()
            plddt_np = confidence[0, :seq_len].cpu().numpy() if confidence is not None else None
            result["pdb_string"] = coords_to_pdb(
                sequence[:seq_len], coords_np, plddt_np
            )
            result["confidence"] = plddt_np.tolist() if plddt_np is not None else None
            result["mean_plddt"] = float(plddt_np.mean()) if plddt_np is not None else None
        else:
            result["pdb_string"] = None
            result["confidence"] = None
            result["mean_plddt"] = None

        return result
