"""LoRA injector: scans a model and replaces target nn.Linear layers with LoRALinear."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from foldfit.domain.interfaces import PeftPort
from foldfit.domain.value_objects import LoraConfig
from foldfit.infrastructure.peft.lora_linear import LoRALinear

logger = logging.getLogger(__name__)


class LoraInjector(PeftPort):
    """Injects LoRA adapters into a model by replacing target nn.Linear modules.

    Target modules are identified by substring matching against the full
    dotted module name (e.g., 'linear_q' matches 'evoformer.blocks.0.msa_att.linear_q').
    """

    def __init__(self) -> None:
        self._replaced_layers: list[tuple[nn.Module, str, LoRALinear]] = []
        self._config: LoraConfig | None = None

    def apply(self, module: nn.Module, config: LoraConfig) -> None:
        """Inject LoRA adapters into all matching linear layers.

        Args:
            module: The nn.Module to inject into (e.g., model.evoformer).
            config: LoRA configuration with rank, alpha, dropout, target_modules.
        """
        self._config = config
        self._replaced_layers.clear()

        # Freeze all parameters in the target module
        for param in module.parameters():
            param.requires_grad = False

        # Collect replacements (cannot modify during iteration)
        replacements: list[tuple[nn.Module, str, LoRALinear]] = []

        for full_name, child in module.named_modules():
            if not isinstance(child, nn.Linear):
                continue

            if not any(target in full_name for target in config.target_modules):
                continue

            lora_layer = LoRALinear(
                original=child,
                rank=config.rank,
                alpha=config.alpha,
                dropout=config.dropout,
            )

            # Find parent module and attribute name
            parts = full_name.rsplit(".", 1)
            if len(parts) == 1:
                parent = module
                attr_name = parts[0]
            else:
                parent = module.get_submodule(parts[0])
                attr_name = parts[1]

            replacements.append((parent, attr_name, lora_layer))

        # Apply replacements
        for parent, attr_name, lora_layer in replacements:
            parent._modules[attr_name] = lora_layer
            self._replaced_layers.append((parent, attr_name, lora_layer))

        count = len(self._replaced_layers)
        trainable = sum(p.numel() for p in self.trainable_parameters())
        logger.info(f"Injected LoRA into {count} layers ({trainable:,} trainable params)")

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Return all trainable LoRA parameters (lora_A and lora_B)."""
        params: list[nn.Parameter] = []
        for _, _, lora_layer in self._replaced_layers:
            params.append(lora_layer.lora_A)
            params.append(lora_layer.lora_B)
        return params

    def merge(self) -> None:
        """Merge all LoRA weights into base weights."""
        for _, _, lora_layer in self._replaced_layers:
            lora_layer.merge()

    def unmerge(self) -> None:
        """Unmerge all LoRA weights from base weights."""
        for _, _, lora_layer in self._replaced_layers:
            lora_layer.unmerge()

    def save(self, path: str | Path) -> None:
        """Save LoRA adapter weights to disk.

        Saves only the lora_A and lora_B parameters, not the frozen base weights.
        """
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        state: dict[str, Any] = {
            "config": self._config.model_dump() if self._config else {},
            "adapters": {},
        }

        for parent, attr_name, lora_layer in self._replaced_layers:
            key = f"{id(parent)}.{attr_name}"
            state["adapters"][key] = {
                "lora_A": lora_layer.lora_A.data.cpu(),
                "lora_B": lora_layer.lora_B.data.cpu(),
            }

        torch.save(state, save_path / "lora_adapter.pt")
        logger.info(f"Saved LoRA adapter to {save_path / 'lora_adapter.pt'}")

    def load(self, path: str | Path) -> None:
        """Load LoRA adapter weights from disk.

        Must be called after apply() so the layer structure is known.
        """
        load_path = Path(path) / "lora_adapter.pt"
        state = torch.load(load_path, weights_only=True)

        adapter_values = list(state["adapters"].values())
        if len(adapter_values) != len(self._replaced_layers):
            raise ValueError(
                f"Adapter count mismatch: file has {len(adapter_values)}, "
                f"model has {len(self._replaced_layers)}"
            )

        for (_, _, lora_layer), adapter_state in zip(
            self._replaced_layers, adapter_values, strict=True
        ):
            device = lora_layer.lora_A.device
            lora_layer.lora_A.data = adapter_state["lora_A"].to(device)
            lora_layer.lora_B.data = adapter_state["lora_B"].to(device)

        logger.info(f"Loaded LoRA adapter from {load_path}")

    @property
    def replaced_count(self) -> int:
        """Number of layers replaced with LoRA."""
        return len(self._replaced_layers)
