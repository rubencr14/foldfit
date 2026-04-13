"""Applies LoRA adapters to an existing model by traversing its module tree."""

import logging
from collections.abc import Iterator

import torch.nn as nn

from finetuning.lora.config import LoRAConfig
from finetuning.lora.layers import LoRALinear

logger = logging.getLogger(__name__)


class LoRAApplicator:
    """Instruments an existing model with LoRA adapters.

    Traverses the module tree and replaces matching Linear layers with
    LoRALinear wrappers. This follows the Open/Closed Principle: the
    original model code is not modified; behavior is extended via
    composition.

    Args:
        config: LoRA configuration specifying rank, alpha, targets, etc.
    """

    def __init__(self, config: LoRAConfig):
        self.config = config

    def apply(self, model: nn.Module) -> int:
        """Walk the module tree and replace target Linear layers with LoRALinear.

        Args:
            model: The model to instrument. Modified in-place.

        Returns:
            The number of layers that were adapted.
        """
        adapted_count = 0
        replacements: list[tuple[nn.Module, str, nn.Module]] = []

        for full_name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if not self._should_adapt(full_name):
                continue

            lora_layer = LoRALinear(
                original=module,
                rank=self.config.rank,
                alpha=self.config.alpha,
                dropout=self.config.dropout,
            )
            # Find the parent module and attribute name
            parent, attr_name = self._get_parent_and_attr(model, full_name)
            replacements.append((parent, attr_name, lora_layer))
            adapted_count += 1

        # Apply replacements after traversal to avoid modifying the tree
        # during iteration
        for parent, attr_name, lora_layer in replacements:
            setattr(parent, attr_name, lora_layer)
            logger.debug(f"Applied LoRA to: {attr_name} in {type(parent).__name__}")

        logger.info(f"Applied LoRA to {adapted_count} layers")
        return adapted_count

    def _should_adapt(self, full_name: str) -> bool:
        """Check if a module matches target_blocks AND target_modules criteria."""
        # Check that the module is inside one of the target blocks
        in_target_block = any(
            block in full_name for block in self.config.target_blocks
        )
        if not in_target_block:
            return False

        # Check that the leaf attribute name matches a target module
        leaf_name = full_name.rsplit(".", maxsplit=1)[-1]
        return leaf_name in self.config.target_modules

    @staticmethod
    def _get_parent_and_attr(
        model: nn.Module, full_name: str
    ) -> tuple[nn.Module, str]:
        """Resolve a dotted name to (parent_module, attribute_name)."""
        parts = full_name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        return parent, parts[-1]

    @staticmethod
    def get_lora_parameters(model: nn.Module) -> Iterator[nn.Parameter]:
        """Yield only LoRA parameters (lora_A, lora_B) from the model."""
        for name, param in model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                yield param

    @staticmethod
    def freeze_base_parameters(model: nn.Module) -> None:
        """Freeze all non-LoRA parameters in the model."""
        for name, param in model.named_parameters():
            if "lora_A" not in name and "lora_B" not in name:
                param.requires_grad = False

    @staticmethod
    def count_parameters(model: nn.Module) -> dict[str, int]:
        """Return counts of total, trainable, and LoRA parameters.

        Returns:
            Dictionary with keys 'total', 'trainable', and 'lora'.
        """
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        lora = sum(
            p.numel()
            for n, p in model.named_parameters()
            if "lora_A" in n or "lora_B" in n
        )
        return {"total": total, "trainable": trainable, "lora": lora}
