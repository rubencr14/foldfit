"""Save, load, and merge LoRA-only weights."""

import logging
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn

from finetuning.lora.config import LoRAConfig
from finetuning.lora.layers import LoRALinear

logger = logging.getLogger(__name__)


class LoRACheckpointManager:
    """Handles saving, loading, and merging LoRA-only weights.

    LoRA checkpoints are small because they only contain lora_A and lora_B
    parameters, not the full base model weights.
    """

    @staticmethod
    def extract_lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
        """Extract only LoRA parameters from the model's state dict.

        Returns:
            Dictionary mapping parameter names to tensors for all
            lora_A and lora_B parameters.
        """
        return {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if "lora_A" in name or "lora_B" in name
        }

    @staticmethod
    def save_lora_weights(
        model: nn.Module,
        path: Path,
        config: LoRAConfig | None = None,
    ) -> None:
        """Save only LoRA parameters and optionally the config.

        Args:
            model: Model with LoRA layers applied.
            path: File path to save the checkpoint.
            config: Optional LoRA config to include in the checkpoint
                for reproducibility.
        """
        lora_state = LoRACheckpointManager.extract_lora_state_dict(model)
        checkpoint = {"lora_state_dict": lora_state}
        if config is not None:
            checkpoint["lora_config"] = asdict(config)

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
        logger.info(f"Saved LoRA weights to {path} ({len(lora_state)} parameters)")

    @staticmethod
    def load_lora_weights(model: nn.Module, path: Path) -> LoRAConfig | None:
        """Load LoRA weights into a model that already has LoRA layers applied.

        Args:
            model: Model with LoRA layers already applied via LoRAApplicator.
            path: Path to a LoRA checkpoint file.

        Returns:
            The LoRAConfig stored in the checkpoint, if present.
        """
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        lora_state = checkpoint["lora_state_dict"]

        model_state = model.state_dict()
        missing = []
        for name, tensor in lora_state.items():
            if name in model_state:
                model_state[name].copy_(tensor)
            else:
                missing.append(name)

        if missing:
            logger.warning(
                f"LoRA checkpoint contains {len(missing)} parameters not found "
                f"in model: {missing[:5]}..."
            )

        model.load_state_dict(model_state, strict=False)
        logger.info(f"Loaded LoRA weights from {path}")

        config_dict = checkpoint.get("lora_config")
        if config_dict is not None:
            return LoRAConfig(**config_dict)
        return None

    @staticmethod
    def merge_lora_into_model(model: nn.Module) -> None:
        """Merge all LoRA weights into the base model in-place.

        After merging, LoRALinear modules are replaced with their
        original (now modified) Linear layers. The model produces
        identical output but without the LoRA computation overhead.

        Args:
            model: Model with LoRA layers applied.
        """
        replacements: list[tuple[nn.Module, str, nn.Module]] = []

        for full_name, module in model.named_modules():
            if isinstance(module, LoRALinear):
                merged_linear = module.merge()
                # Re-enable gradients on the merged layer
                for param in merged_linear.parameters():
                    param.requires_grad = True
                parts = full_name.split(".")
                parent = model
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                replacements.append((parent, parts[-1], merged_linear))

        for parent, attr_name, merged_linear in replacements:
            setattr(parent, attr_name, merged_linear)

        logger.info(f"Merged {len(replacements)} LoRA layers into base model")

    @staticmethod
    def merge_and_save(model: nn.Module, path: Path) -> None:
        """Merge LoRA into base weights and save the full model state dict.

        Args:
            model: Model with LoRA layers applied.
            path: File path to save the merged checkpoint.
        """
        LoRACheckpointManager.merge_lora_into_model(model)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), path)
        logger.info(f"Saved merged model to {path}")
