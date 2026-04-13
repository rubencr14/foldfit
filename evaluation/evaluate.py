"""Evaluation entry point for antibody-fine-tuned models."""

import logging
from pathlib import Path

import torch

from finetuning.evaluation.metrics import AntibodyMetrics
from finetuning.lora.applicator import LoRAApplicator
from finetuning.lora.checkpoint import LoRACheckpointManager
from finetuning.lora.config import LoRAConfig

logger = logging.getLogger(__name__)


def load_model_with_lora(
    model_config,
    lora_config: LoRAConfig,
    base_checkpoint_path: Path,
    lora_checkpoint_path: Path,
    device: str = "cpu",
):
    """Load an OpenFold3 model with LoRA weights for evaluation.

    Args:
        model_config: ml_collections ConfigDict for OpenFold3.
        lora_config: LoRA adapter configuration.
        base_checkpoint_path: Path to pretrained OpenFold3 checkpoint.
        lora_checkpoint_path: Path to LoRA-only checkpoint.
        device: Device to load the model on.

    Returns:
        The model with LoRA weights loaded, in eval mode.
    """
    from openfold3.core.utils.checkpoint_loading_utils import (
        get_state_dict_from_checkpoint,
    )
    from openfold3.projects.of3_all_atom.model import OpenFold3

    # Build and load base model
    model = OpenFold3(model_config)
    checkpoint = torch.load(base_checkpoint_path, map_location="cpu", weights_only=False)
    state_dict, _ = get_state_dict_from_checkpoint(checkpoint)
    cleaned = {k.removeprefix("model."): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned, strict=False)

    # Apply LoRA and load LoRA weights
    applicator = LoRAApplicator(lora_config)
    applicator.apply(model)
    LoRACheckpointManager.load_lora_weights(model, lora_checkpoint_path)

    model = model.to(device)
    model.eval()
    logger.info("Model loaded with LoRA weights for evaluation")
    return model


def evaluate_predictions(
    pred_positions: torch.Tensor,
    gt_positions: torch.Tensor,
    cdr_masks: dict[str, torch.Tensor] | None = None,
    heavy_mask: torch.Tensor | None = None,
    light_mask: torch.Tensor | None = None,
) -> dict[str, float]:
    """Compute evaluation metrics for a single prediction.

    Args:
        pred_positions: Predicted atom positions, shape [N, 3].
        gt_positions: Ground truth positions, shape [N, 3].
        cdr_masks: Optional CDR region masks.
        heavy_mask: Optional heavy chain mask.
        light_mask: Optional light chain mask.

    Returns:
        Dictionary of metric names to values.
    """
    metrics_calc = AntibodyMetrics()
    results: dict[str, float] = {}

    # Global RMSD
    results["global_rmsd"] = metrics_calc.rmsd(pred_positions, gt_positions).item()

    # Global dRMSD
    results["global_drmsd"] = metrics_calc.drmsd(pred_positions, gt_positions).item()

    # CDR-specific metrics
    if cdr_masks:
        cdr_rmsds = metrics_calc.cdr_rmsd(pred_positions, gt_positions, cdr_masks)
        for cdr_name, value in cdr_rmsds.items():
            results[f"rmsd_{cdr_name}"] = value.item()

        # Framework RMSD
        results["framework_rmsd"] = metrics_calc.framework_rmsd(
            pred_positions, gt_positions, cdr_masks
        ).item()

    # Interface contacts
    if heavy_mask is not None and light_mask is not None:
        results["hl_contacts_pred"] = metrics_calc.interface_contacts(
            pred_positions, heavy_mask, light_mask
        )
        results["hl_contacts_gt"] = metrics_calc.interface_contacts(
            gt_positions, heavy_mask, light_mask
        )

    return results
