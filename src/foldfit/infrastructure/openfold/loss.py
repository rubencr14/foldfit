"""OpenFold loss for structure fine-tuning.

Wraps OpenFold's native loss functions (FAPE, distogram, masked MSA,
supervised chi, pLDDT, violations) with AlphaFold2 default weights.
"""

from __future__ import annotations

import logging
from typing import Any

import ml_collections
import torch
import torch.nn as nn

from openfold.utils.loss import (
    AlphaFoldLoss,
    backbone_loss,
    distogram_loss,
    lddt_loss,
    masked_msa_loss,
    supervised_chi_loss,
)

logger = logging.getLogger(__name__)


def _default_loss_config() -> ml_collections.ConfigDict:
    """AlphaFold2 default loss config with weights tuned for LoRA fine-tuning."""
    eps = 1e-8
    cfg = ml_collections.ConfigDict(
        {
            "distogram": {
                "min_bin": 2.3125,
                "max_bin": 21.6875,
                "no_bins": 64,
                "eps": eps,
                "weight": 0.3,
            },
            "experimentally_resolved": {
                "eps": eps,
                "min_resolution": 0.1,
                "max_resolution": 3.0,
                "weight": 0.0,
            },
            "fape": {
                "backbone": {
                    "clamp_distance": 10.0,
                    "loss_unit_distance": 10.0,
                    "weight": 0.5,
                },
                "sidechain": {
                    "clamp_distance": 10.0,
                    "length_scale": 10.0,
                    "weight": 0.5,
                },
                "eps": 1e-4,
                "weight": 1.0,
            },
            "plddt_loss": {
                "min_resolution": 0.1,
                "max_resolution": 3.0,
                "cutoff": 15.0,
                "no_bins": 50,
                "eps": eps,
                "weight": 0.01,
            },
            "masked_msa": {
                "num_classes": 23,
                "eps": eps,
                "weight": 2.0,
            },
            "supervised_chi": {
                "chi_weight": 0.5,
                "angle_norm_weight": 0.01,
                "eps": eps,
                "weight": 1.0,
            },
            "violation": {
                "violation_tolerance_factor": 12.0,
                "clash_overlap_tolerance": 1.5,
                "average_clashes": False,
                "eps": eps,
                "weight": 0.0,
            },
            "tm": {
                "max_bin": 31,
                "no_bins": 64,
                "min_resolution": 0.1,
                "max_resolution": 3.0,
                "eps": eps,
                "weight": 0.0,
                "enabled": False,
            },
            "chain_center_of_mass": {
                "clamp_distance": -4.0,
                "weight": 0.0,
                "enabled": False,
            },
            "eps": eps,
        }
    )
    return cfg


class OpenFoldLoss(nn.Module):
    """Wraps OpenFold's native AlphaFoldLoss for LoRA fine-tuning.

    Uses the full AlphaFold2 loss formulation: FAPE + distogram +
    masked MSA + supervised chi + pLDDT + violations.

    Args:
        loss_config: Optional ml_collections.ConfigDict overriding default weights.
    """

    def __init__(self, loss_config: ml_collections.ConfigDict | None = None) -> None:
        super().__init__()
        self._config = loss_config or _default_loss_config()
        self._af_loss = AlphaFoldLoss(self._config)

    def forward(
        self,
        preds: dict[str, Any],
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Compute the full AlphaFold2 loss.

        Args:
            preds: Model prediction dict (may be wrapped in '_raw_outputs').
            batch: Feature dict with ground truth (with recycling dim).

        Returns:
            Dict with 'loss' (total) and per-term breakdown.
        """
        # Strip recycling dimension from batch tensors
        clean: dict[str, Any] = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor) and v.dim() > 0 and v.shape[-1] == 1:
                clean[k] = v.squeeze(-1)
            else:
                clean[k] = v

        raw = preds.get("_raw_outputs", preds)

        # AlphaFoldLoss expects 'out' dict with specific keys
        try:
            cum_loss, breakdown = self._af_loss(raw, clean, _return_breakdown=True)
            result = {k: v for k, v in breakdown.items()}
            result["loss"] = cum_loss
            return result
        except (KeyError, RuntimeError) as e:
            # Fallback for incomplete model outputs (e.g. missing logits)
            logger.debug(f"Full AlphaFoldLoss failed ({e}), using partial losses")
            return self._partial_loss(raw, clean)

    def _partial_loss(
        self,
        raw: dict[str, Any],
        clean: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Compute available losses when model output is incomplete.

        Tries each loss independently and skips those with missing inputs.
        """
        device = _get_device(clean)
        losses: dict[str, torch.Tensor] = {}
        total = torch.tensor(0.0, device=device, requires_grad=True)

        # FAPE backbone
        sm_out = raw.get("sm", {})
        frames = sm_out.get("frames")
        gt_rigid = clean.get("backbone_rigid_tensor")
        gt_mask = clean.get("backbone_rigid_mask")

        if frames is not None and gt_rigid is not None and gt_mask is not None:
            fape = backbone_loss(
                backbone_rigid_tensor=gt_rigid,
                backbone_rigid_mask=gt_mask,
                traj=frames,
                clamp_distance=self._config.fape.backbone.clamp_distance,
                loss_unit_distance=self._config.fape.backbone.loss_unit_distance,
            )
            losses["fape"] = fape
            total = total + self._config.fape.weight * fape
        else:
            # CA coordinate fallback
            pred_pos = raw.get("final_atom_positions")
            gt_pos = clean.get("all_atom_positions")
            gt_atom_mask = clean.get("all_atom_mask")
            if pred_pos is not None and gt_pos is not None and gt_atom_mask is not None:
                pred_ca = pred_pos[:, :, 1, :]
                gt_ca = gt_pos[:, :, 1, :]
                ca_mask = gt_atom_mask[:, :, 1]
                diff = ((pred_ca - gt_ca) ** 2).sum(-1)
                fape = (diff * ca_mask).sum() / ca_mask.sum().clamp(min=1)
                losses["fape"] = fape
                total = total + self._config.fape.weight * fape

        # Distogram
        if "distogram_logits" in raw:
            try:
                dg = distogram_loss(
                    logits=raw["distogram_logits"],
                    **{**clean, **self._config.distogram},
                )
                losses["distogram"] = dg
                total = total + self._config.distogram.weight * dg
            except Exception:
                pass

        # Masked MSA
        if "masked_msa_logits" in raw:
            try:
                msa = masked_msa_loss(
                    logits=raw["masked_msa_logits"],
                    **{**clean, **self._config.masked_msa},
                )
                losses["masked_msa"] = msa
                total = total + self._config.masked_msa.weight * msa
            except Exception:
                pass

        # Supervised chi
        angles = sm_out.get("angles")
        unnorm_angles = sm_out.get("unnormalized_angles")
        if angles is not None and unnorm_angles is not None:
            try:
                chi = supervised_chi_loss(
                    angles, unnorm_angles,
                    **{**clean, **self._config.supervised_chi},
                )
                losses["supervised_chi"] = chi
                total = total + self._config.supervised_chi.weight * chi
            except Exception:
                pass

        # pLDDT loss
        if "lddt_logits" in raw and "final_atom_positions" in raw:
            try:
                plddt = lddt_loss(
                    logits=raw["lddt_logits"],
                    all_atom_pred_pos=raw["final_atom_positions"],
                    **{**clean, **self._config.plddt_loss},
                )
                losses["plddt_loss"] = plddt
                total = total + self._config.plddt_loss.weight * plddt
            except Exception:
                pass

        losses["loss"] = total
        return losses


def _get_device(tensors: dict[str, Any]) -> torch.device:
    """Get device from the first tensor in a dict."""
    for v in tensors.values():
        if isinstance(v, torch.Tensor):
            return v.device
    return torch.device("cpu")
