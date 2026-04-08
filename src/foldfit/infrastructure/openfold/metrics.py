"""Structure prediction evaluation metrics.

Uses OpenFold's native superimposition for RMSD and compute_plddt for confidence.
GDT-TS computed on top of superimposed coordinates.
"""

from __future__ import annotations

import torch

from openfold.utils.loss import compute_plddt
from openfold.utils.superimposition import superimpose


def ca_rmsd(
    pred_pos: torch.Tensor,
    gt_pos: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Compute CA-atom RMSD per sample using OpenFold's SVD superimposition.

    Args:
        pred_pos: Predicted all-atom positions [B, L, 37, 3].
        gt_pos: Ground truth all-atom positions [B, L, 37, 3].
        mask: All-atom mask [B, L, 37].

    Returns:
        Per-sample RMSD [B] in angstroms.
    """
    pred_ca = pred_pos[:, :, 1, :]  # [B, L, 3]
    gt_ca = gt_pos[:, :, 1, :]
    ca_mask = mask[:, :, 1]  # [B, L]

    _, rmsds = superimpose(gt_ca, pred_ca, ca_mask)
    return rmsds


def gdt_ts(
    pred_pos: torch.Tensor,
    gt_pos: torch.Tensor,
    mask: torch.Tensor,
    thresholds: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0),
) -> torch.Tensor:
    """Compute GDT-TS after superimposition.

    Args:
        pred_pos: Predicted all-atom positions [B, L, 37, 3].
        gt_pos: Ground truth all-atom positions [B, L, 37, 3].
        mask: All-atom mask [B, L, 37].
        thresholds: Distance thresholds in angstroms.

    Returns:
        Per-sample GDT-TS [B] in [0, 1].
    """
    pred_ca = pred_pos[:, :, 1, :]
    gt_ca = gt_pos[:, :, 1, :]
    ca_mask = mask[:, :, 1]

    # Superimpose first, then measure distances
    aligned_pred, _ = superimpose(gt_ca, pred_ca, ca_mask)
    dists = ((aligned_pred - gt_ca) ** 2).sum(-1).sqrt()  # [B, L]
    n = ca_mask.sum(-1).clamp(min=1)

    score = torch.zeros(pred_pos.shape[0], device=pred_pos.device)
    for t in thresholds:
        within = ((dists < t).float() * ca_mask).sum(-1)
        score += within / n

    return score / len(thresholds)


def plddt_score(logits: torch.Tensor) -> torch.Tensor:
    """Compute mean pLDDT per sample from model logits.

    Args:
        logits: pLDDT logits [B, L, num_bins].

    Returns:
        Mean pLDDT per sample [B] in [0, 100].
    """
    per_residue = compute_plddt(logits)
    return per_residue.mean(-1)


def compute_metrics(
    preds: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> dict[str, float]:
    """Compute all structure metrics from model outputs and batch.

    Returns dict with: ca_rmsd, gdt_ts, plddt (when available).
    """
    clean = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor) and v.dim() > 0 and v.shape[-1] == 1:
            clean[k] = v.squeeze(-1)
        else:
            clean[k] = v

    raw = preds.get("_raw_outputs", preds)
    metrics: dict[str, float] = {}

    pred_pos = raw.get("final_atom_positions")
    gt_pos = clean.get("all_atom_positions")
    gt_mask = clean.get("all_atom_mask")

    if pred_pos is not None and gt_pos is not None and gt_mask is not None:
        metrics["ca_rmsd"] = ca_rmsd(pred_pos, gt_pos, gt_mask).mean().item()
        metrics["gdt_ts"] = gdt_ts(pred_pos, gt_pos, gt_mask).mean().item()

    plddt_logits = raw.get("plddt_logits")
    if plddt_logits is not None:
        metrics["plddt"] = plddt_score(plddt_logits).mean().item()

    return metrics
