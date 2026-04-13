"""Antibody-specific evaluation metrics."""

import torch


class AntibodyMetrics:
    """Computes antibody-specific structural quality metrics.

    All methods are stateless and operate on coordinate tensors with
    optional boolean masks to select relevant atoms/residues.
    """

    @staticmethod
    def rmsd(
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute RMSD between predicted and target positions.

        Args:
            pred: Predicted atom positions, shape [..., N, 3].
            target: Ground truth atom positions, shape [..., N, 3].
            mask: Optional boolean mask, shape [..., N]. True for atoms
                to include in the calculation.

        Returns:
            Scalar RMSD value.
        """
        diff = pred - target
        sq_dist = (diff * diff).sum(dim=-1)

        if mask is not None:
            mask = mask.bool()
            n_atoms = mask.sum().clamp(min=1)
            sq_dist = sq_dist * mask.float()
            return torch.sqrt(sq_dist.sum() / n_atoms)

        return torch.sqrt(sq_dist.mean())

    @staticmethod
    def cdr_rmsd(
        pred_positions: torch.Tensor,
        gt_positions: torch.Tensor,
        cdr_masks: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Compute RMSD per CDR region.

        Args:
            pred_positions: Predicted positions, shape [N, 3].
            gt_positions: Ground truth positions, shape [N, 3].
            cdr_masks: Dictionary mapping CDR names (e.g., "CDR-H1",
                "CDR-H3") to boolean masks of shape [N].

        Returns:
            Dictionary mapping CDR names to their RMSD values.
        """
        results = {}
        for cdr_name, mask in cdr_masks.items():
            if mask.any():
                results[cdr_name] = AntibodyMetrics.rmsd(
                    pred_positions, gt_positions, mask
                )
        return results

    @staticmethod
    def framework_rmsd(
        pred_positions: torch.Tensor,
        gt_positions: torch.Tensor,
        cdr_masks: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute RMSD over framework regions (non-CDR residues).

        Args:
            pred_positions: Predicted positions, shape [N, 3].
            gt_positions: Ground truth positions, shape [N, 3].
            cdr_masks: Dictionary of CDR boolean masks of shape [N].

        Returns:
            RMSD over framework residues.
        """
        n = pred_positions.shape[0]
        combined_cdr = torch.zeros(n, dtype=torch.bool, device=pred_positions.device)
        for mask in cdr_masks.values():
            combined_cdr = combined_cdr | mask.bool()
        framework_mask = ~combined_cdr
        return AntibodyMetrics.rmsd(pred_positions, gt_positions, framework_mask)

    @staticmethod
    def interface_contacts(
        positions: torch.Tensor,
        chain_a_mask: torch.Tensor,
        chain_b_mask: torch.Tensor,
        contact_threshold: float = 8.0,
    ) -> int:
        """Count inter-chain contacts within a distance threshold.

        Args:
            positions: Atom positions, shape [N, 3].
            chain_a_mask: Boolean mask for chain A atoms.
            chain_b_mask: Boolean mask for chain B atoms.
            contact_threshold: Distance threshold in Angstroms.

        Returns:
            Number of inter-chain contacts.
        """
        pos_a = positions[chain_a_mask.bool()]
        pos_b = positions[chain_b_mask.bool()]

        if pos_a.shape[0] == 0 or pos_b.shape[0] == 0:
            return 0

        # Pairwise distances: [Na, Nb]
        dists = torch.cdist(pos_a.float(), pos_b.float())
        return int((dists < contact_threshold).sum().item())

    @staticmethod
    def drmsd(
        pred_positions: torch.Tensor,
        gt_positions: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute distance-RMSD (dRMSD) between structures.

        dRMSD measures the RMSD of pairwise distances rather than
        coordinates, making it alignment-free.

        Args:
            pred_positions: Predicted positions, shape [N, 3].
            gt_positions: Ground truth positions, shape [N, 3].
            mask: Optional boolean mask, shape [N].

        Returns:
            Scalar dRMSD value.
        """
        if mask is not None:
            mask = mask.bool()
            pred_positions = pred_positions[mask]
            gt_positions = gt_positions[mask]

        pred_dists = torch.cdist(pred_positions.float(), pred_positions.float())
        gt_dists = torch.cdist(gt_positions.float(), gt_positions.float())

        # Upper triangle only to avoid double counting
        n = pred_dists.shape[0]
        triu_idx = torch.triu_indices(n, n, offset=1)
        pred_upper = pred_dists[triu_idx[0], triu_idx[1]]
        gt_upper = gt_dists[triu_idx[0], triu_idx[1]]

        diff = pred_upper - gt_upper
        return torch.sqrt((diff * diff).mean())
