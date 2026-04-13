"""Tests for antibody evaluation metrics."""

import torch

from finetuning.evaluation.metrics import AntibodyMetrics


class TestAntibodyMetrics:
    """Tests for antibody-specific metrics."""

    def test_rmsd_identical_structures(self):
        """RMSD of identical structures should be zero."""
        pos = torch.randn(100, 3)
        rmsd = AntibodyMetrics.rmsd(pos, pos)
        assert rmsd.item() < 1e-6

    def test_rmsd_with_mask(self):
        """RMSD with mask should only consider masked atoms."""
        pred = torch.zeros(10, 3)
        target = torch.ones(10, 3)
        mask = torch.zeros(10, dtype=torch.bool)
        mask[:5] = True

        rmsd_masked = AntibodyMetrics.rmsd(pred, target, mask)
        rmsd_full = AntibodyMetrics.rmsd(pred, target)

        # Both should be the same here since all offsets are identical
        assert rmsd_masked.item() > 0
        assert abs(rmsd_masked.item() - rmsd_full.item()) < 1e-6

    def test_rmsd_known_value(self):
        """RMSD should match known analytical value."""
        pred = torch.zeros(1, 3)
        target = torch.tensor([[3.0, 4.0, 0.0]])
        rmsd = AntibodyMetrics.rmsd(pred, target)
        # Distance = sqrt(9 + 16) = 5.0
        assert abs(rmsd.item() - 5.0) < 1e-5

    def test_cdr_rmsd_per_region(self):
        """Should compute separate RMSD for each CDR region."""
        pred = torch.randn(100, 3)
        target = pred + 0.1 * torch.randn(100, 3)

        cdr_masks = {
            "CDR-H1": torch.zeros(100, dtype=torch.bool),
            "CDR-H2": torch.zeros(100, dtype=torch.bool),
            "CDR-H3": torch.zeros(100, dtype=torch.bool),
        }
        cdr_masks["CDR-H1"][26:35] = True
        cdr_masks["CDR-H2"][50:58] = True
        cdr_masks["CDR-H3"][93:102] = True

        results = AntibodyMetrics.cdr_rmsd(pred, target, cdr_masks)

        assert "CDR-H1" in results
        assert "CDR-H2" in results
        assert "CDR-H3" in results
        for v in results.values():
            assert v.item() > 0

    def test_cdr_rmsd_empty_mask(self):
        """CDR with empty mask should be excluded from results."""
        pred = torch.randn(50, 3)
        target = pred.clone()

        cdr_masks = {
            "CDR-H1": torch.zeros(50, dtype=torch.bool),  # All False
            "CDR-H3": torch.ones(50, dtype=torch.bool),
        }
        cdr_masks["CDR-H1"][:] = False  # Empty

        results = AntibodyMetrics.cdr_rmsd(pred, target, cdr_masks)
        assert "CDR-H1" not in results
        assert "CDR-H3" in results

    def test_framework_rmsd(self):
        """Framework RMSD should exclude CDR regions."""
        n = 100
        pred = torch.randn(n, 3)
        target = pred.clone()

        # Make CDR regions differ
        cdr_masks = {"CDR-H3": torch.zeros(n, dtype=torch.bool)}
        cdr_masks["CDR-H3"][90:100] = True
        pred[90:100] += 5.0

        fw_rmsd = AntibodyMetrics.framework_rmsd(pred, target, cdr_masks)
        # Framework regions are identical, so RMSD should be ~0
        assert fw_rmsd.item() < 1e-5

    def test_interface_contacts(self):
        """Should count contacts within threshold."""
        positions = torch.zeros(10, 3)
        # Place chain A at origin, chain B at various distances
        positions[5:] = torch.tensor([
            [1.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [50.0, 0.0, 0.0],
        ])

        chain_a = torch.zeros(10, dtype=torch.bool)
        chain_a[:5] = True
        chain_b = ~chain_a

        contacts = AntibodyMetrics.interface_contacts(
            positions, chain_a, chain_b, contact_threshold=8.0
        )
        # Atoms at 1.0 and 5.0 should be within 8A of the origin cluster
        assert contacts > 0

    def test_interface_contacts_empty_chain(self):
        """Should return 0 when a chain has no atoms."""
        positions = torch.randn(10, 3)
        chain_a = torch.ones(10, dtype=torch.bool)
        chain_b = torch.zeros(10, dtype=torch.bool)

        contacts = AntibodyMetrics.interface_contacts(positions, chain_a, chain_b)
        assert contacts == 0

    def test_drmsd_identical(self):
        """dRMSD of identical structures should be zero."""
        pos = torch.randn(50, 3)
        drmsd = AntibodyMetrics.drmsd(pos, pos)
        assert drmsd.item() < 1e-5

    def test_drmsd_positive(self):
        """dRMSD of different structures should be positive."""
        pred = torch.randn(50, 3)
        target = pred + torch.randn(50, 3)
        drmsd = AntibodyMetrics.drmsd(pred, target)
        assert drmsd.item() > 0

    def test_drmsd_with_mask(self):
        """dRMSD with mask should only use selected atoms."""
        pred = torch.randn(50, 3)
        target = pred.clone()
        target[40:] += 10.0  # Perturb last 10

        mask = torch.zeros(50, dtype=torch.bool)
        mask[:40] = True  # Only use first 40 (identical)

        drmsd = AntibodyMetrics.drmsd(pred, target, mask)
        assert drmsd.item() < 1e-5
