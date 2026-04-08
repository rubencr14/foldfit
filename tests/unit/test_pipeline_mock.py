"""Lightweight mock test for the full pipeline dimensions and data flow.

Tests featurizer → dataset → collation → loss → metrics → trainer
with tiny synthetic data. No GPU, no real weights, no network.
Validates that all tensor shapes are consistent end-to-end.

Usage:
    pytest tests/unit/test_pipeline_mock.py -v -s
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

openfold = pytest.importorskip("openfold", reason="OpenFold required")

# Tiny dimensions for fast testing
L = 16          # sequence length
N_MSA = 4       # MSA depth
N_EXTRA = 2     # extra MSA
B = 2           # batch size
T = 4           # templates
NUM_BINS = 64   # distogram bins
PLDDT_BINS = 50 # pLDDT bins


# ── Helpers ───────────────────────────────────────────────────────────────

def _make_synthetic_features(seq_len: int = L, n_msa: int = N_MSA) -> dict[str, torch.Tensor]:
    """Build a synthetic feature dict that matches OpenFold's expected shapes.

    All tensors include the trailing recycling dimension (R=1).
    """
    from openfold.np import residue_constants as rc

    aatype = torch.randint(0, 20, (seq_len,))
    residue_index = torch.arange(seq_len)

    # Atom positions: place CA at random coords, rest zero
    atom_pos = torch.zeros(seq_len, 37, 3)
    atom_pos[:, 1, :] = torch.randn(seq_len, 3) * 10  # CA
    atom_mask = torch.zeros(seq_len, 37)
    atom_mask[:, 1] = 1.0  # CA exists

    # MSA
    msa = torch.randint(0, 21, (n_msa, seq_len))
    msa[0] = aatype  # first row = query
    deletion_matrix = torch.zeros(n_msa, seq_len)
    msa_mask = torch.ones(n_msa, seq_len)
    true_msa = msa.clone()
    bert_mask = (torch.rand(n_msa, seq_len) < 0.15).float()

    # MSA feat: [N, L, 49]
    msa_feat = torch.randn(n_msa, seq_len, 49)

    # Target feat: [L, 22]
    target_feat = torch.zeros(seq_len, 22)
    target_feat[range(seq_len), aatype.clamp(max=21)] = 1.0

    # Extra MSA
    extra_n = min(n_msa, N_EXTRA)
    extra_msa = msa[:extra_n]
    extra_msa_mask = torch.ones(extra_n, seq_len)

    # GT transforms (minimal)
    backbone_rigid_tensor = torch.randn(seq_len, 4, 4)
    backbone_rigid_mask = torch.ones(seq_len)

    # Pseudo-beta
    pseudo_beta = atom_pos[:, 1, :].clone()  # use CA as pseudo-beta
    pseudo_beta_mask = atom_mask[:, 1].clone()

    # Templates (zeros)
    template_aatype = torch.zeros(T, seq_len, dtype=torch.long)
    template_all_atom_positions = torch.zeros(T, seq_len, 37, 3)
    template_all_atom_mask = torch.zeros(T, seq_len, 37)
    template_mask = torch.zeros(T)

    features = {
        "aatype": aatype,
        "residue_index": residue_index,
        "all_atom_positions": atom_pos,
        "all_atom_mask": atom_mask,
        "between_segment_residues": torch.zeros(seq_len, dtype=torch.long),
        "msa": msa,
        "deletion_matrix": deletion_matrix,
        "msa_mask": msa_mask,
        "msa_feat": msa_feat,
        "true_msa": true_msa,
        "bert_mask": bert_mask,
        "target_feat": target_feat,
        "extra_msa": extra_msa,
        "extra_msa_mask": extra_msa_mask,
        "backbone_rigid_tensor": backbone_rigid_tensor,
        "backbone_rigid_mask": backbone_rigid_mask,
        "pseudo_beta": pseudo_beta,
        "pseudo_beta_mask": pseudo_beta_mask,
        "template_aatype": template_aatype,
        "template_all_atom_positions": template_all_atom_positions,
        "template_all_atom_mask": template_all_atom_mask,
        "template_mask": template_mask,
        "seq_length": torch.tensor([seq_len], dtype=torch.int64),
        "seq_mask": torch.ones(seq_len),
    }

    # Add recycling dimension
    skip = {"seq_length"}
    return {
        k: v.unsqueeze(-1) if isinstance(v, torch.Tensor) and k not in skip else v
        for k, v in features.items()
    }


def _make_mock_model_output(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Create fake model outputs matching OpenFold's output format."""
    # Strip recycling dim to get shapes
    clean = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor) and v.dim() > 0 and v.shape[-1] == 1:
            clean[k] = v.squeeze(-1)
        else:
            clean[k] = v

    B_dim = clean["aatype"].shape[0]
    L_dim = clean["aatype"].shape[1]

    return {
        "final_atom_positions": torch.randn(B_dim, L_dim, 37, 3),
        "distogram_logits": torch.randn(B_dim, L_dim, L_dim, NUM_BINS),
        "masked_msa_logits": torch.randn(
            B_dim, clean["msa"].shape[1], L_dim, 23
        ),
        "lddt_logits": torch.randn(B_dim, L_dim, PLDDT_BINS),
        "plddt_logits": torch.randn(B_dim, L_dim, PLDDT_BINS),
        "sm": {
            "frames": torch.randn(8, B_dim, L_dim, 4, 4),
            "angles": torch.randn(8, B_dim, L_dim, 7, 2),
            "unnormalized_angles": torch.randn(8, B_dim, L_dim, 7, 2),
            "positions": torch.randn(8, B_dim, L_dim, 14, 3),
        },
    }


# ── Test: Synthetic Features ─────────────────────────────────────────────

class TestSyntheticFeatures:
    """Verify synthetic feature dict has correct shapes."""

    def test_shapes(self) -> None:
        f = _make_synthetic_features()
        # Recycling dim adds 1 to the end
        assert f["aatype"].shape == (L, 1)
        assert f["all_atom_positions"].shape == (L, 37, 3, 1)
        assert f["msa"].shape == (N_MSA, L, 1)
        assert f["msa_feat"].shape == (N_MSA, L, 49, 1)
        assert f["target_feat"].shape == (L, 22, 1)
        assert f["seq_length"].shape == (1,)

    def test_variable_lengths(self) -> None:
        f8 = _make_synthetic_features(seq_len=8)
        f32 = _make_synthetic_features(seq_len=32)
        assert f8["aatype"].shape[0] == 8
        assert f32["aatype"].shape[0] == 32


# ── Test: Collation ──────────────────────────────────────────────────────

class TestCollation:
    """Verify the collate function pads and batches correctly."""

    def test_collate_same_length(self) -> None:
        from foldfit.infrastructure.data.structure_dataset import collate_structure_batch

        f1 = _make_synthetic_features(seq_len=L)
        f2 = _make_synthetic_features(seq_len=L)
        label = torch.tensor(0.0)

        batch_result, labels = collate_structure_batch([(f1, label), (f2, label)])

        assert labels.shape == (2,)
        assert batch_result["aatype"].shape[0] == 2  # batch dim
        assert batch_result["aatype"].shape[1] == L

    def test_collate_variable_length(self) -> None:
        from foldfit.infrastructure.data.structure_dataset import collate_structure_batch

        f_short = _make_synthetic_features(seq_len=8)
        f_long = _make_synthetic_features(seq_len=16)
        label = torch.tensor(0.0)

        batch_result, labels = collate_structure_batch([(f_short, label), (f_long, label)])

        # Should pad to max length (16)
        assert batch_result["aatype"].shape[1] == 16

    def test_collate_skips_none(self) -> None:
        from foldfit.infrastructure.data.structure_dataset import collate_structure_batch

        f1 = _make_synthetic_features()
        label = torch.tensor(0.0)

        batch_result, labels = collate_structure_batch([None, (f1, label), None])
        assert labels.shape == (1,)


# ── Test: Loss ────────────────────────────────────────────────────────────

class TestLoss:
    """Verify loss computation with mock model outputs."""

    def test_partial_loss_fape_fallback(self) -> None:
        from foldfit.infrastructure.openfold.loss import OpenFoldLoss

        loss_fn = OpenFoldLoss()

        f = _make_synthetic_features()
        # Add batch dim
        batch = {k: v.unsqueeze(0) for k, v in f.items()}

        # Mock output with only final_atom_positions (triggers CA fallback)
        preds = {
            "final_atom_positions": torch.randn(1, L, 37, 3, requires_grad=True),
        }

        result = loss_fn(preds, batch)
        assert "loss" in result
        assert result["loss"].requires_grad
        assert not result["loss"].isnan()

    def test_partial_loss_with_distogram(self) -> None:
        from foldfit.infrastructure.openfold.loss import OpenFoldLoss

        loss_fn = OpenFoldLoss()

        f = _make_synthetic_features()
        batch = {k: v.unsqueeze(0) for k, v in f.items()}

        preds = {
            "final_atom_positions": torch.randn(1, L, 37, 3, requires_grad=True),
            "distogram_logits": torch.randn(1, L, L, NUM_BINS, requires_grad=True),
        }

        result = loss_fn(preds, batch)
        assert "loss" in result
        assert "distogram" in result or "fape" in result

    def test_partial_loss_with_masked_msa(self) -> None:
        from foldfit.infrastructure.openfold.loss import OpenFoldLoss

        loss_fn = OpenFoldLoss()

        f = _make_synthetic_features()
        batch = {k: v.unsqueeze(0) for k, v in f.items()}

        preds = {
            "final_atom_positions": torch.randn(1, L, 37, 3, requires_grad=True),
            "masked_msa_logits": torch.randn(1, N_MSA, L, 23, requires_grad=True),
        }

        result = loss_fn(preds, batch)
        assert "loss" in result
        assert result["loss"].requires_grad


# ── Test: Metrics ─────────────────────────────────────────────────────────

class TestMetrics:
    """Verify metrics produce valid outputs."""

    def test_ca_rmsd_shape(self) -> None:
        from foldfit.infrastructure.openfold.metrics import ca_rmsd

        pred = torch.randn(B, L, 37, 3)
        gt = torch.randn(B, L, 37, 3)
        mask = torch.ones(B, L, 37)

        result = ca_rmsd(pred, gt, mask)
        assert result.shape == (B,)
        assert (result >= 0).all()

    def test_gdt_ts_range(self) -> None:
        from foldfit.infrastructure.openfold.metrics import gdt_ts

        # Identical structures → GDT-TS = 1.0
        coords = torch.randn(B, L, 37, 3)
        mask = torch.ones(B, L, 37)

        result = gdt_ts(coords, coords, mask)
        assert result.shape == (B,)
        assert torch.allclose(result, torch.ones(B), atol=1e-5)

    def test_plddt_score_range(self) -> None:
        from foldfit.infrastructure.openfold.metrics import plddt_score

        logits = torch.randn(B, L, PLDDT_BINS)
        result = plddt_score(logits)
        assert result.shape == (B,)
        # pLDDT should be in [0, 100]
        assert (result >= 0).all()
        assert (result <= 100).all()

    def test_compute_metrics_with_mock_output(self) -> None:
        from foldfit.infrastructure.openfold.metrics import compute_metrics

        f = _make_synthetic_features()
        batch = {k: v.unsqueeze(0) for k, v in f.items()}
        preds = _make_mock_model_output(batch)

        metrics = compute_metrics(preds, batch)
        assert "ca_rmsd" in metrics
        assert "gdt_ts" in metrics
        assert metrics["ca_rmsd"] >= 0
        assert 0 <= metrics["gdt_ts"] <= 1


# ── Test: Trainer with Mock Model ─────────────────────────────────────────

class TestTrainerMock:
    """Test the trainer with a tiny mock model that mimics OpenFold output shapes."""

    def test_training_loop_runs(self) -> None:
        from foldfit.domain.value_objects import LoraConfig, TrainingConfig
        from foldfit.infrastructure.peft.injector import LoraInjector
        from foldfit.infrastructure.training.trainer import Trainer

        # Tiny model that produces loss-compatible outputs
        class MockOpenFoldModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear_q = nn.Linear(8, 8)
                self.linear_v = nn.Linear(8, 8)
                self.proj = nn.Linear(8, 3)

            def forward(self, batch: dict) -> dict:
                # Not used directly — we use model_forward_fn
                return {}

        model = MockOpenFoldModel()

        # Inject LoRA
        injector = LoraInjector()
        lora_config = LoraConfig(rank=2, alpha=4.0, target_modules=["linear_q", "linear_v"])
        injector.apply(model, lora_config)

        # Custom forward that produces a simple loss
        def mock_forward(m: nn.Module, batch: tuple) -> dict:
            features, _labels = batch
            # Strip recycling dim
            aatype = features["aatype"]
            if aatype.dim() > 2 and aatype.shape[-1] == 1:
                aatype = aatype.squeeze(-1)

            B_dim, L_dim = aatype.shape
            x = torch.randn(B_dim, L_dim, 8)
            h = m.linear_q(x) + m.linear_v(x)
            pred = m.proj(h)  # [B, L, 3]

            # Simple coordinate loss against random target
            target = torch.randn_like(pred)
            loss = ((pred - target) ** 2).mean()
            return {"loss": loss}

        # Build tiny dataloader
        samples = []
        for _ in range(8):
            f = _make_synthetic_features(seq_len=L)
            label = torch.tensor(0.0)
            samples.append((f, label))

        from foldfit.infrastructure.data.structure_dataset import collate_structure_batch

        loader = torch.utils.data.DataLoader(
            samples,
            batch_size=2,
            collate_fn=collate_structure_batch,
        )

        config = TrainingConfig(
            epochs=2,
            learning_rate=1e-3,
            accumulation_steps=1,
            amp=False,
            scheduler="constant",
            early_stopping_patience=0,
        )

        trainer = Trainer(config)
        history = trainer.fit(
            model=model,
            loss_fn=nn.Identity(),  # not used — model_forward_fn returns loss
            peft=injector,
            head=None,
            train_loader=loader,
            val_loader=loader,
            model_forward_fn=mock_forward,
        )

        assert len(history) == 2
        assert all("train_loss" in h for h in history)
        assert all("val_loss" in h for h in history)
        assert history[0]["train_loss"] > 0


# ── Test: Featurizer with real OpenFold transforms ────────────────────────

class TestFeaturizerReal:
    """Test the featurizer with a tiny synthetic PDB-like input."""

    def test_from_sequence_shapes(self) -> None:
        from foldfit.infrastructure.openfold.featurizer import OpenFoldFeaturizer

        featurizer = OpenFoldFeaturizer(
            max_seq_len=L, num_msa=N_MSA, num_extra_msa=N_EXTRA, training=False,
        )
        features = featurizer.from_sequence("ACDEFGHIKLMNPQRS")  # 16 AAs

        assert features, "Featurizer returned empty dict"
        assert "aatype" in features
        assert "msa_feat" in features
        assert "target_feat" in features
        assert "seq_mask" in features

        # Check recycling dim is present
        assert features["aatype"].shape[-1] == 1

    def test_from_sequence_training_mode(self) -> None:
        from foldfit.infrastructure.openfold.featurizer import OpenFoldFeaturizer

        featurizer = OpenFoldFeaturizer(
            max_seq_len=L, num_msa=N_MSA, training=True,
        )
        features = featurizer.from_sequence("ACDEFGHIKLMNPQRS")

        assert features
        # In training mode, bert_mask should have some masked positions
        bert_mask = features["bert_mask"].squeeze(-1)
        # With single-sequence MSA and 15% mask rate, we might get 0 masks
        # but the key should exist
        assert bert_mask.shape[0] >= 1

    def test_msa_data_injection(self) -> None:
        from foldfit.infrastructure.openfold.featurizer import OpenFoldFeaturizer

        featurizer = OpenFoldFeaturizer(
            max_seq_len=L, num_msa=N_MSA, training=False,
        )

        # Synthetic MSA data (like from MsaProvider)
        msa_data = {
            "msa": torch.randint(0, 21, (10, L)),
            "deletion_matrix": torch.zeros(10, L),
            "msa_mask": torch.ones(10, L),
        }

        features = featurizer.from_sequence.__func__  # can't easily call from_pdb without file
        # Instead, test _assemble_features directly
        import numpy as np
        from openfold.np import residue_constants as rc

        seq = "ACDEFGHIKLMNPQRS"
        aatype = np.array([rc.restype_order.get(aa, rc.restype_num) for aa in seq])
        atom_pos = np.zeros((L, 37, 3), dtype=np.float32)
        atom_mask = np.zeros((L, 37), dtype=np.float32)

        result = featurizer._assemble_features(
            aatype, atom_pos, atom_mask, L, seq,
            msa_data=msa_data, msa_path=None,
        )
        assert result
        # MSA should have been sampled down to num_msa
        msa_shape = result["msa"].squeeze(-1).shape
        assert msa_shape[0] <= N_MSA
        assert msa_shape[1] == L


# ── Test: PDB Writer ─────────────────────────────────────────────────────

class TestPDBWriter:
    """Test PDB output generation."""

    def test_coords_to_pdb(self) -> None:
        import numpy as np

        from foldfit.infrastructure.openfold.pdb_writer import coords_to_pdb

        seq = "ACDEF"
        coords = np.random.randn(5, 37, 3).astype(np.float32)
        plddt = np.array([80.0, 75.0, 90.0, 85.0, 70.0], dtype=np.float32)

        pdb_str = coords_to_pdb(seq, coords, plddt)
        assert "ATOM" in pdb_str
        assert "END" in pdb_str

    def test_coords_to_pdb_no_plddt(self) -> None:
        import numpy as np

        from foldfit.infrastructure.openfold.pdb_writer import coords_to_pdb

        seq = "GHI"
        coords = np.random.randn(3, 37, 3).astype(np.float32)

        pdb_str = coords_to_pdb(seq, coords)
        assert "ATOM" in pdb_str
