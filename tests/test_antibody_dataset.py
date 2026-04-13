"""Tests for antibody dataset."""

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from finetuning.data.antibody_dataset import AntibodyPDBDataset


class TestAntibodyPDBDataset:
    """Tests for the AntibodyPDBDataset class."""

    @patch.object(AntibodyPDBDataset, "__init__", lambda self, **kwargs: None)
    def test_getitem_returns_dict(self):
        """__getitem__ should return a dict with expected keys."""
        ds = AntibodyPDBDataset()
        ds.entries = [
            {"pdb_id": "1IGT", "path": Path("/tmp/1igt.cif")},
            {"pdb_id": "7FAE", "path": Path("/tmp/7fae.cif")},
        ]

        item = ds[0]
        assert "pdb_id" in item
        assert "structure_path" in item
        assert item["pdb_id"] == "1IGT"

    @patch.object(AntibodyPDBDataset, "__init__", lambda self, **kwargs: None)
    def test_len(self):
        """__len__ should return the number of entries."""
        ds = AntibodyPDBDataset()
        ds.entries = [
            {"pdb_id": "1IGT", "path": Path("/tmp/1igt.cif")},
            {"pdb_id": "7FAE", "path": Path("/tmp/7fae.cif")},
            {"pdb_id": "8ABC", "path": Path("/tmp/8abc.cif")},
        ]

        assert len(ds) == 3

    @patch.object(AntibodyPDBDataset, "__init__", lambda self, **kwargs: None)
    def test_train_val_split(self):
        """get_train_val_split should partition the dataset."""
        ds = AntibodyPDBDataset()
        ds.entries = [
            {"pdb_id": f"PDB{i}", "path": Path(f"/tmp/pdb{i}.cif")}
            for i in range(100)
        ]

        train_ds, val_ds = ds.get_train_val_split(train_ratio=0.8, seed=42)

        assert len(train_ds) == 80
        assert len(val_ds) == 20
        assert len(train_ds) + len(val_ds) == len(ds)

    @patch.object(AntibodyPDBDataset, "__init__", lambda self, **kwargs: None)
    def test_train_val_split_reproducible(self):
        """Same seed should produce the same split."""
        ds = AntibodyPDBDataset()
        ds.entries = [
            {"pdb_id": f"PDB{i}", "path": Path(f"/tmp/pdb{i}.cif")}
            for i in range(50)
        ]

        train1, val1 = ds.get_train_val_split(train_ratio=0.8, seed=123)
        train2, val2 = ds.get_train_val_split(train_ratio=0.8, seed=123)

        ids1 = [train1[i]["pdb_id"] for i in range(len(train1))]
        ids2 = [train2[i]["pdb_id"] for i in range(len(train2))]
        assert ids1 == ids2
