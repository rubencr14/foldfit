"""PyTorch Dataset for protein structures with collation support."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class StructureDataset(Dataset):  # type: ignore[type-arg]
    """Dataset for protein structures.

    Loads PDB files using a featurizer and returns feature dicts
    compatible with OpenFold's forward pass.

    Args:
        pdb_paths: List of paths to PDB/mmCIF files.
        featurizer: Object with from_pdb(path, msa_path, chain_id) -> dict method.
        msa_provider: Optional MSA provider for alignment data.
        labels: Optional dict mapping PDB stems to scalar labels.
        transform: Optional transform applied to feature dict.
    """

    def __init__(
        self,
        pdb_paths: list[str | Path],
        featurizer: Any = None,
        msa_provider: Any = None,
        labels: dict[str, float] | None = None,
        transform: Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]] | None = None,
    ) -> None:
        self.pdb_paths = [Path(p) for p in pdb_paths]
        self.featurizer = featurizer
        self.msa_provider = msa_provider
        self.labels = labels or {}
        self.transform = transform

        # Pre-filter to existing files
        self.pdb_paths = [p for p in self.pdb_paths if p.exists()]
        if len(self.pdb_paths) == 0:
            logger.warning("No valid PDB files found in provided paths")

    def __len__(self) -> int:
        return len(self.pdb_paths)

    def __getitem__(self, idx: int) -> tuple[dict[str, torch.Tensor], torch.Tensor] | None:
        """Load and featurize a single structure.

        Returns None if featurization fails (skipped by collate_fn).
        """
        pdb_path = self.pdb_paths[idx]
        stem = pdb_path.stem.upper()

        try:
            if self.featurizer is not None:
                msa_data = None
                if self.msa_provider is not None:
                    # Extract sequence first so MSA provider can use it
                    seq, _, _ = self.featurizer._parse_pdb(pdb_path, "A")
                    if seq:
                        msa_data = self.msa_provider.get(sequence=seq, pdb_id=stem)

                features = self.featurizer.from_pdb(pdb_path, msa_data=msa_data)
                if not features:
                    logger.warning(f"Skipping {pdb_path.name}: empty features")
                    return None
            else:
                features = {"path": str(pdb_path)}

            if self.transform is not None:
                features = self.transform(features)

            label = torch.tensor(self.labels.get(stem, 0.0), dtype=torch.float32)
            return features, label

        except Exception as e:
            logger.warning(f"Skipping {pdb_path.name}: {e}")
            return None


def collate_structure_batch(
    batch: list[tuple[dict[str, torch.Tensor], torch.Tensor] | None],
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Collate structure samples into a batch by padding to max length.

    Skips None entries (failed featurizations) automatically.
    """
    # Filter out failed samples
    batch = [b for b in batch if b is not None]
    if not batch:
        return {}, torch.tensor([])

    features_list, labels = zip(*batch, strict=True)
    labels_tensor = torch.stack(list(labels))

    # Find max sequence length across batch
    max_len = 0
    for features in features_list:
        for key, val in features.items():
            if isinstance(val, torch.Tensor) and val.dim() >= 1:
                if key in ("aatype", "residue_index", "target_feat"):
                    max_len = max(max_len, val.shape[0])

    if max_len == 0:
        max_len = 1

    # Pad and stack
    batched: dict[str, list[torch.Tensor]] = {}
    for features in features_list:
        for key, val in features.items():
            if key not in batched:
                batched[key] = []
            if isinstance(val, torch.Tensor):
                batched[key].append(_pad_to_length(val, max_len, key))
            else:
                batched[key].append(val)

    result: dict[str, torch.Tensor] = {}
    for key, vals in batched.items():
        if all(isinstance(v, torch.Tensor) for v in vals):
            try:
                result[key] = torch.stack(vals)
            except RuntimeError:
                result[key] = vals[0]  # Fallback for incompatible shapes
        else:
            result[key] = vals[0]  # type: ignore[assignment]

    return result, labels_tensor


def _pad_to_length(tensor: torch.Tensor, target_len: int, key: str) -> torch.Tensor:
    """Pad a tensor's first sequence dimension to target_len."""
    if tensor.dim() == 0:
        return tensor

    seq_keys = {"aatype", "residue_index", "all_atom_positions", "all_atom_mask", "target_feat"}
    msa_keys = {
        "msa", "deletion_matrix", "msa_mask", "msa_feat", "true_msa", "bert_mask",
        "extra_msa", "extra_msa_mask", "extra_has_deletion",
        "extra_deletion_value", "extra_msa_deletion_value",
    }

    current_len = tensor.shape[0]

    if key in seq_keys and current_len < target_len:
        pad_size = target_len - current_len
        pad_shape = [pad_size] + list(tensor.shape[1:])
        padding = torch.zeros(pad_shape, dtype=tensor.dtype)
        return torch.cat([tensor, padding], dim=0)

    if key in msa_keys and tensor.dim() >= 2 and tensor.shape[1] < target_len:
        pad_size = target_len - tensor.shape[1]
        pad_shape = [tensor.shape[0], pad_size] + list(tensor.shape[2:])
        padding = torch.zeros(pad_shape, dtype=tensor.dtype)
        return torch.cat([tensor, padding], dim=1)

    return tensor
