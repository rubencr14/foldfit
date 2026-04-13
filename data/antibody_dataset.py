"""Dataset for antibody structures from RCSB PDB.

Provides antibody-specific data loading and CDR annotation on top of
OpenFold3's existing data pipeline.
"""

import logging
from pathlib import Path

import torch
from torch.utils.data import Dataset

from finetuning.data.antibody_fetcher import RCSBAntibodyFetcher
from finetuning.data.cdr_annotator import CDRAnnotator, CDRScheme
from finetuning.data.filters import (
    AntibodyFilter,
    ChainTypeFilter,
    CompositeFilter,
    ResolutionFilter,
)

logger = logging.getLogger(__name__)


class AntibodyPDBDataset(Dataset):
    """Dataset for antibody structures from RCSB PDB.

    Loads antibody structures, applies filtering, and annotates CDR
    regions. Structures are stored as mmCIF files in a cache directory.

    This dataset can be used standalone or integrated with OpenFold3's
    data pipeline via the provided collation utilities.

    Args:
        pdb_ids: List of PDB IDs to include. If None, will search RCSB.
        cache_dir: Directory to cache downloaded structures.
        max_resolution: Maximum resolution for structure filtering.
        require_heavy: Require heavy chain presence.
        require_light: Require light chain presence.
        cdr_scheme: CDR numbering scheme for annotation.
        filters: Optional list of additional filters to apply.
    """

    def __init__(
        self,
        pdb_ids: list[str] | None = None,
        cache_dir: Path = Path("./data/antibody_cache"),
        max_resolution: float = 3.0,
        require_heavy: bool = True,
        require_light: bool = True,
        cdr_scheme: str = "imgt",
        filters: list[AntibodyFilter] | None = None,
        search_max_results: int = 1000,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.cdr_annotator = CDRAnnotator(
            scheme=CDRScheme(cdr_scheme.lower())
        )

        # Build filter pipeline
        base_filters: list[AntibodyFilter] = [
            ResolutionFilter(max_resolution=max_resolution),
            ChainTypeFilter(
                require_heavy=require_heavy,
                require_light=require_light,
            ),
        ]
        if filters:
            base_filters.extend(filters)
        self.filter_pipeline = CompositeFilter(base_filters)

        # Fetch or use provided PDB IDs
        self.fetcher = RCSBAntibodyFetcher(
            cache_dir=self.cache_dir,
            max_resolution=max_resolution,
        )

        if pdb_ids is not None:
            self.pdb_ids = pdb_ids
        else:
            self.pdb_ids = self.fetcher.search(max_results=search_max_results)

        # Download structures
        self.structure_paths = self.fetcher.download(self.pdb_ids)

        # Map PDB IDs to file paths
        self.entries: list[dict] = []
        for pdb_id, path in zip(self.pdb_ids, self.structure_paths):
            if path.exists():
                self.entries.append({"pdb_id": pdb_id, "path": path})

        logger.info(
            f"AntibodyPDBDataset initialized with {len(self.entries)} structures"
        )

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict:
        """Return structure metadata and file path for a single entry.

        Returns a dictionary with:
            - pdb_id: The PDB identifier
            - structure_path: Path to the mmCIF file
            - cdr_annotations: CDR region annotations (if available)
        """
        entry = self.entries[idx]
        result = {
            "pdb_id": entry["pdb_id"],
            "structure_path": str(entry["path"]),
        }
        return result

    def get_train_val_split(
        self, train_ratio: float = 0.9, seed: int = 42
    ) -> tuple["AntibodyPDBDataset", "AntibodyPDBDataset"]:
        """Split the dataset into training and validation subsets.

        Args:
            train_ratio: Fraction of data to use for training.
            seed: Random seed for reproducible splitting.

        Returns:
            Tuple of (train_dataset, val_dataset).
        """
        generator = torch.Generator().manual_seed(seed)
        n_total = len(self.entries)
        n_train = int(n_total * train_ratio)

        indices = torch.randperm(n_total, generator=generator).tolist()
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]

        train_dataset = _SubsetAntibodyDataset(self, train_indices)
        val_dataset = _SubsetAntibodyDataset(self, val_indices)

        logger.info(
            f"Split dataset: {len(train_indices)} train, "
            f"{len(val_indices)} validation"
        )
        return train_dataset, val_dataset


class _SubsetAntibodyDataset(Dataset):
    """Subset wrapper for AntibodyPDBDataset.

    Args:
        parent: The parent dataset.
        indices: Indices into the parent dataset.
    """

    def __init__(self, parent: AntibodyPDBDataset, indices: list[int]):
        self.parent = parent
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        return self.parent[self.indices[idx]]
