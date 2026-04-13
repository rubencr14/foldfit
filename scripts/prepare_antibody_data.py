#!/usr/bin/env python
"""Prepare antibody data for LoRA fine-tuning.

End-to-end pipeline:
  1. Download antibody mmCIF files from RCSB PDB
  2. Preprocess structures (mmCIF -> NPZ + metadata cache)
  3. Create training dataset cache
  4. (Optional) Generate MSAs via ColabFold server

Usage:
    PYTHONPATH="$PWD:$PWD/openfold-3" python -m finetuning.scripts.prepare_antibody_data \
        --output-dir ./data/antibody_training \
        --ccd-path ~/.openfold3/chemical_component_dictionary.cif \
        --max-structures 100 \
        --max-resolution 3.0
"""

import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import click

logger = logging.getLogger(__name__)


@dataclass
class AntibodyDataPaths:
    """Paths produced by the data preparation pipeline."""

    root: Path
    raw_cif_dir: Path = field(init=False)
    preprocessed_dir: Path = field(init=False)
    structure_files_dir: Path = field(init=False)
    reference_mols_dir: Path = field(init=False)
    metadata_cache_path: Path = field(init=False)
    dataset_cache_path: Path = field(init=False)
    fasta_dir: Path = field(init=False)
    pdb_ids_file: Path = field(init=False)

    def __post_init__(self):
        self.raw_cif_dir = self.root / "raw_cif"
        self.preprocessed_dir = self.root / "preprocessed"
        self.structure_files_dir = self.preprocessed_dir / "structure_files"
        self.reference_mols_dir = self.preprocessed_dir / "reference_mols"
        self.metadata_cache_path = self.preprocessed_dir / "metadata_cache.json"
        self.dataset_cache_path = self.root / "dataset_cache.json"
        self.fasta_dir = self.preprocessed_dir / "fasta"
        self.pdb_ids_file = self.root / "pdb_ids.txt"

    def create_dirs(self):
        for d in [
            self.root,
            self.raw_cif_dir,
            self.preprocessed_dir,
            self.structure_files_dir,
            self.reference_mols_dir,
            self.fasta_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)


def step_1_fetch_antibodies(
    paths: AntibodyDataPaths,
    max_structures: int,
    max_resolution: float,
    pdb_ids_file: str | None = None,
) -> list[str]:
    """Step 1: Fetch antibody PDB IDs and download mmCIF files."""
    logger.info("=" * 60)
    logger.info("STEP 1: Fetching antibody structures from RCSB PDB")
    logger.info("=" * 60)

    from finetuning.data.antibody_fetcher import RCSBAntibodyFetcher

    fetcher = RCSBAntibodyFetcher(
        cache_dir=paths.raw_cif_dir,
        max_resolution=max_resolution,
    )

    if pdb_ids_file:
        pdb_ids = RCSBAntibodyFetcher.load_pdb_ids_from_file(pdb_ids_file)
        logger.info(f"Loaded {len(pdb_ids)} PDB IDs from {pdb_ids_file}")
    else:
        pdb_ids = fetcher.search(max_results=max_structures)
        logger.info(f"Found {len(pdb_ids)} antibody structures")

    fetcher.download(pdb_ids, file_format="cif")

    # Save PDB IDs list
    with open(paths.pdb_ids_file, "w") as f:
        f.write("# Antibody PDB IDs for LoRA fine-tuning\n")
        for pid in pdb_ids:
            f.write(f"{pid}\n")

    logger.info(f"Downloaded {len(pdb_ids)} structures to {paths.raw_cif_dir}")
    return pdb_ids


def step_2_preprocess(paths: AntibodyDataPaths, ccd_path: Path, num_workers: int):
    """Step 2: Preprocess mmCIF files into NPZ + metadata cache."""
    logger.info("=" * 60)
    logger.info("STEP 2: Preprocessing structures")
    logger.info("=" * 60)

    from openfold3.core.data.pipelines.preprocessing.structure import (
        preprocess_cif_dir_of3,
    )

    import multiprocessing as mp

    if num_workers > 0:
        log_queue: mp.Queue = mp.Queue(-1)
    else:
        log_queue = None

    preprocess_cif_dir_of3(
        cif_dir=paths.raw_cif_dir,
        ccd_path=ccd_path,
        biotite_ccd_path=None,
        out_dir=paths.preprocessed_dir,
        output_formats=["npz"],
        max_polymer_chains=20,
        num_workers=num_workers,
        chunksize=10,
        log_queue=log_queue,
        early_stop=None,
    )

    logger.info(f"Preprocessing complete. Output in {paths.preprocessed_dir}")

    # Verify outputs
    npz_count = len(list(paths.structure_files_dir.glob("*.npz")))
    logger.info(f"Generated {npz_count} NPZ structure files")

    if paths.metadata_cache_path.exists():
        logger.info(f"Metadata cache: {paths.metadata_cache_path}")
    else:
        # The metadata cache might be named differently (e.g. metadata.json)
        cache_files = list(paths.preprocessed_dir.glob("metadata*.json"))
        if cache_files:
            paths.metadata_cache_path = cache_files[0]
            logger.info(f"Found metadata cache: {paths.metadata_cache_path}")
        else:
            logger.warning("No metadata cache found! Check preprocessing output.")


def step_3_create_dataset_cache(paths: AntibodyDataPaths, max_resolution: float):
    """Step 3: Create training dataset cache from preprocessing metadata."""
    logger.info("=" * 60)
    logger.info("STEP 3: Creating training dataset cache")
    logger.info("=" * 60)

    if not paths.metadata_cache_path.exists():
        logger.error(f"Metadata cache not found at {paths.metadata_cache_path}")
        raise FileNotFoundError(paths.metadata_cache_path)

    # Read the metadata cache and build a simplified dataset cache
    _create_simple_dataset_cache(
        metadata_cache_path=paths.metadata_cache_path,
        output_path=paths.dataset_cache_path,
        max_resolution=max_resolution,
    )

    logger.info(f"Dataset cache created at {paths.dataset_cache_path}")


def _create_simple_dataset_cache(
    metadata_cache_path: Path,
    output_path: Path,
    max_resolution: float,
):
    """Create a dataset cache compatible with WeightedPDBDataset.

    This builds the cache in the format expected by OpenFold3's
    DatasetCache, filtering by resolution.
    """
    import json

    with open(metadata_cache_path) as f:
        metadata = json.load(f)

    structure_data = metadata.get("structure_data", {})
    ref_mol_data = metadata.get("reference_molecule_data", {})

    # Build a simplified dataset cache compatible with WeightedPDBDataset
    dataset_cache = {
        "_type": "DatasetCache",
        "name": "antibody-lora-training",
        "structure_data": {},
        "reference_molecule_data": {},
    }

    skipped = 0
    included = 0

    for pdb_id, entry in structure_data.items():
        status = entry.get("status", "")
        if "skipped" in str(status).lower():
            skipped += 1
            continue

        resolution = entry.get("resolution")
        if resolution is not None and resolution > max_resolution:
            skipped += 1
            continue

        # Build chain data from preprocessing metadata
        chains = {}
        for chain_id, chain_info in entry.get("chains", {}).items():
            chains[chain_id] = {
                "molecule_type": chain_info.get("molecule_type"),
                "sequence": chain_info.get("sequence", ""),
                "alignment_representative_id": None,
                "template_ids": [],
            }

        # Get preferred chains/interfaces
        preferred_chains = entry.get("preferred_chains")
        preferred_interfaces = entry.get("preferred_interfaces")

        dataset_cache["structure_data"][pdb_id] = {
            "chains": chains,
            "resolution": resolution,
            "preferred_chains": preferred_chains,
            "preferred_interfaces": preferred_interfaces,
            "cluster_id": None,
            "cluster_size": 1,
        }
        included += 1

    # Reference molecule data
    for mol_id, mol_info in ref_mol_data.items():
        dataset_cache["reference_molecule_data"][mol_id] = {
            "set_fallback_to_nan": False,
        }

    with open(output_path, "w") as f:
        json.dump(dataset_cache, f, indent=2)

    logger.info(f"Dataset cache: {included} structures included, {skipped} skipped")


def generate_training_yaml(
    paths: AntibodyDataPaths,
    checkpoint_path: Path,
    output_path: Path,
):
    """Generate a training YAML config for LoRA fine-tuning."""
    logger.info("=" * 60)
    logger.info("Generating training YAML config")
    logger.info("=" * 60)

    config = {
        "lora": {
            "rank": 8,
            "alpha": 16.0,
            "dropout": 0.05,
            "target_modules": ["linear_q", "linear_k", "linear_v", "linear_o"],
            "target_blocks": ["pairformer_stack"],
        },
        "training": {
            "learning_rate": 5e-5,
            "weight_decay": 0.01,
            "max_epochs": 10,
            "warmup_steps": 100,
            "batch_size": 1,
            "ema_decay": 0.999,
            "gradient_clip_val": 1.0,
            "scheduler": "cosine",
            "num_workers": 4,
            "precision": "bf16-mixed",
        },
        "data": {
            "dataset_cache_path": str(paths.dataset_cache_path),
            "structure_files_dir": str(paths.structure_files_dir),
            "reference_mols_dir": str(paths.reference_mols_dir),
            "structure_file_format": "npz",
        },
        "pretrained_checkpoint": str(checkpoint_path),
        "output_dir": str(paths.root / "output"),
        "seed": 42,
    }

    import yaml

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Training config saved to {output_path}")


@click.command()
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    required=True,
    help="Root directory for all prepared data.",
)
@click.option(
    "--ccd-path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to chemical_component_dictionary.cif",
)
@click.option(
    "--max-structures",
    type=int,
    default=100,
    help="Max number of antibody structures to fetch.",
)
@click.option(
    "--max-resolution",
    type=float,
    default=3.0,
    help="Max resolution cutoff in Angstroms.",
)
@click.option(
    "--pdb-ids-file",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Optional file with PDB IDs (one per line). Skips RCSB search.",
)
@click.option(
    "--checkpoint-path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to pretrained checkpoint (for generating training YAML).",
)
@click.option(
    "--num-workers",
    type=int,
    default=4,
    help="Number of workers for preprocessing.",
)
@click.option(
    "--skip-download",
    is_flag=True,
    help="Skip download step (use existing mmCIF files).",
)
def main(
    output_dir: Path,
    ccd_path: Path,
    max_structures: int,
    max_resolution: float,
    pdb_ids_file: Path | None,
    checkpoint_path: Path | None,
    num_workers: int,
    skip_download: bool,
):
    """Prepare antibody data for OpenFold3 LoRA fine-tuning.

    This script runs the full data preparation pipeline:
    1. Fetch antibody structures from RCSB PDB
    2. Preprocess mmCIF files (tokenization, bond detection, etc.)
    3. Create training dataset cache
    4. Generate training configuration YAML
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    paths = AntibodyDataPaths(root=output_dir)
    paths.create_dirs()

    # Step 1: Download
    if not skip_download:
        pdb_ids = step_1_fetch_antibodies(
            paths, max_structures, max_resolution, pdb_ids_file
        )
    else:
        logger.info("Skipping download step")
        pdb_ids = []
        if paths.pdb_ids_file.exists():
            from finetuning.data.antibody_fetcher import RCSBAntibodyFetcher

            pdb_ids = RCSBAntibodyFetcher.load_pdb_ids_from_file(paths.pdb_ids_file)

    # Step 2: Preprocess
    step_2_preprocess(paths, ccd_path, num_workers)

    # Step 3: Dataset cache
    step_3_create_dataset_cache(paths, max_resolution)

    # Generate training config
    if checkpoint_path:
        generate_training_yaml(
            paths,
            checkpoint_path,
            paths.root / "train_config.yml",
        )

    logger.info("=" * 60)
    logger.info("DATA PREPARATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Output directory:    {paths.root}")
    logger.info(f"Structure files:     {paths.structure_files_dir}")
    logger.info(f"Dataset cache:       {paths.dataset_cache_path}")
    if checkpoint_path:
        logger.info(f"Training config:     {paths.root / 'train_config.yml'}")
    logger.info("")
    logger.info("Next step: run training with:")
    logger.info(
        f"  python -m finetuning.scripts.train_lora "
        f"--config {paths.root / 'train_config.yml'}"
    )


if __name__ == "__main__":
    main()
