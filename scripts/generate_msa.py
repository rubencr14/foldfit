"""Generate MSA files for training structures using the configured backend.

Reads PDB files, extracts sequences, and computes MSAs in batch.
Saves results as cached .msa.pt files for fast reloading during training.

Usage:
    # ColabFold public API
    python scripts/generate_msa.py --pdb-dir data/sabdab --output-dir data/msa

    # Self-hosted ColabFold server
    python scripts/generate_msa.py --pdb-dir data/sabdab --output-dir data/msa \
        --backend colabfold --colabfold-server http://localhost:8080

    # Local MMseqs2 with OAS database
    python scripts/generate_msa.py --pdb-dir data/sabdab --output-dir data/msa \
        --backend local --database /data/oas/oas_db

    # Local MMseqs2 with multiple databases (OAS + UniRef30)
    python scripts/generate_msa.py --pdb-dir data/sabdab --output-dir data/msa \
        --backend local --database /data/oas/oas_db --database /data/uniref30/uniref30
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import torch
import typer

from openfold.np import protein as protein_utils
from openfold.np import residue_constants as rc

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = typer.Typer()


def _extract_sequence_from_pdb(pdb_path: Path) -> str:
    """Extract the first viable chain sequence from a PDB file."""
    pdb_str = pdb_path.read_text()
    for chain_id in ["H", "L", "A", "B", None]:
        try:
            prot = protein_utils.from_pdb_string(pdb_str, chain_id=chain_id)
            if len(prot.aatype) > 0:
                return "".join(rc.restypes_with_x[a] for a in prot.aatype)
        except Exception:
            continue
    return ""


@app.command()
def generate(
    pdb_dir: Path = typer.Option(..., help="Directory containing PDB/mmCIF files"),
    output_dir: Path = typer.Option(..., help="Output directory for MSA files"),
    backend: str = typer.Option("colabfold", help="MSA backend: colabfold / local / single"),
    max_msa_depth: int = typer.Option(512, help="Maximum MSA depth"),
    delay: float = typer.Option(1.0, help="Delay between API calls (colabfold only)"),
    skip_existing: bool = typer.Option(True, help="Skip PDBs with existing MSA files"),
    # ColabFold options
    colabfold_server: str = typer.Option(
        "https://api.colabfold.com", help="ColabFold server URL"
    ),
    # Local tool options
    tool: str = typer.Option("mmseqs2", help="Local tool: mmseqs2 / hhblits / jackhmmer"),
    database: Optional[list[str]] = typer.Option(None, help="Database path(s) for local backend"),
    tool_binary: Optional[str] = typer.Option(None, help="Path to tool binary"),
    n_cpu: int = typer.Option(4, help="CPUs for local alignment"),
) -> None:
    """Generate MSA files for all PDB structures in a directory."""
    from foldfit.domain.value_objects import MsaConfig
    from foldfit.infrastructure.data.msa_provider import MsaProvider

    output_dir.mkdir(parents=True, exist_ok=True)

    pdb_files = sorted(
        list(pdb_dir.glob("*.pdb"))
        + list(pdb_dir.glob("*.cif"))
        + list(pdb_dir.glob("*.mmcif"))
    )
    logger.info(f"Found {len(pdb_files)} PDB files in {pdb_dir}")

    if not pdb_files:
        logger.error("No PDB files found")
        raise typer.Exit(1)

    config = MsaConfig(
        backend=backend,
        msa_dir=str(output_dir),
        max_msa_depth=max_msa_depth,
        colabfold_server=colabfold_server,
        tool=tool,
        tool_binary=tool_binary,
        database_paths=database or [],
        n_cpu=n_cpu,
    )
    provider = MsaProvider(config)

    success = 0
    failed = 0

    for i, pdb_path in enumerate(pdb_files):
        pdb_id = pdb_path.stem.upper()
        cache_path = output_dir / f"{pdb_id}.msa.pt"

        if skip_existing and cache_path.exists():
            logger.info(f"[{i+1}/{len(pdb_files)}] Skipping {pdb_id} (exists)")
            success += 1
            continue

        logger.info(f"[{i+1}/{len(pdb_files)}] Processing {pdb_id}...")

        try:
            seq = _extract_sequence_from_pdb(pdb_path)
            if not seq:
                logger.warning(f"  No sequence from {pdb_path.name}, skipping")
                failed += 1
                continue

            logger.info(f"  Sequence length: {len(seq)}")
            result = provider.get(sequence=seq, pdb_id=pdb_id)

            torch.save(result, cache_path)
            n_seqs = result["msa"].shape[0]
            logger.info(f"  Saved {n_seqs} MSA sequences to {cache_path}")
            success += 1

            if backend == "colabfold" and delay > 0:
                time.sleep(delay)

        except Exception as e:
            logger.error(f"  Failed: {e}")
            failed += 1

    logger.info(f"Done: {success}/{len(pdb_files)} ({failed} failed)")


if __name__ == "__main__":
    app()
