#!/usr/bin/env python
"""Compute MSAs for antibody sequences using the ColabFold MSA server.

Reads FASTA files from preprocessed structures and queries the free
ColabFold API (https://api.colabfold.com) to generate MSAs.

Usage:
    PYTHONPATH="$PWD:$PWD/openfold-3" python -m finetuning.scripts.compute_msas \
        --data-dir ./data/antibody_training \
        --user-agent "openfold3-lora-finetuning"
"""

import logging
import time
from pathlib import Path

import click
import numpy as np

logger = logging.getLogger(__name__)


def parse_fasta(fasta_path: Path) -> dict[str, str]:
    """Parse a FASTA file into {chain_id: sequence} dict."""
    sequences = {}
    current_id = None
    current_seq = []

    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_id is not None:
                    sequences[current_id] = "".join(current_seq)
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)

    if current_id is not None:
        sequences[current_id] = "".join(current_seq)

    return sequences


def query_colabfold(
    sequence: str,
    output_dir: Path,
    chain_id: str,
    user_agent: str = "openfold3-lora",
    host_url: str = "https://api.colabfold.com",
    max_retries: int = 5,
) -> Path | None:
    """Query ColabFold MSA server for a single sequence.

    Args:
        sequence: Amino acid sequence.
        output_dir: Directory to save the a3m MSA file.
        chain_id: Chain identifier for naming.
        user_agent: User agent string for the API.
        host_url: ColabFold API URL.
        max_retries: Maximum number of retries on failure.

    Returns:
        Path to the a3m file, or None if failed.
    """
    import requests

    output_dir.mkdir(parents=True, exist_ok=True)
    a3m_path = output_dir / f"{chain_id}.a3m"

    if a3m_path.exists() and a3m_path.stat().st_size > 0:
        logger.info(f"  MSA already exists: {a3m_path}")
        return a3m_path

    # Submit job
    headers = {"User-Agent": user_agent}

    for attempt in range(max_retries):
        try:
            # Submit search
            data = {"q": f">query\n{sequence}", "mode": "all", "database": "uniref30"}
            response = requests.post(
                f"{host_url}/ticket/msa",
                data=data,
                headers=headers,
                timeout=60,
            )
            response.raise_for_status()
            ticket = response.json()
            ticket_id = ticket["id"]
            logger.info(f"  Submitted job {ticket_id} for chain {chain_id}")

            # Poll for results
            while True:
                status_response = requests.get(
                    f"{host_url}/ticket/{ticket_id}",
                    headers=headers,
                    timeout=30,
                )
                status_response.raise_for_status()
                status = status_response.json()

                if status["status"] == "COMPLETE":
                    break
                elif status["status"] == "ERROR":
                    logger.warning(f"  MSA server error for chain {chain_id}")
                    break

                time.sleep(2)

            # Download result
            if status["status"] == "COMPLETE":
                result_response = requests.get(
                    f"{host_url}/result/download/{ticket_id}",
                    headers=headers,
                    timeout=120,
                )
                result_response.raise_for_status()

                # The response is a tar.gz with a3m files
                import io
                import tarfile

                tar_bytes = io.BytesIO(result_response.content)
                with tarfile.open(fileobj=tar_bytes, mode="r:gz") as tar:
                    # Find the main a3m file (uniref.a3m or similar)
                    for member in tar.getmembers():
                        if member.name.endswith(".a3m") and "uniref" in member.name.lower():
                            f = tar.extractfile(member)
                            if f:
                                a3m_content = f.read().decode("utf-8")
                                with open(a3m_path, "w") as out:
                                    out.write(a3m_content)
                                logger.info(
                                    f"  Saved MSA: {a3m_path} "
                                    f"({len(a3m_content.splitlines())} lines)"
                                )
                                return a3m_path

                    # If no uniref, take the first a3m
                    for member in tar.getmembers():
                        if member.name.endswith(".a3m"):
                            f = tar.extractfile(member)
                            if f:
                                a3m_content = f.read().decode("utf-8")
                                with open(a3m_path, "w") as out:
                                    out.write(a3m_content)
                                logger.info(f"  Saved MSA: {a3m_path}")
                                return a3m_path

                logger.warning(f"  No a3m file found in response for chain {chain_id}")
                return None

        except requests.exceptions.RequestException as e:
            logger.warning(
                f"  Attempt {attempt + 1}/{max_retries} failed for chain {chain_id}: {e}"
            )
            if attempt < max_retries - 1:
                wait = 10 * (attempt + 1)
                logger.info(f"  Retrying in {wait}s...")
                time.sleep(wait)

    logger.error(f"  Failed to get MSA for chain {chain_id} after {max_retries} attempts")
    return None


def create_alignment_arrays(
    msa_dir: Path,
    structure_dir: Path,
    output_dir: Path,
):
    """Convert a3m MSA files to alignment arrays for OpenFold3.

    Creates numpy arrays in the format expected by BaseOF3Dataset's
    MSA sample processor.

    Args:
        msa_dir: Directory containing per-PDB a3m files.
        structure_dir: Directory containing preprocessed FASTA files.
        output_dir: Directory to save alignment arrays.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for pdb_dir in sorted(msa_dir.iterdir()):
        if not pdb_dir.is_dir():
            continue

        pdb_id = pdb_dir.name
        pdb_output = output_dir / pdb_id
        pdb_output.mkdir(parents=True, exist_ok=True)

        a3m_files = list(pdb_dir.glob("*.a3m"))
        if not a3m_files:
            logger.warning(f"  No a3m files for {pdb_id}, creating empty alignment")
            # Create minimal empty alignment
            fasta_path = structure_dir / pdb_id / f"{pdb_id}.fasta"
            if fasta_path.exists():
                sequences = parse_fasta(fasta_path)
                for chain_id, seq in sequences.items():
                    # Save a minimal alignment (just the query sequence)
                    arr_path = pdb_output / f"{chain_id}.npy"
                    # Encode sequence as integer array (A=0, C=1, ..., gap=20)
                    aa_map = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
                    encoded = np.array(
                        [[aa_map.get(c, 20) for c in seq]], dtype=np.int8
                    )
                    np.save(arr_path, encoded)
            continue

        for a3m_file in a3m_files:
            chain_id = a3m_file.stem
            arr_path = pdb_output / f"{chain_id}.npy"

            # Parse a3m to numpy array
            sequences_in_msa = []
            current_seq = []

            with open(a3m_file) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith(">"):
                        if current_seq:
                            sequences_in_msa.append("".join(current_seq))
                        current_seq = []
                    elif line:
                        # Remove lowercase insertions (a3m format)
                        filtered = "".join(c for c in line if not c.islower())
                        current_seq.append(filtered)

            if current_seq:
                sequences_in_msa.append("".join(current_seq))

            if sequences_in_msa:
                aa_map = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY-")}
                max_len = max(len(s) for s in sequences_in_msa)
                encoded = np.full(
                    (len(sequences_in_msa), max_len), 20, dtype=np.int8
                )
                for i, seq in enumerate(sequences_in_msa):
                    for j, c in enumerate(seq):
                        encoded[i, j] = aa_map.get(c, 20)

                np.save(arr_path, encoded)
                logger.info(
                    f"  Alignment array: {arr_path} "
                    f"({encoded.shape[0]} seqs x {encoded.shape[1]} positions)"
                )


@click.command()
@click.option(
    "--data-dir",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Root data directory (with preprocessed/ subdirectory).",
)
@click.option(
    "--user-agent",
    type=str,
    default="openfold3-antibody-lora",
    help="User agent for ColabFold API.",
)
@click.option(
    "--skip-existing",
    is_flag=True,
    default=True,
    help="Skip sequences that already have MSAs computed.",
)
def main(data_dir: Path, user_agent: str, skip_existing: bool):
    """Compute MSAs for preprocessed antibody structures using ColabFold."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    structure_dir = data_dir / "preprocessed" / "structure_files"
    msa_dir = data_dir / "msas"
    alignment_dir = data_dir / "alignment_arrays"

    if not structure_dir.exists():
        logger.error(f"Structure directory not found: {structure_dir}")
        logger.error("Run prepare_antibody_data.py first!")
        return

    # Find all PDB IDs with FASTA files
    pdb_dirs = sorted(
        d for d in structure_dir.iterdir()
        if d.is_dir() and (d / f"{d.name}.fasta").exists()
    )
    logger.info(f"Found {len(pdb_dirs)} structures with FASTA files")

    # Step 1: Compute MSAs via ColabFold
    logger.info("=" * 60)
    logger.info("STEP 1: Computing MSAs via ColabFold server")
    logger.info("=" * 60)

    total_queries = 0
    seen_sequences = {}  # sequence -> a3m path (avoid querying duplicates)

    for pdb_dir in pdb_dirs:
        pdb_id = pdb_dir.name
        fasta_path = pdb_dir / f"{pdb_id}.fasta"
        sequences = parse_fasta(fasta_path)

        pdb_msa_dir = msa_dir / pdb_id
        logger.info(f"Processing {pdb_id} ({len(sequences)} chains)")

        for chain_id, seq in sequences.items():
            # Skip non-protein (too short)
            if len(seq) < 20:
                logger.info(f"  Skipping chain {chain_id} (too short: {len(seq)} aa)")
                continue

            # Check if we already computed this exact sequence
            if seq in seen_sequences:
                # Reuse existing MSA
                existing_path = seen_sequences[seq]
                pdb_msa_dir.mkdir(parents=True, exist_ok=True)
                target = pdb_msa_dir / f"{chain_id}.a3m"
                if not target.exists():
                    import shutil
                    shutil.copy2(existing_path, target)
                    logger.info(f"  Reused MSA for chain {chain_id} (duplicate sequence)")
                continue

            result = query_colabfold(
                sequence=seq,
                output_dir=pdb_msa_dir,
                chain_id=chain_id,
                user_agent=user_agent,
            )

            if result is not None:
                seen_sequences[seq] = result

            total_queries += 1

            # Rate limiting: be polite to the free API
            if total_queries % 5 == 0:
                logger.info(f"  Rate limiting pause (queried {total_queries} sequences)...")
                time.sleep(5)

    logger.info(f"Computed {total_queries} MSA queries")

    # Step 2: Create alignment arrays
    logger.info("=" * 60)
    logger.info("STEP 2: Creating alignment arrays")
    logger.info("=" * 60)

    create_alignment_arrays(
        msa_dir=msa_dir,
        structure_dir=structure_dir,
        output_dir=alignment_dir,
    )

    logger.info("=" * 60)
    logger.info("MSA COMPUTATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"MSA files:        {msa_dir}")
    logger.info(f"Alignment arrays: {alignment_dir}")
    logger.info(f"Total queries:    {total_queries}")


if __name__ == "__main__":
    main()
