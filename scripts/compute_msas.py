#!/usr/bin/env python
"""Compute MSAs for antibody sequences using the ColabFold MSA server.

Reads FASTA files from preprocessed structures and queries the free
ColabFold API (https://api.colabfold.com) to generate MSAs.

Usage:
    PYTHONPATH="$PWD:$PWD/openfold-3" python -m finetuning.scripts.compute_msas \
        --data-dir ./data/antibody_training
"""

import io
import logging
import shutil
import tarfile
import time
from pathlib import Path
from typing import Annotated

import numpy as np
import requests
import typer

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

    Returns path to the a3m file, or None if failed.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    a3m_path = output_dir / f"{chain_id}.a3m"

    if a3m_path.exists() and a3m_path.stat().st_size > 0:
        logger.info(f"  MSA already exists: {a3m_path}")
        return a3m_path

    headers = {"User-Agent": user_agent}

    for attempt in range(max_retries):
        try:
            data = {"q": f">query\n{sequence}", "mode": "all", "database": "uniref30"}
            response = requests.post(
                f"{host_url}/ticket/msa", data=data, headers=headers, timeout=60,
            )
            response.raise_for_status()
            ticket_id = response.json()["id"]
            logger.info(f"  Submitted job {ticket_id} for chain {chain_id}")

            while True:
                status_response = requests.get(
                    f"{host_url}/ticket/{ticket_id}", headers=headers, timeout=30,
                )
                status_response.raise_for_status()
                status = status_response.json()

                if status["status"] == "COMPLETE":
                    break
                elif status["status"] == "ERROR":
                    logger.warning(f"  MSA server error for chain {chain_id}")
                    break
                time.sleep(2)

            if status["status"] == "COMPLETE":
                result_response = requests.get(
                    f"{host_url}/result/download/{ticket_id}",
                    headers=headers, timeout=120,
                )
                result_response.raise_for_status()

                tar_bytes = io.BytesIO(result_response.content)
                with tarfile.open(fileobj=tar_bytes, mode="r:gz") as tar:
                    for member in tar.getmembers():
                        if member.name.endswith(".a3m") and "uniref" in member.name.lower():
                            f = tar.extractfile(member)
                            if f:
                                a3m_content = f.read().decode("utf-8")
                                with open(a3m_path, "w") as out:
                                    out.write(a3m_content)
                                logger.info(f"  Saved MSA: {a3m_path} ({len(a3m_content.splitlines())} lines)")
                                return a3m_path

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
            logger.warning(f"  Attempt {attempt + 1}/{max_retries} failed for chain {chain_id}: {e}")
            if attempt < max_retries - 1:
                wait = 10 * (attempt + 1)
                logger.info(f"  Retrying in {wait}s...")
                time.sleep(wait)

    logger.error(f"  Failed to get MSA for chain {chain_id} after {max_retries} attempts")
    return None


def fix_a3m_lengths(a3m_path: Path) -> int:
    """Fix a3m: ensure all sequences match query length after removing insertions."""
    with open(a3m_path) as f:
        lines = f.readlines()

    entries = []
    header = None
    seq_lines = []
    for line in lines:
        if line.startswith(">"):
            if header is not None:
                entries.append((header, "".join(seq_lines)))
            header = line
            seq_lines = []
        elif line.strip():
            seq_lines.append(line.strip())
    if header:
        entries.append((header, "".join(seq_lines)))

    if not entries:
        return 0

    query_filtered = "".join(c for c in entries[0][1] if not c.islower())
    query_len = len(query_filtered)

    fixed = 0
    with open(a3m_path, "w") as f:
        for header, seq in entries:
            filtered = "".join(c for c in seq if not c.islower())
            if len(filtered) > query_len:
                filtered = filtered[:query_len]
                fixed += 1
            elif len(filtered) < query_len:
                filtered += "-" * (query_len - len(filtered))
                fixed += 1
            f.write(header)
            f.write(filtered + "\n")

    return fixed


app = typer.Typer()


@app.command()
def main(
    data_dir: Annotated[Path, typer.Option(help="Root data directory (with preprocessed/ subdirectory)")],
    user_agent: Annotated[str, typer.Option(help="User agent for ColabFold API")] = "openfold3-antibody-lora",
    skip_existing: Annotated[bool, typer.Option(help="Skip sequences with existing MSAs")] = True,
):
    """Compute MSAs for preprocessed antibody structures using ColabFold."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    structure_dir = data_dir / "preprocessed" / "structure_files"
    msa_dir = data_dir / "msas"
    alignment_dir = data_dir / "alignments"

    if not structure_dir.exists():
        logger.error(f"Structure directory not found: {structure_dir}")
        logger.error("Run prepare_antibody_data.py first!")
        raise typer.Exit(1)

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
    seen_sequences: dict[str, Path] = {}

    for pdb_dir in pdb_dirs:
        pdb_id = pdb_dir.name
        fasta_path = pdb_dir / f"{pdb_id}.fasta"
        sequences = parse_fasta(fasta_path)

        pdb_msa_dir = msa_dir / pdb_id
        logger.info(f"Processing {pdb_id} ({len(sequences)} chains)")

        for chain_id, seq in sequences.items():
            if len(seq) < 20:
                logger.info(f"  Skipping chain {chain_id} (too short: {len(seq)} aa)")
                continue

            if seq in seen_sequences:
                existing_path = seen_sequences[seq]
                pdb_msa_dir.mkdir(parents=True, exist_ok=True)
                target = pdb_msa_dir / f"{chain_id}.a3m"
                if not target.exists():
                    shutil.copy2(existing_path, target)
                    logger.info(f"  Reused MSA for chain {chain_id} (duplicate sequence)")
                continue

            result = query_colabfold(
                sequence=seq, output_dir=pdb_msa_dir, chain_id=chain_id,
                user_agent=user_agent,
            )

            if result is not None:
                seen_sequences[seq] = result

            total_queries += 1

            if total_queries % 5 == 0:
                logger.info(f"  Rate limiting pause (queried {total_queries} sequences)...")
                time.sleep(5)

    logger.info(f"Computed {total_queries} MSA queries")

    # Step 2: Fix a3m lengths and create alignment directories
    logger.info("=" * 60)
    logger.info("STEP 2: Creating alignment directories for OF3")
    logger.info("=" * 60)

    for pdb_dir in sorted(msa_dir.iterdir()):
        if not pdb_dir.is_dir():
            continue
        pdb_id = pdb_dir.name

        for a3m_file in pdb_dir.glob("*.a3m"):
            chain_id = a3m_file.stem

            n_fixed = fix_a3m_lengths(a3m_file)
            if n_fixed:
                logger.info(f"  Fixed {n_fixed} sequence lengths in {a3m_file.name}")

            rep_id = f"{pdb_id}_{chain_id}"
            rep_dir = alignment_dir / rep_id
            rep_dir.mkdir(parents=True, exist_ok=True)
            target = rep_dir / "colabfold_main.a3m"
            if not target.exists():
                shutil.copy2(a3m_file, target)
            logger.info(f"  Alignment: {rep_dir.name}/colabfold_main.a3m")

    logger.info("=" * 60)
    logger.info("MSA COMPUTATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"MSA files:        {msa_dir}")
    logger.info(f"Alignment dirs:   {alignment_dir}")
    logger.info(f"Total queries:    {total_queries}")


if __name__ == "__main__":
    app()
