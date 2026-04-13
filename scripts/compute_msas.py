#!/usr/bin/env python
"""Compute MSAs for antibody sequences.

Supports two methods:
  - jackhmmer: local search against UniRef90, Mgnify, UniProt, PDB SeqRes
  - colabfold: free API at https://api.colabfold.com

Usage:
    PYTHONPATH="$PWD:$PWD/openfold-3" python -m finetuning.scripts.compute_msas \
        --config config.yaml
"""

import io
import logging
import os
import shutil
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Annotated, Optional

import requests
import typer
import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def parse_fasta(fasta_path: Path) -> dict[str, str]:
    """Parse a FASTA file into {chain_id: sequence} dict."""
    sequences = {}
    current_id = None
    current_seq: list[str] = []

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


# ---------------------------------------------------------------------------
# Jackhmmer MSA computation
# ---------------------------------------------------------------------------
def run_jackhmmer(
    sequence: str,
    output_dir: Path,
    binary_path: str,
    database_path: str,
    database_name: str,
    n_cpu: int = 8,
    n_iter: int = 1,
    e_value: float = 0.0001,
    max_sequences: int | None = None,
) -> Path | None:
    """Run jackhmmer search for a single sequence against a single database.

    Returns path to the output .sto file, or None on failure.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    sto_path = output_dir / f"{database_name}_hits.sto"

    if sto_path.exists() and sto_path.stat().st_size > 0:
        logger.info(f"    {database_name}: already exists, skipping")
        return sto_path

    with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as f:
        f.write(f">query\n{sequence}\n")
        fasta_path = f.name

    try:
        cmd = [
            binary_path,
            "-o", "/dev/null",
            "-A", str(sto_path),
            "--noali",
            "--F1", "0.0005",
            "--F2", "0.00005",
            "--F3", "0.0000005",
            "--incE", str(e_value),
            "-E", str(e_value),
            "--cpu", str(n_cpu),
            "-N", str(n_iter),
            fasta_path,
            database_path,
        ]

        logger.info(f"    {database_name}: running jackhmmer...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

        if result.returncode != 0:
            logger.warning(f"    {database_name}: jackhmmer failed: {result.stderr[:200]}")
            return None

        if sto_path.exists():
            logger.info(f"    {database_name}: done -> {sto_path}")
            return sto_path

        return None
    except subprocess.TimeoutExpired:
        logger.warning(f"    {database_name}: jackhmmer timed out (1h limit)")
        return None
    finally:
        os.unlink(fasta_path)


def compute_msas_jackhmmer(
    structure_dir: Path,
    alignment_dir: Path,
    config: dict,
) -> int:
    """Compute MSAs using local jackhmmer searches.

    Runs jackhmmer against each configured database for each protein chain.
    Output files are named to match OpenFold3's max_seq_counts keys:
      uniref90_hits.sto, mgnify_hits.sto, uniprot_hits.sto, etc.
    """
    jackhmmer_cfg = config["msa"]["jackhmmer"]
    binary_path = jackhmmer_cfg["binary_path"]
    n_cpu = jackhmmer_cfg.get("n_cpu", 8)
    n_iter = jackhmmer_cfg.get("n_iter", 1)
    e_value = jackhmmer_cfg.get("e_value", 0.0001)
    databases = jackhmmer_cfg["databases"]

    pdb_dirs = sorted(
        d for d in structure_dir.iterdir()
        if d.is_dir() and (d / f"{d.name}.fasta").exists()
    )

    total_searches = 0
    seen_sequences: dict[str, Path] = {}

    for pdb_dir in pdb_dirs:
        pdb_id = pdb_dir.name
        fasta_path = pdb_dir / f"{pdb_id}.fasta"
        sequences = parse_fasta(fasta_path)

        logger.info(f"Processing {pdb_id} ({len(sequences)} chains)")

        for chain_id, seq in sequences.items():
            if len(seq) < 20:
                logger.info(f"  Skipping chain {chain_id} (too short: {len(seq)} aa)")
                continue

            rep_id = f"{pdb_id}_{chain_id}"
            chain_alignment_dir = alignment_dir / rep_id

            if seq in seen_sequences:
                src_dir = seen_sequences[seq]
                if not chain_alignment_dir.exists():
                    shutil.copytree(src_dir, chain_alignment_dir)
                    logger.info(f"  chain {chain_id}: reused MSAs from {src_dir.name}")
                continue

            logger.info(f"  chain {chain_id} ({len(seq)} aa):")

            for db_name, db_config in databases.items():
                db_path = db_config["path"]
                max_seq = db_config.get("max_sequences")

                if not Path(db_path).exists():
                    logger.warning(f"    {db_name}: database not found at {db_path}, skipping")
                    continue

                run_jackhmmer(
                    sequence=seq,
                    output_dir=chain_alignment_dir,
                    binary_path=binary_path,
                    database_path=db_path,
                    database_name=db_name,
                    n_cpu=n_cpu,
                    n_iter=n_iter,
                    e_value=e_value,
                    max_sequences=max_seq,
                )
                total_searches += 1

            seen_sequences[seq] = chain_alignment_dir

    return total_searches


# ---------------------------------------------------------------------------
# ColabFold MSA computation
# ---------------------------------------------------------------------------
def query_colabfold(
    sequence: str,
    output_dir: Path,
    chain_id: str,
    host_url: str = "https://api.colabfold.com",
    user_agent: str = "openfold3-lora",
    max_retries: int = 5,
) -> Path | None:
    """Query ColabFold MSA server for a single sequence."""
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
                if status["status"] in ("COMPLETE", "ERROR"):
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
                        if member.name.endswith(".a3m"):
                            f = tar.extractfile(member)
                            if f:
                                a3m_content = f.read().decode("utf-8")
                                with open(a3m_path, "w") as out:
                                    out.write(a3m_content)
                                logger.info(f"  Saved MSA: {a3m_path} ({len(a3m_content.splitlines())} lines)")
                                return a3m_path

            return None

        except requests.exceptions.RequestException as e:
            logger.warning(f"  Attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(10 * (attempt + 1))

    return None


def fix_a3m_lengths(a3m_path: Path) -> int:
    """Fix a3m: ensure all sequences match query length after removing insertions."""
    with open(a3m_path) as f:
        lines = f.readlines()

    entries = []
    header = None
    seq_lines: list[str] = []
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
            if len(filtered) != query_len:
                filtered = filtered[:query_len] if len(filtered) > query_len else filtered + "-" * (query_len - len(filtered))
                fixed += 1
            f.write(header)
            f.write(filtered + "\n")

    return fixed


# ---------------------------------------------------------------------------
# MMseqs2 MSA computation
# ---------------------------------------------------------------------------
def run_mmseqs(
    sequence: str,
    output_dir: Path,
    database_name: str,
    binary_path: str,
    database_path: str,
    n_cpu: int = 8,
    sensitivity: float = 7.5,
    max_seqs: int = 10000,
    e_value: float = 0.001,
) -> Path | None:
    """Run MMseqs2 search for a single sequence against a database.

    Returns path to the output .a3m file, or None on failure.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    a3m_path = output_dir / f"{database_name}_hits.a3m"

    if a3m_path.exists() and a3m_path.stat().st_size > 0:
        logger.info(f"    {database_name}: already exists, skipping")
        return a3m_path

    with tempfile.TemporaryDirectory() as tmpdir:
        query_fasta = Path(tmpdir) / "query.fasta"
        query_fasta.write_text(f">query\n{sequence}\n")

        query_db = Path(tmpdir) / "query_db"
        result_db = Path(tmpdir) / "result_db"
        result_a3m = Path(tmpdir) / "result.a3m"
        tmp_dir = Path(tmpdir) / "tmp"
        tmp_dir.mkdir()

        try:
            # Step 1: Create query database
            subprocess.run(
                [binary_path, "createdb", str(query_fasta), str(query_db)],
                capture_output=True, text=True, check=True,
            )

            # Step 2: Search
            search_cmd = [
                binary_path, "search",
                str(query_db), database_path, str(result_db), str(tmp_dir),
                "--threads", str(n_cpu),
                "-s", str(sensitivity),
                "--max-seqs", str(max_seqs),
                "-e", str(e_value),
            ]
            logger.info(f"    {database_name}: running mmseqs search...")
            result = subprocess.run(
                search_cmd, capture_output=True, text=True, timeout=3600,
            )
            if result.returncode != 0:
                logger.warning(f"    {database_name}: mmseqs search failed: {result.stderr[:200]}")
                return None

            # Step 3: Convert to a3m
            subprocess.run(
                [binary_path, "result2msa",
                 str(query_db), database_path, str(result_db), str(result_a3m),
                 "--msa-format-mode", "6"],
                capture_output=True, text=True, check=True,
            )

            if result_a3m.exists() and result_a3m.stat().st_size > 0:
                shutil.copy2(result_a3m, a3m_path)
                n_seqs = sum(1 for line in open(a3m_path) if line.startswith(">"))
                logger.info(f"    {database_name}: done -> {n_seqs} sequences")
                return a3m_path

            return None

        except subprocess.TimeoutExpired:
            logger.warning(f"    {database_name}: mmseqs timed out (1h limit)")
            return None
        except subprocess.CalledProcessError as e:
            logger.warning(f"    {database_name}: mmseqs failed: {e}")
            return None


def compute_msas_mmseqs(
    structure_dir: Path,
    alignment_dir: Path,
    config: dict,
) -> int:
    """Compute MSAs using local MMseqs2 searches.

    Faster than jackhmmer with comparable sensitivity at high -s values.
    Output files named {db_name}_hits.a3m in each alignment directory.
    """
    mmseqs_cfg = config["msa"]["mmseqs"]
    binary_path = mmseqs_cfg["binary_path"]
    n_cpu = mmseqs_cfg.get("n_cpu", 8)
    sensitivity = mmseqs_cfg.get("sensitivity", 7.5)
    max_seqs = mmseqs_cfg.get("max_seqs", 10000)
    e_value = mmseqs_cfg.get("e_value", 0.001)
    databases = mmseqs_cfg["databases"]

    pdb_dirs = sorted(
        d for d in structure_dir.iterdir()
        if d.is_dir() and (d / f"{d.name}.fasta").exists()
    )

    total_searches = 0
    seen_sequences: dict[str, Path] = {}

    for pdb_dir in pdb_dirs:
        pdb_id = pdb_dir.name
        sequences = parse_fasta(pdb_dir / f"{pdb_id}.fasta")
        logger.info(f"Processing {pdb_id} ({len(sequences)} chains)")

        for chain_id, seq in sequences.items():
            if len(seq) < 20:
                logger.info(f"  Skipping chain {chain_id} (too short: {len(seq)} aa)")
                continue

            rep_id = f"{pdb_id}_{chain_id}"
            chain_alignment_dir = alignment_dir / rep_id

            if seq in seen_sequences:
                src_dir = seen_sequences[seq]
                if not chain_alignment_dir.exists():
                    shutil.copytree(src_dir, chain_alignment_dir)
                    logger.info(f"  chain {chain_id}: reused MSAs from {src_dir.name}")
                continue

            logger.info(f"  chain {chain_id} ({len(seq)} aa):")

            for db_name, db_config in databases.items():
                db_path = db_config["path"]

                if not Path(db_path).exists() and not Path(db_path + ".index").exists():
                    logger.warning(f"    {db_name}: database not found at {db_path}, skipping")
                    continue

                # output_name must match OF3's max_seq_counts keys
                out_name = db_config.get("output_name", f"{db_name}_hits")

                result = run_mmseqs(
                    sequence=seq,
                    output_dir=chain_alignment_dir,
                    database_name=out_name,
                    binary_path=binary_path,
                    database_path=db_path,
                    n_cpu=n_cpu,
                    sensitivity=sensitivity,
                    max_seqs=max_seqs,
                    e_value=e_value,
                )

                if result is not None:
                    fix_a3m_lengths(result)

                total_searches += 1

            seen_sequences[seq] = chain_alignment_dir

    return total_searches


# ---------------------------------------------------------------------------
# ColabFold MSA computation
# ---------------------------------------------------------------------------
def compute_msas_colabfold(
    structure_dir: Path,
    alignment_dir: Path,
    msa_dir: Path,
    config: dict,
) -> int:
    """Compute MSAs using the free ColabFold API."""
    colabfold_cfg = config["msa"]["colabfold"]
    host_url = colabfold_cfg.get("host_url", "https://api.colabfold.com")
    user_agent = colabfold_cfg.get("user_agent", "openfold3-lora")

    pdb_dirs = sorted(
        d for d in structure_dir.iterdir()
        if d.is_dir() and (d / f"{d.name}.fasta").exists()
    )

    total_queries = 0
    seen_sequences: dict[str, Path] = {}

    for pdb_dir in pdb_dirs:
        pdb_id = pdb_dir.name
        sequences = parse_fasta(pdb_dir / f"{pdb_id}.fasta")
        logger.info(f"Processing {pdb_id} ({len(sequences)} chains)")

        for chain_id, seq in sequences.items():
            if len(seq) < 20:
                continue

            pdb_msa_dir = msa_dir / pdb_id
            rep_id = f"{pdb_id}_{chain_id}"

            if seq in seen_sequences:
                src = seen_sequences[seq]
                rep_dir = alignment_dir / rep_id
                if not rep_dir.exists():
                    shutil.copytree(src, rep_dir)
                    logger.info(f"  chain {chain_id}: reused from {src.name}")
                continue

            result = query_colabfold(
                sequence=seq, output_dir=pdb_msa_dir, chain_id=chain_id,
                host_url=host_url, user_agent=user_agent,
            )

            if result is not None:
                fix_a3m_lengths(result)
                rep_dir = alignment_dir / rep_id
                rep_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(result, rep_dir / "colabfold_main.a3m")
                seen_sequences[seq] = rep_dir

            total_queries += 1
            if total_queries % 5 == 0:
                logger.info(f"  Rate limiting pause ({total_queries} queries)...")
                time.sleep(5)

    return total_queries


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
app = typer.Typer()


@app.command()
def main(
    config: Annotated[Path, typer.Option(help="Path to config.yaml")],
    data_dir: Annotated[Optional[Path], typer.Option(help="Override data directory from config")] = None,
):
    """Compute MSAs for preprocessed antibody structures.

    Uses jackhmmer (local databases) or ColabFold (free API) based on config.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    cfg = load_config(config)

    data_path = Path(data_dir) if data_dir else Path(cfg["paths"]["data_dir"])
    structure_dir = data_path / "preprocessed" / "structure_files"
    alignment_dir = data_path / "alignments"
    msa_dir = data_path / "msas"

    if not structure_dir.exists():
        logger.error(f"Structure directory not found: {structure_dir}")
        logger.error("Run prepare_antibody_data first!")
        raise typer.Exit(1)

    method = cfg["msa"]["method"]
    logger.info(f"MSA method: {method}")
    logger.info("=" * 60)

    if method == "jackhmmer":
        total = compute_msas_jackhmmer(structure_dir, alignment_dir, cfg)
        logger.info(f"Completed {total} jackhmmer searches")
    elif method == "mmseqs":
        total = compute_msas_mmseqs(structure_dir, alignment_dir, cfg)
        logger.info(f"Completed {total} MMseqs2 searches")
    elif method == "colabfold":
        total = compute_msas_colabfold(structure_dir, alignment_dir, msa_dir, cfg)
        logger.info(f"Completed {total} ColabFold queries")
    else:
        logger.error(f"Unknown MSA method: {method}. Use 'jackhmmer', 'mmseqs', or 'colabfold'")
        raise typer.Exit(1)

    logger.info("=" * 60)
    logger.info("MSA COMPUTATION COMPLETE")
    logger.info(f"Alignments: {alignment_dir}")


if __name__ == "__main__":
    app()
