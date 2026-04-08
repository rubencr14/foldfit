"""MSA providers: precomputed, ColabFold API, local tools, and single-sequence fallback."""

from __future__ import annotations

import gzip
import logging
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx
import torch

from foldfit.domain.interfaces import MsaPort
from foldfit.domain.value_objects import MsaConfig

logger = logging.getLogger(__name__)

AA_TO_IDX = {
    "A": 0, "R": 1, "N": 2, "D": 3, "C": 4, "Q": 5, "E": 6, "G": 7,
    "H": 8, "I": 9, "L": 10, "K": 11, "M": 12, "F": 13, "P": 14, "S": 15,
    "T": 16, "W": 17, "Y": 18, "V": 19, "X": 20, "-": 21,
}


class MsaProvider(MsaPort):
    """Provides MSA features using configurable backends.

    Backends:
        single: Dummy MSA with just the query sequence.
        precomputed: Load .a3m files from a local directory.
        colabfold: Query ColabFold MMseqs2 server (public or self-hosted).
        local: Run MMseqs2/HHblits/JackHMMER locally against custom databases.
    """

    def __init__(self, config: MsaConfig) -> None:
        self._config = config
        self._cache: dict[str, dict[str, torch.Tensor]] = {}

    def get(self, sequence: str, pdb_id: str) -> dict[str, Any]:
        """Get MSA for a sequence."""
        cache_key = f"{pdb_id}_{len(sequence)}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        backend = self._config.backend
        if backend == "precomputed":
            result = self._from_precomputed(sequence, pdb_id)
        elif backend == "colabfold":
            result = self._from_colabfold(sequence)
        elif backend == "local":
            result = self._from_local(sequence, pdb_id)
        elif backend == "single":
            result = self._single_sequence(sequence)
        else:
            raise ValueError(f"Unknown MSA backend: {backend}")

        self._cache[cache_key] = result
        return result

    # ── Single ────────────────────────────────────────────────────────────

    def _single_sequence(self, sequence: str) -> dict[str, torch.Tensor]:
        """Create a dummy MSA with just the query sequence."""
        seq_len = len(sequence)
        encoded = [AA_TO_IDX.get(aa, 20) for aa in sequence.upper()]
        return {
            "msa": torch.tensor([encoded], dtype=torch.long),
            "deletion_matrix": torch.zeros(1, seq_len, dtype=torch.float32),
            "msa_mask": torch.ones(1, seq_len, dtype=torch.float32),
        }

    # ── Precomputed ───────────────────────────────────────────────────────

    def _from_precomputed(self, sequence: str, pdb_id: str) -> dict[str, torch.Tensor]:
        """Load MSA from precomputed .a3m file."""
        if self._config.msa_dir is None:
            logger.warning("No msa_dir configured, falling back to single sequence")
            return self._single_sequence(sequence)

        msa_dir = Path(self._config.msa_dir)

        for pattern in [f"{pdb_id}.a3m", f"{pdb_id}.a3m.gz", f"{pdb_id.lower()}.a3m"]:
            msa_path = msa_dir / pattern
            if msa_path.exists():
                return self._parse_a3m(msa_path, sequence)

        cache_path = msa_dir / f"{pdb_id}.msa.pt"
        if cache_path.exists():
            return torch.load(cache_path, weights_only=True)

        logger.warning(f"No MSA found for {pdb_id}, using single sequence")
        return self._single_sequence(sequence)

    # ── ColabFold ─────────────────────────────────────────────────────────

    def _from_colabfold(self, sequence: str) -> dict[str, torch.Tensor]:
        """Query ColabFold MMseqs2 server for MSA."""
        server = self._config.colabfold_server.rstrip("/")
        api_url = f"{server}/ticket/msa"

        try:
            response = httpx.post(
                api_url,
                json={"q": f">query\n{sequence}", "mode": "all"},
                timeout=30.0,
            )
            response.raise_for_status()
            ticket = response.json()

            result_url = f"{server}/result/msa/{ticket['id']}"
            for _ in range(120):
                result = httpx.get(result_url, timeout=10.0)
                if result.status_code == 200:
                    return self._parse_a3m_string(result.text, sequence)
                time.sleep(2)

            logger.warning("ColabFold server timeout, using single sequence")
            return self._single_sequence(sequence)

        except httpx.HTTPError as e:
            logger.error(f"ColabFold server error: {e}")
            return self._single_sequence(sequence)

    # ── Local Tool ────────────────────────────────────────────────────────

    def _from_local(self, sequence: str, pdb_id: str) -> dict[str, torch.Tensor]:
        """Run a local alignment tool against configured databases.

        Supports MMseqs2, HHblits, and JackHMMER. Results from multiple
        databases are merged into a single MSA.
        """
        tool = self._config.tool
        databases = self._config.database_paths

        if not databases:
            logger.warning("No database_paths configured for local backend, using single sequence")
            return self._single_sequence(sequence)

        binary = self._resolve_binary(tool)
        if binary is None:
            logger.error(f"{tool} not found. Install it or set tool_binary in config.")
            return self._single_sequence(sequence)

        all_sequences: list[str] = [sequence]  # query always first
        all_deletions: list[list[int]] = [[0] * len(sequence)]

        with tempfile.TemporaryDirectory(prefix="foldfit_msa_") as tmpdir:
            tmpdir_path = Path(tmpdir)
            query_fasta = tmpdir_path / "query.fasta"
            query_fasta.write_text(f">{pdb_id}\n{sequence}\n")

            for db_path in databases:
                try:
                    a3m_result = self._run_tool(tool, binary, query_fasta, db_path, tmpdir_path)
                    if a3m_result:
                        seqs, dels = self._parse_a3m_to_lists(a3m_result)
                        # Skip query (first seq) from additional databases
                        all_sequences.extend(seqs[1:])
                        all_deletions.extend(dels[1:])
                except Exception as e:
                    logger.warning(f"Local MSA failed for db {db_path}: {e}")

        return self._encode_alignment(sequence, all_sequences, all_deletions)

    def _resolve_binary(self, tool: str) -> str | None:
        """Find the tool binary from config or PATH."""
        if self._config.tool_binary:
            p = Path(self._config.tool_binary)
            return str(p) if p.exists() else None

        name_map = {
            "mmseqs2": "mmseqs",
            "hhblits": "hhblits",
            "jackhmmer": "jackhmmer",
        }
        return shutil.which(name_map.get(tool, tool))

    def _run_tool(
        self, tool: str, binary: str, query: Path, db: str, tmpdir: Path,
    ) -> str | None:
        """Run alignment tool and return A3M content."""
        output_a3m = tmpdir / f"result_{Path(db).name}.a3m"
        n_cpu = str(self._config.n_cpu)

        if tool == "mmseqs2":
            result_db = tmpdir / "result_db"
            tmp_dir = tmpdir / "tmp"
            tmp_dir.mkdir(exist_ok=True)
            cmds = [
                [binary, "createdb", str(query), str(tmpdir / "query_db")],
                [binary, "search", str(tmpdir / "query_db"), db, str(result_db),
                 str(tmp_dir), "--threads", n_cpu, "-s", "7.5"],
                [binary, "result2msa", str(tmpdir / "query_db"), db, str(result_db),
                 str(output_a3m), "--msa-format-mode", "6"],
            ]
            for cmd in cmds:
                subprocess.run(cmd, check=True, capture_output=True, timeout=600)

        elif tool == "hhblits":
            cmd = [
                binary, "-i", str(query), "-d", db,
                "-oa3m", str(output_a3m),
                "-n", "2", "-cpu", n_cpu,
                "-maxmem", "4", "-v", "0",
            ]
            subprocess.run(cmd, check=True, capture_output=True, timeout=600)

        elif tool == "jackhmmer":
            sto_out = tmpdir / "result.sto"
            cmd = [
                binary, "--noali", "--cpu", n_cpu,
                "-A", str(sto_out), str(query), db,
            ]
            subprocess.run(cmd, check=True, capture_output=True, timeout=600)
            # Convert STO to A3M via simple parsing
            return self._sto_to_a3m(sto_out, query.read_text().split("\n")[1].strip())

        if output_a3m.exists():
            return output_a3m.read_text()
        return None

    def _sto_to_a3m(self, sto_path: Path, query_seq: str) -> str:
        """Minimal Stockholm to A3M conversion."""
        sequences: dict[str, str] = {}
        if not sto_path.exists():
            return f">query\n{query_seq}\n"

        for line in sto_path.read_text().split("\n"):
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("//"):
                continue
            parts = line.split()
            if len(parts) == 2:
                name, seq = parts
                sequences[name] = sequences.get(name, "") + seq

        lines = [f">query\n{query_seq}"]
        for name, seq in sequences.items():
            # Remove gaps for A3M format
            clean = seq.replace(".", "").replace("-", "-")
            lines.append(f">{name}\n{clean}")

        return "\n".join(lines)

    # ── Parsing ───────────────────────────────────────────────────────────

    def _parse_a3m(self, path: Path, query_sequence: str) -> dict[str, torch.Tensor]:
        """Parse an A3M file into MSA tensors."""
        open_fn = gzip.open if str(path).endswith(".gz") else open
        with open_fn(path, "rt") as f:  # type: ignore[call-overload]
            content = f.read()

        seqs, dels = self._parse_a3m_to_lists(content)
        if not seqs:
            return self._single_sequence(query_sequence)

        return self._encode_alignment(query_sequence, seqs, dels)

    def _parse_a3m_string(self, content: str, query_sequence: str) -> dict[str, torch.Tensor]:
        """Parse A3M content string into MSA tensors."""
        seqs, dels = self._parse_a3m_to_lists(content)
        if not seqs:
            return self._single_sequence(query_sequence)
        return self._encode_alignment(query_sequence, seqs, dels)

    def _parse_a3m_to_lists(self, content: str) -> tuple[list[str], list[list[int]]]:
        """Parse A3M content into sequence and deletion lists."""
        sequences: list[str] = []
        deletion_matrices: list[list[int]] = []
        current_seq = ""

        for line in content.split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith(">"):
                if current_seq:
                    seq, dels = _process_a3m_sequence(current_seq)
                    sequences.append(seq)
                    deletion_matrices.append(dels)
                current_seq = ""
            else:
                current_seq += line

        if current_seq:
            seq, dels = _process_a3m_sequence(current_seq)
            sequences.append(seq)
            deletion_matrices.append(dels)

        return sequences, deletion_matrices

    def _encode_alignment(
        self,
        query_sequence: str,
        sequences: list[str],
        deletion_matrices: list[list[int]],
    ) -> dict[str, torch.Tensor]:
        """Encode parsed alignment into tensors."""
        max_depth = self._config.max_msa_depth
        sequences = sequences[:max_depth]
        deletion_matrices = deletion_matrices[:max_depth]

        if not sequences:
            return self._single_sequence(query_sequence)

        seq_len = len(sequences[0])
        msa_encoded = []
        del_encoded = []

        for seq, dels in zip(sequences, deletion_matrices, strict=True):
            row = [AA_TO_IDX.get(c, 20) for c in seq[:seq_len]]
            while len(row) < seq_len:
                row.append(21)  # gap
            msa_encoded.append(row[:seq_len])

            del_row = dels[:seq_len]
            while len(del_row) < seq_len:
                del_row.append(0)
            del_encoded.append(del_row[:seq_len])

        msa = torch.tensor(msa_encoded, dtype=torch.long)
        deletion_matrix = torch.tensor(del_encoded, dtype=torch.float32)
        msa_mask = torch.ones_like(msa, dtype=torch.float32)

        return {"msa": msa, "deletion_matrix": deletion_matrix, "msa_mask": msa_mask}


def _process_a3m_sequence(raw: str) -> tuple[str, list[int]]:
    """Process an A3M sequence into aligned sequence and deletion counts."""
    sequence = ""
    deletions: list[int] = []
    del_count = 0

    for char in raw:
        if char.islower():
            del_count += 1
        elif char != "\n":
            sequence += char.upper()
            deletions.append(del_count)
            del_count = 0

    return sequence, deletions
