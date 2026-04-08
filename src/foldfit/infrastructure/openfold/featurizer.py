"""OpenFold featurizer: converts PDB structures to OpenFold-compatible feature dicts.

Delegates PDB parsing to openfold.np.protein and feature construction to
openfold.data.data_transforms, keeping only the orchestration logic.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from openfold.data import data_transforms as dt
from openfold.np import protein as protein_utils
from openfold.np import residue_constants as rc

logger = logging.getLogger(__name__)


class OpenFoldFeaturizer:
    """Converts PDB files to full OpenFold feature dictionaries.

    Args:
        max_seq_len: Maximum sequence length (crop longer sequences).
        num_msa: Maximum number of MSA sequences.
        num_extra_msa: Maximum number of extra MSA sequences.
        num_templates: Number of template slots (zero-filled).
    """

    def __init__(
        self,
        config: Any = None,
        max_seq_len: int = 256,
        num_msa: int = 512,
        num_extra_msa: int = 1024,
        num_templates: int = 4,
        masked_msa_replace_fraction: float = 0.15,
        training: bool = True,
    ) -> None:
        self.max_seq_len = max_seq_len
        self.num_msa = num_msa
        self.num_extra_msa = num_extra_msa
        self.num_templates = num_templates
        self.masked_msa_replace_fraction = masked_msa_replace_fraction
        self.training = training

    # ── Public API ────────────────────────────────────────────────────────

    def from_sequence(self, sequence: str) -> dict[str, torch.Tensor]:
        """Build features from a raw amino acid sequence (no PDB).

        Used for inference when only the sequence is known.
        """
        seq = sequence.upper()[:self.max_seq_len]
        L = len(seq)

        aatype = np.array(
            [rc.restype_order.get(aa, rc.restype_num) for aa in seq],
            dtype=np.int64,
        )
        atom_pos = np.zeros((L, 37, 3), dtype=np.float32)
        atom_mask = np.zeros((L, 37), dtype=np.float32)

        features = self._assemble_features(
            aatype, atom_pos, atom_mask, L, seq,
            msa_data=None, msa_path=None,
        )
        return features

    def from_pdb(
        self,
        pdb_path: str | Path,
        msa_path: str | Path | None = None,
        msa_data: dict[str, torch.Tensor] | None = None,
        chain_id: str = "A",
    ) -> dict[str, torch.Tensor]:
        """Convert a PDB file to an OpenFold feature dict.

        Auto-detects antibody chains (H, L) if the requested chain is empty.
        """
        pdb_path = Path(pdb_path)
        pdb_str = pdb_path.read_text()

        # Try requested chain, then antibody chains, then any
        for try_chain in [chain_id, "H", "L", "A", "B", None]:
            try:
                prot = protein_utils.from_pdb_string(pdb_str, chain_id=try_chain)
                if len(prot.aatype) > 0:
                    if try_chain != chain_id:
                        logger.info(
                            f"{pdb_path.name}: chain '{chain_id}' empty, "
                            f"using '{try_chain}' ({len(prot.aatype)} residues)"
                        )
                    break
            except Exception:
                continue
        else:
            logger.warning(f"{pdb_path.name}: no residues found in any chain")
            return {}

        # Crop to max_seq_len
        L = min(len(prot.aatype), self.max_seq_len)
        aatype = prot.aatype[:L]
        atom_pos = prot.atom_positions[:L]
        atom_mask = prot.atom_mask[:L]
        seq = "".join(rc.restypes_with_x[a] for a in aatype)

        features = self._assemble_features(
            aatype, atom_pos, atom_mask, L, seq,
            msa_data=msa_data, msa_path=msa_path,
        )
        return features

    def _parse_pdb(
        self, path: Path, chain_id: str
    ) -> tuple[str, np.ndarray, np.ndarray]:
        """Parse PDB and return (sequence, atom_pos, atom_mask).

        Thin wrapper kept for backward compat with StructureDataset.
        """
        pdb_str = path.read_text()
        for try_chain in [chain_id, "H", "L", "A", "B", None]:
            try:
                prot = protein_utils.from_pdb_string(pdb_str, chain_id=try_chain)
                if len(prot.aatype) > 0:
                    seq = "".join(rc.restypes_with_x[a] for a in prot.aatype)
                    return seq, prot.atom_positions, prot.atom_mask
            except Exception:
                continue
        return "", np.zeros((0, 37, 3)), np.zeros((0, 37))

    # ── Core Assembly ─────────────────────────────────────────────────────

    def _assemble_features(
        self,
        aatype: np.ndarray,
        atom_pos: np.ndarray,
        atom_mask: np.ndarray,
        L: int,
        seq: str,
        msa_data: dict[str, torch.Tensor] | None,
        msa_path: str | Path | None,
    ) -> dict[str, torch.Tensor]:
        """Build the complete feature dict from raw arrays.

        Uses OpenFold's native data transforms for the full MSA pipeline:
        raw MSA → masks → HHblits profile → sample → BERT masking → msa_feat.
        """
        import ml_collections

        # Base protein features
        protein: dict[str, Any] = {
            "aatype": torch.tensor(aatype, dtype=torch.long),
            "residue_index": torch.arange(L, dtype=torch.long),
            "all_atom_positions": torch.tensor(atom_pos, dtype=torch.float32),
            "all_atom_mask": torch.tensor(atom_mask, dtype=torch.float32),
            "between_segment_residues": torch.zeros(L, dtype=torch.long),
        }

        # GT transforms via OpenFold native functions
        dt.make_atom14_masks(protein)
        dt.atom37_to_frames(protein)
        dt.get_backbone_frames(protein)
        dt.make_atom14_positions(protein)
        dt.atom37_to_torsion_angles("")(protein)
        dt.get_chi_angles(protein)
        dt.make_pseudo_beta("")(protein)

        # ── MSA pipeline (OpenFold native) ────────────────────────────────
        # 1. Raw MSA tensors
        self._add_msa_tensors(protein, seq, L, msa_data, msa_path)

        # 2. Standard masks
        dt.make_msa_mask(protein)

        # 3. HHblits profile (mean of one-hot MSA — needed for masked MSA)
        dt.make_hhblits_profile(protein)

        # 4. Sample MSA + separate extra sequences
        dt.sample_msa(self.num_msa, keep_extra=True)(protein)
        dt.crop_extra_msa(self.num_extra_msa)(protein)

        # 5. BERT-style masking for masked MSA loss (training only)
        if self.training:
            masked_msa_config = ml_collections.ConfigDict({
                "uniform_prob": 0.1,
                "profile_prob": 0.1,
                "same_prob": 0.1,
            })
            dt.make_masked_msa(
                masked_msa_config,
                self.masked_msa_replace_fraction,
                seed=None,
            )(protein)
        else:
            protein["true_msa"] = protein["msa"].clone()
            protein["bert_mask"] = torch.zeros_like(
                protein["msa"], dtype=torch.float32
            )

        # 6. Build msa_feat and target_feat
        dt.make_msa_feat(protein)

        # Template placeholders
        self._add_template_placeholders(protein, L)

        # Metadata
        protein["seq_length"] = torch.tensor([L], dtype=torch.int64)
        dt.make_seq_mask(protein)

        # Recycling dimension
        return self._add_recycling_dim(protein)

    # ── MSA ───────────────────────────────────────────────────────────────

    def _add_msa_tensors(
        self,
        protein: dict[str, Any],
        seq: str,
        L: int,
        msa_data: dict[str, torch.Tensor] | None,
        msa_path: str | Path | None,
    ) -> None:
        """Populate raw MSA tensors into the protein dict.

        Only sets msa + deletion_matrix. The rest (masks, sampling,
        BERT masking, extra_msa) is handled by the native pipeline.
        """
        if msa_data is not None:
            protein["msa"] = msa_data["msa"][:, :L]
            protein["deletion_matrix"] = msa_data["deletion_matrix"][:, :L]
        elif msa_path is not None:
            msa_seqs = self._parse_a3m(Path(msa_path))
            msa_t, del_t = self._encode_msa_seqs(msa_seqs, L)
            protein["msa"] = msa_t
            protein["deletion_matrix"] = del_t
        else:
            encoded = [rc.restype_order.get(aa, rc.restype_num) for aa in seq[:L]]
            protein["msa"] = torch.tensor([encoded], dtype=torch.long)
            protein["deletion_matrix"] = torch.zeros(1, L, dtype=torch.float32)

    def _encode_msa_seqs(
        self, msa_seqs: list[str], L: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode MSA sequences to integer tensor."""
        N = len(msa_seqs)
        msa = np.zeros((N, L), dtype=np.int64)
        for i, s in enumerate(msa_seqs):
            for j, aa in enumerate(s[:L]):
                msa[i, j] = rc.restype_order.get(aa, rc.restype_num)
        return torch.tensor(msa, dtype=torch.long), torch.zeros(N, L, dtype=torch.float32)

    def _parse_a3m(self, path: Path) -> list[str]:
        """Parse an A3M file into a list of aligned sequences."""
        import gzip

        open_fn = gzip.open if str(path).endswith(".gz") else open
        with open_fn(path, "rt") as f:
            content = f.read()

        sequences: list[str] = []
        current = ""
        for line in content.split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith(">"):
                if current:
                    sequences.append("".join(c for c in current if not c.islower()))
                current = ""
            else:
                current += line
        if current:
            sequences.append("".join(c for c in current if not c.islower()))

        return sequences

    # ── Templates ─────────────────────────────────────────────────────────

    def _add_template_placeholders(self, protein: dict[str, Any], L: int) -> None:
        """Add zero-filled template tensors (no templates used)."""
        T = self.num_templates
        protein["template_aatype"] = torch.zeros(T, L, dtype=torch.long)
        protein["template_all_atom_positions"] = torch.zeros(T, L, 37, 3, dtype=torch.float32)
        protein["template_all_atom_mask"] = torch.zeros(T, L, 37, dtype=torch.float32)
        protein["template_mask"] = torch.zeros(T, dtype=torch.float32)
        protein["template_pseudo_beta"] = torch.zeros(T, L, 3, dtype=torch.float32)
        protein["template_pseudo_beta_mask"] = torch.zeros(T, L, dtype=torch.float32)
        protein["template_torsion_angles_sin_cos"] = torch.zeros(T, L, 7, 2, dtype=torch.float32)
        protein["template_alt_torsion_angles_sin_cos"] = torch.zeros(T, L, 7, 2, dtype=torch.float32)
        protein["template_torsion_angles_mask"] = torch.zeros(T, L, 7, dtype=torch.float32)
        protein["template_sum_probs"] = torch.zeros(T, 1, dtype=torch.float32)

    # ── Recycling Dimension ───────────────────────────────────────────────

    @staticmethod
    def _add_recycling_dim(protein: dict[str, Any]) -> dict[str, Any]:
        """Add trailing recycling dimension (R=1) to all tensors."""
        skip = {"seq_length"}
        out = {}
        for k, v in protein.items():
            if k in skip or not isinstance(v, torch.Tensor):
                out[k] = v
            else:
                out[k] = v.unsqueeze(-1)
        return out
