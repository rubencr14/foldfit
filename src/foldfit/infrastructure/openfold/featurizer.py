"""OpenFold featurizer: converts PDB structures to OpenFold-compatible feature dicts.

Parses PDB/mmCIF files using BioPython, builds ground truth labels,
constructs MSA features, and adds the recycling dimension required by OpenFold.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

try:
    from openfold.data import data_transforms as dt
    from openfold.np import residue_constants as rc

    HAS_OPENFOLD = True
except ImportError:
    HAS_OPENFOLD = False

logger = logging.getLogger(__name__)

ATOM_ORDER = None  # populated lazily


def _require_openfold() -> None:
    if not HAS_OPENFOLD:
        raise ImportError(
            "openfold is required for featurization. "
            "Install it from: https://github.com/aqlaboratory/openfold"
        )


def _get_atom_order() -> dict[str, int]:
    global ATOM_ORDER
    if ATOM_ORDER is None:
        ATOM_ORDER = {a: i for i, a in enumerate(rc.atom_types)}
    return ATOM_ORDER


class OpenFoldFeaturizer:
    """Converts PDB files to full OpenFold feature dictionaries.

    Produces all tensors required by the AlphaFold model forward pass,
    including msa_feat, ground truth atom positions, torsion angles,
    and the recycling dimension.

    Args:
        config: OpenFold model config (optional, not used in current impl).
        max_seq_len: Maximum sequence length (crop longer sequences).
        num_msa: Maximum number of MSA sequences.
        num_extra_msa: Maximum number of extra MSA sequences.
    """

    def __init__(
        self,
        config: Any = None,
        max_seq_len: int = 256,
        num_msa: int = 512,
        num_extra_msa: int = 1024,
    ) -> None:
        self._config = config
        self.max_seq_len = max_seq_len
        self.num_msa = num_msa
        self.num_extra_msa = num_extra_msa

    def from_sequence(self, sequence: str) -> dict[str, torch.Tensor]:
        """Build OpenFold features from a raw amino acid sequence (no PDB file).

        Used for inference when only the sequence is known. No ground truth
        atom positions are available, so all GT fields are zero-filled.

        Args:
            sequence: Amino acid sequence string (e.g. 'EVQLVESGG...').

        Returns:
            Feature dict ready for OpenFold forward pass, or empty dict on error.
        """
        _require_openfold()

        seq = sequence.upper()
        L = min(len(seq), self.max_seq_len)
        seq = seq[:L]

        # No ground truth coordinates for sequence-only prediction
        atom_pos = np.zeros((L, 37, 3), dtype=np.float32)
        atom_mask = np.zeros((L, 37), dtype=np.float32)

        protein = self._build_protein_dict(seq, atom_pos, atom_mask)
        self._apply_gt_transforms(protein)
        protein.update(self._build_msa_features(seq, None, L))
        protein.update(self._build_template_placeholders(L))
        protein["seq_length"] = torch.tensor([L], dtype=torch.int64)
        protein["seq_mask"] = torch.ones(L, dtype=torch.float32)
        protein = self._add_recycling_dim(protein)
        return protein

    def from_pdb(
        self,
        pdb_path: str | Path,
        msa_path: str | Path | None = None,
        chain_id: str = "A",
    ) -> dict[str, torch.Tensor]:
        """Convert a PDB file to an OpenFold feature dict.

        Auto-detects antibody chains (H, L) if the requested chain is empty.
        """
        _require_openfold()

        pdb_path = Path(pdb_path)
        seq, atom_pos, atom_mask = self._parse_pdb(pdb_path, chain_id)

        if len(seq) == 0:
            logger.warning(f"No residues found in {pdb_path.name}, skipping")
            return {}

        L = min(len(seq), self.max_seq_len)
        seq = seq[:L]
        atom_pos = atom_pos[:L]
        atom_mask = atom_mask[:L]

        # Build protein dict
        protein = self._build_protein_dict(seq, atom_pos, atom_mask)

        # Apply GT transforms (atom14, backbone frames, chi angles, pseudo-beta)
        self._apply_gt_transforms(protein)

        # MSA features
        protein.update(self._build_msa_features(seq, msa_path, L))

        # Template placeholders
        protein.update(self._build_template_placeholders(L))

        # Metadata
        protein["seq_length"] = torch.tensor([L], dtype=torch.int64)
        protein["seq_mask"] = torch.ones(L, dtype=torch.float32)

        # Add recycling dimension
        protein = self._add_recycling_dim(protein)
        return protein

    # ── PDB Parsing ───────────────────────────────────────────────────────

    def _parse_pdb(
        self, path: Path, chain_id: str
    ) -> tuple[str, np.ndarray, np.ndarray]:
        """Parse PDB and return (sequence, atom_pos[L,37,3], atom_mask[L,37]).

        Auto-detects antibody chains (H, L) if the requested chain yields 0 residues.
        """
        atom_order = _get_atom_order()

        try:
            from Bio.PDB import MMCIFParser, PDBParser

            if path.suffix.lower() in (".cif", ".mmcif"):
                parser = MMCIFParser(QUIET=True)
            else:
                parser = PDBParser(QUIET=True)
            structure = parser.get_structure("s", str(path))
        except ImportError:
            raise ImportError("BioPython is required: pip install biopython")

        # Try requested chain, then antibody chains H/L, then any chain
        for try_chain in [chain_id, "H", "L", "A", "B"]:
            seq, pos, mask = self._extract_chain(structure, try_chain, atom_order)
            if len(seq) > 0:
                if try_chain != chain_id:
                    logger.info(
                        f"{path.name}: chain '{chain_id}' empty, "
                        f"using '{try_chain}' ({len(seq)} residues)"
                    )
                return seq, pos, mask

        logger.warning(f"{path.name}: no residues found in any chain")
        return "", np.zeros((0, 37, 3)), np.zeros((0, 37))

    def _extract_chain(
        self, structure: Any, chain_id: str, atom_order: dict[str, int]
    ) -> tuple[str, np.ndarray, np.ndarray]:
        """Extract a single chain from a BioPython structure."""
        three_to_one = {
            "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
            "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
            "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
            "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
        }

        model = structure[0]
        if chain_id not in [c.id for c in model.get_chains()]:
            return "", np.zeros((0, 37, 3)), np.zeros((0, 37))

        chain = model[chain_id]
        residues = [
            r for r in chain.get_residues()
            if r.get_resname() in three_to_one and r.id[0] == " "
        ]

        if not residues:
            return "", np.zeros((0, 37, 3)), np.zeros((0, 37))

        L = len(residues)
        seq = ""
        atom_pos = np.zeros((L, 37, 3), dtype=np.float32)
        atom_mask = np.zeros((L, 37), dtype=np.float32)

        for i, res in enumerate(residues):
            seq += three_to_one.get(res.get_resname(), "X")
            for atom in res.get_atoms():
                name = atom.get_name()
                if name in atom_order:
                    idx = atom_order[name]
                    atom_pos[i, idx] = atom.get_vector().get_array()
                    atom_mask[i, idx] = 1.0

        return seq, atom_pos, atom_mask

    # ── Protein Dict ──────────────────────────────────────────────────────

    def _build_protein_dict(
        self, seq: str, atom_pos: np.ndarray, atom_mask: np.ndarray
    ) -> dict[str, torch.Tensor]:
        L = len(seq)
        aatype = np.array(
            [rc.restype_order.get(aa, rc.restype_num) for aa in seq],
            dtype=np.int64,
        )
        aatype_t = torch.tensor(aatype, dtype=torch.long)
        target_feat = F.one_hot(aatype_t.clamp(max=21), num_classes=22).float()

        return {
            "aatype": aatype_t,
            "residue_index": torch.arange(L, dtype=torch.long),
            "all_atom_positions": torch.tensor(atom_pos, dtype=torch.float32),
            "all_atom_mask": torch.tensor(atom_mask, dtype=torch.float32),
            "target_feat": target_feat,
        }

    # ── Ground Truth Transforms ───────────────────────────────────────────

    def _apply_gt_transforms(self, protein: dict[str, torch.Tensor]) -> None:
        """Apply OpenFold data transforms to generate GT fields in-place."""
        dt.make_atom14_masks(protein)
        dt.atom37_to_frames(protein)
        dt.get_backbone_frames(protein)
        dt.make_atom14_positions(protein)
        dt.atom37_to_torsion_angles("")(protein)
        dt.get_chi_angles(protein)
        dt.make_pseudo_beta("")(protein)

    # ── MSA Features ──────────────────────────────────────────────────────

    def _build_msa_features(
        self, seq: str, msa_path: Path | None, L: int
    ) -> dict[str, torch.Tensor]:
        """Build MSA-related tensors including the 49-dim msa_feat."""
        if msa_path is not None:
            msa_seqs = self._parse_a3m(msa_path)
        else:
            msa_seqs = [seq]

        msa_seqs = msa_seqs[: self.num_msa]
        N = len(msa_seqs)

        msa = np.zeros((N, L), dtype=np.int64)
        deletion_matrix = np.zeros((N, L), dtype=np.float32)

        for i, s in enumerate(msa_seqs):
            for j, aa in enumerate(s[:L]):
                msa[i, j] = rc.restype_order.get(aa, rc.restype_num)

        msa_t = torch.tensor(msa, dtype=torch.long)
        del_t = torch.tensor(deletion_matrix, dtype=torch.float32)

        # Build msa_feat: [N, L, 49]
        # Channels: 23 (one-hot MSA) + 1 (has_deletion) + 1 (deletion_value) + 24 (padding)
        msa_one_hot = torch.zeros(N, L, 23, dtype=torch.float32)
        for i in range(N):
            for j in range(L):
                idx = int(msa_t[i, j])
                if idx < 23:
                    msa_one_hot[i, j, idx] = 1.0

        msa_feat = torch.zeros(N, L, 49, dtype=torch.float32)
        msa_feat[:, :, :23] = msa_one_hot
        has_del = (del_t > 0).float().unsqueeze(-1)
        del_val = (torch.atan(del_t / 3.0) * (2.0 / np.pi)).unsqueeze(-1)
        msa_feat[:, :, 25:26] = has_del
        msa_feat[:, :, 26:27] = del_val

        msa_mask = torch.ones(N, L, dtype=torch.float32)

        # Extra MSA (same as MSA for single-sequence)
        extra_N = min(N, self.num_extra_msa)

        # Bert mask (for masked MSA loss — all ones = no masking)
        bert_mask = torch.ones(N, L, dtype=torch.float32)

        return {
            "msa": msa_t,
            "deletion_matrix": del_t,
            "msa_feat": msa_feat,
            "msa_mask": msa_mask,
            "true_msa": msa_t.clone(),
            "bert_mask": bert_mask,
            "extra_msa": msa_t[:extra_N],
            "extra_msa_mask": msa_mask[:extra_N],
            "extra_has_deletion": has_del[:extra_N].squeeze(-1),
            "extra_deletion_value": del_val[:extra_N].squeeze(-1),
            "extra_msa_deletion_value": del_val[:extra_N].squeeze(-1),
        }

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
                    # Remove insertions (lowercase)
                    sequences.append("".join(c for c in current if not c.islower()))
                current = ""
            else:
                current += line
        if current:
            sequences.append("".join(c for c in current if not c.islower()))

        return sequences

    # ── Template Placeholders ─────────────────────────────────────────────

    def _build_template_placeholders(self, L: int) -> dict[str, torch.Tensor]:
        """Return zero-filled template tensors (no templates used)."""
        T = 4
        return {
            "template_aatype": torch.zeros(T, L, dtype=torch.long),
            "template_all_atom_positions": torch.zeros(T, L, 37, 3, dtype=torch.float32),
            "template_all_atom_mask": torch.zeros(T, L, 37, dtype=torch.float32),
            "template_mask": torch.zeros(T, dtype=torch.float32),
            "template_pseudo_beta": torch.zeros(T, L, 3, dtype=torch.float32),
            "template_pseudo_beta_mask": torch.zeros(T, L, dtype=torch.float32),
            "template_torsion_angles_sin_cos": torch.zeros(T, L, 7, 2, dtype=torch.float32),
            "template_alt_torsion_angles_sin_cos": torch.zeros(T, L, 7, 2, dtype=torch.float32),
            "template_torsion_angles_mask": torch.zeros(T, L, 7, dtype=torch.float32),
            "template_sum_probs": torch.zeros(T, 1, dtype=torch.float32),
        }

    # ── Recycling Dimension ───────────────────────────────────────────────

    @staticmethod
    def _add_recycling_dim(protein: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Add trailing recycling dimension (R=1) to all tensors.

        OpenFold uses batch[k][..., cycle_no] to extract per-cycle features.
        """
        skip = {"seq_length"}
        out = {}
        for k, v in protein.items():
            if k in skip or not isinstance(v, torch.Tensor):
                out[k] = v
            else:
                out[k] = v.unsqueeze(-1)
        return out
