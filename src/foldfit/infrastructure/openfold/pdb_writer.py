"""Convert OpenFold output coordinates to PDB format strings.

Delegates to openfold.np.protein for PDB formatting.
"""

from __future__ import annotations

import numpy as np

from openfold.np import protein as protein_utils
from openfold.np import residue_constants as rc


def coords_to_pdb(
    sequence: str,
    atom_positions: np.ndarray,
    plddt: np.ndarray | None = None,
    chain_id: str = "A",
) -> str:
    """Convert predicted coordinates to a PDB format string.

    Args:
        sequence: Amino acid sequence.
        atom_positions: Atom coordinates [L, 37, 3].
        plddt: Per-residue pLDDT confidence [L] (written as B-factor).
        chain_id: Chain identifier.

    Returns:
        PDB file content as a string.
    """
    L = len(sequence)
    aatype = np.array(
        [rc.restype_order.get(aa, rc.restype_num) for aa in sequence],
        dtype=np.int64,
    )

    # B-factors: broadcast pLDDT per-residue to all atoms [L, 37]
    if plddt is not None:
        b_factors = np.repeat(plddt[:, None], 37, axis=1)
    else:
        b_factors = np.zeros((L, 37), dtype=np.float32)

    prot = protein_utils.Protein(
        atom_positions=atom_positions,
        aatype=aatype,
        atom_mask=(atom_positions.sum(-1) != 0).astype(np.float32),
        residue_index=np.arange(L, dtype=np.int32),
        b_factors=b_factors,
        chain_index=np.zeros(L, dtype=np.int32),
    )
    return protein_utils.to_pdb(prot)
