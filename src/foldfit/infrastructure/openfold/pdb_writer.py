"""Convert OpenFold output coordinates to PDB format strings."""

from __future__ import annotations

import numpy as np

try:
    from openfold.np import residue_constants as rc

    HAS_OPENFOLD = True
except ImportError:
    HAS_OPENFOLD = False

# Mapping from index to 3-letter amino acid code
IDX_TO_RESNAME = {
    0: "ALA", 1: "ARG", 2: "ASN", 3: "ASP", 4: "CYS",
    5: "GLN", 6: "GLU", 7: "GLY", 8: "HIS", 9: "ILE",
    10: "LEU", 11: "LYS", 12: "MET", 13: "PHE", 14: "PRO",
    15: "SER", 16: "THR", 17: "TRP", 18: "TYR", 19: "VAL",
    20: "UNK",
}

ONE_TO_THREE = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}


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
    if HAS_OPENFOLD:
        atom_types = rc.atom_types
    else:
        atom_types = [
            "N", "CA", "C", "CB", "O", "CG", "CG1", "CG2", "CD", "CD1",
            "CD2", "CE", "CE1", "CE2", "CE3", "CZ", "CZ2", "CZ3", "CH2",
            "NE", "NE1", "NE2", "ND1", "ND2", "NZ", "NH1", "NH2", "OD1",
            "OD2", "OE1", "OE2", "OG", "OG1", "OH", "OXT", "SD", "SG",
        ]

    L = len(sequence)
    lines = []
    atom_num = 1

    for res_idx in range(L):
        resname = ONE_TO_THREE.get(sequence[res_idx], "UNK")
        bfactor = float(plddt[res_idx]) if plddt is not None else 0.0

        for atom_idx, atom_name in enumerate(atom_types):
            if atom_idx >= atom_positions.shape[1]:
                break

            x, y, z = atom_positions[res_idx, atom_idx]

            # Skip atoms with zero coordinates (not predicted)
            if x == 0.0 and y == 0.0 and z == 0.0:
                continue

            # PDB ATOM record format
            element = atom_name[0]
            lines.append(
                f"ATOM  {atom_num:5d} {atom_name:<4s} {resname:>3s} {chain_id}{res_idx + 1:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{bfactor:6.2f}          {element:>2s}  "
            )
            atom_num += 1

    lines.append("TER")
    lines.append("END")
    return "\n".join(lines)
