"""MSA service: orchestrates MSA computation."""

from __future__ import annotations

from typing import Any

from foldfit.domain.interfaces import MsaPort
from foldfit.domain.value_objects import MsaConfig


class MsaService:
    """Provides MSA computation through configurable backends.

    Args:
        msa: MSA port implementation.
    """

    def __init__(self, msa: MsaPort) -> None:
        self._msa = msa

    def compute(self, sequence: str, pdb_id: str) -> dict[str, Any]:
        """Compute MSA for a given sequence.

        Args:
            sequence: Amino acid sequence.
            pdb_id: PDB identifier for caching.

        Returns:
            Dict with msa, deletion_matrix, and msa_mask tensors.
        """
        return self._msa.get(sequence, pdb_id)
