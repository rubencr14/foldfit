"""CDR (Complementarity-Determining Region) annotation for antibody sequences."""

import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class CDRScheme(Enum):
    """Antibody numbering scheme for CDR definition."""

    IMGT = "imgt"
    CHOTHIA = "chothia"
    KABAT = "kabat"


@dataclass(frozen=True)
class CDRRegion:
    """A single CDR region annotation.

    Attributes:
        name: CDR name (e.g., "CDR-H1", "CDR-H2", "CDR-H3",
            "CDR-L1", "CDR-L2", "CDR-L3").
        start_idx: Start residue index (0-based, inclusive).
        end_idx: End residue index (0-based, exclusive).
        sequence: Amino acid sequence of the CDR region.
        chain_id: PDB chain identifier.
    """

    name: str
    start_idx: int
    end_idx: int
    sequence: str
    chain_id: str

    @property
    def length(self) -> int:
        return self.end_idx - self.start_idx


# IMGT CDR boundaries in IMGT numbering
# These are the standard IMGT definition boundaries.
# Format: (start_imgt_number, end_imgt_number) inclusive
_IMGT_CDR_BOUNDARIES = {
    "heavy": {
        "CDR-H1": (27, 38),
        "CDR-H2": (56, 65),
        "CDR-H3": (105, 117),
    },
    "light": {
        "CDR-L1": (27, 38),
        "CDR-L2": (56, 65),
        "CDR-L3": (105, 117),
    },
}

# Chothia CDR boundaries in Chothia numbering
_CHOTHIA_CDR_BOUNDARIES = {
    "heavy": {
        "CDR-H1": (26, 32),
        "CDR-H2": (52, 56),
        "CDR-H3": (95, 102),
    },
    "light": {
        "CDR-L1": (24, 34),
        "CDR-L2": (50, 56),
        "CDR-L3": (89, 97),
    },
}

# Kabat CDR boundaries in Kabat numbering
_KABAT_CDR_BOUNDARIES = {
    "heavy": {
        "CDR-H1": (31, 35),
        "CDR-H2": (50, 65),
        "CDR-H3": (95, 102),
    },
    "light": {
        "CDR-L1": (24, 34),
        "CDR-L2": (50, 56),
        "CDR-L3": (89, 97),
    },
}

_SCHEME_BOUNDARIES = {
    CDRScheme.IMGT: _IMGT_CDR_BOUNDARIES,
    CDRScheme.CHOTHIA: _CHOTHIA_CDR_BOUNDARIES,
    CDRScheme.KABAT: _KABAT_CDR_BOUNDARIES,
}


def _try_import_anarci():
    """Attempt to import ANARCI for antibody numbering."""
    try:
        from anarci import anarci as run_anarci
        return run_anarci
    except ImportError:
        return None


class CDRAnnotator:
    """Annotates CDR regions on antibody sequences.

    Uses ANARCI (Antibody Numbering And Receptor ClassIfication) when
    available for accurate numbering. Falls back to a simple
    positional heuristic when ANARCI is not installed.

    Args:
        scheme: CDR numbering scheme to use.
    """

    def __init__(self, scheme: CDRScheme = CDRScheme.IMGT):
        self.scheme = scheme
        self._anarci_fn = _try_import_anarci()
        if self._anarci_fn is None:
            logger.info(
                "ANARCI not available, using positional heuristic for CDR annotation. "
                "Install ANARCI for accurate numbering: pip install anarci"
            )

    def annotate(
        self, sequence: str, chain_type: str, chain_id: str = "A"
    ) -> list[CDRRegion]:
        """Annotate CDR regions for a single antibody chain.

        Args:
            sequence: Amino acid sequence of the chain.
            chain_type: Either "heavy" or "light".
            chain_id: PDB chain identifier.

        Returns:
            List of CDRRegion annotations found in the sequence.
        """
        if chain_type not in ("heavy", "light"):
            raise ValueError(f"chain_type must be 'heavy' or 'light', got '{chain_type}'")

        if self._anarci_fn is not None:
            return self._annotate_with_anarci(sequence, chain_type, chain_id)
        return self._annotate_with_heuristic(sequence, chain_type, chain_id)

    def annotate_structure(
        self, chains: dict[str, dict]
    ) -> dict[str, list[CDRRegion]]:
        """Annotate CDR regions for all chains in a structure.

        Args:
            chains: Dictionary mapping chain_id to a dict with keys
                'sequence' (str) and 'chain_type' ("heavy" or "light").

        Returns:
            Dictionary mapping chain_id to list of CDR regions.
        """
        result = {}
        for chain_id, info in chains.items():
            chain_type = info.get("chain_type")
            sequence = info.get("sequence", "")
            if chain_type in ("heavy", "light") and sequence:
                result[chain_id] = self.annotate(sequence, chain_type, chain_id)
        return result

    def _annotate_with_anarci(
        self, sequence: str, chain_type: str, chain_id: str
    ) -> list[CDRRegion]:
        """Use ANARCI for accurate CDR annotation."""
        scheme_name = self.scheme.value
        numbering_results = self._anarci_fn(
            [("query", sequence)], scheme=scheme_name, allowed_species=["human", "mouse"]
        )

        if not numbering_results or numbering_results[0] is None:
            logger.warning(
                f"ANARCI failed to number chain {chain_id}, "
                f"falling back to heuristic"
            )
            return self._annotate_with_heuristic(sequence, chain_type, chain_id)

        numbering_list = numbering_results[0][0][0]
        boundaries = _SCHEME_BOUNDARIES[self.scheme][chain_type]
        regions = []

        for cdr_name, (start_num, end_num) in boundaries.items():
            cdr_residues = []
            cdr_start = None

            for i, ((pos_num, _insertion), aa) in enumerate(numbering_list):
                if start_num <= pos_num <= end_num and aa != "-":
                    if cdr_start is None:
                        cdr_start = i
                    cdr_residues.append(aa)

            if cdr_residues and cdr_start is not None:
                cdr_seq = "".join(cdr_residues)
                regions.append(
                    CDRRegion(
                        name=cdr_name,
                        start_idx=cdr_start,
                        end_idx=cdr_start + len(cdr_residues),
                        sequence=cdr_seq,
                        chain_id=chain_id,
                    )
                )

        return regions

    def _annotate_with_heuristic(
        self, sequence: str, chain_type: str, chain_id: str
    ) -> list[CDRRegion]:
        """Simple positional heuristic for CDR annotation.

        This is a rough approximation based on typical antibody sequence
        lengths. For production use, install ANARCI for accurate results.
        """
        seq_len = len(sequence)
        if seq_len < 80:
            logger.warning(
                f"Sequence too short for CDR annotation "
                f"(length={seq_len}, chain={chain_id})"
            )
            return []

        # Approximate CDR positions based on typical variable domain
        # structure (~110-120 residues for the variable region)
        if chain_type == "heavy":
            cdr_ranges = {
                "CDR-H1": (26, 35),
                "CDR-H2": (50, 58),
                "CDR-H3": (93, min(102, seq_len)),
            }
        else:
            cdr_ranges = {
                "CDR-L1": (24, 34),
                "CDR-L2": (50, 56),
                "CDR-L3": (89, min(97, seq_len)),
            }

        regions = []
        for cdr_name, (start, end) in cdr_ranges.items():
            if end <= seq_len:
                regions.append(
                    CDRRegion(
                        name=cdr_name,
                        start_idx=start,
                        end_idx=end,
                        sequence=sequence[start:end],
                        chain_id=chain_id,
                    )
                )

        return regions
