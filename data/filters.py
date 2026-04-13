"""Antibody structure filtering strategies."""

from abc import ABC, abstractmethod


class AntibodyFilter(ABC):
    """Abstract base for antibody structure filters (Strategy Pattern).

    Each filter receives metadata about a structure and decides whether
    to accept or reject it. Filters can be composed via CompositeFilter.
    """

    @abstractmethod
    def accept(self, entry: dict) -> bool:
        """Return True if the entry passes this filter.

        Args:
            entry: Dictionary containing structure metadata. Expected
                keys depend on the specific filter implementation.
        """


class ResolutionFilter(AntibodyFilter):
    """Filter structures by crystallographic resolution.

    Args:
        max_resolution: Maximum allowed resolution in Angstroms.
            Structures with resolution above this are rejected.
    """

    def __init__(self, max_resolution: float = 3.5):
        if max_resolution <= 0:
            raise ValueError(f"max_resolution must be positive, got {max_resolution}")
        self.max_resolution = max_resolution

    def accept(self, entry: dict) -> bool:
        resolution = entry.get("resolution")
        if resolution is None:
            return False
        return resolution <= self.max_resolution


class ChainTypeFilter(AntibodyFilter):
    """Filter by presence of heavy and/or light antibody chains.

    Args:
        require_heavy: If True, structure must contain a heavy chain.
        require_light: If True, structure must contain a light chain.
    """

    def __init__(
        self, require_heavy: bool = True, require_light: bool = False
    ):
        self.require_heavy = require_heavy
        self.require_light = require_light

    def accept(self, entry: dict) -> bool:
        chains = entry.get("chain_types", [])
        if self.require_heavy and "heavy" not in chains:
            return False
        if self.require_light and "light" not in chains:
            return False
        return True


class SequenceLengthFilter(AntibodyFilter):
    """Filter structures by total sequence length.

    Args:
        min_length: Minimum total residue count.
        max_length: Maximum total residue count.
    """

    def __init__(
        self, min_length: int = 50, max_length: int = 2000
    ):
        self.min_length = min_length
        self.max_length = max_length

    def accept(self, entry: dict) -> bool:
        length = entry.get("sequence_length", 0)
        return self.min_length <= length <= self.max_length


class CompositeFilter(AntibodyFilter):
    """Combines multiple filters with AND logic.

    A structure is accepted only if ALL sub-filters accept it.

    Args:
        filters: List of filters to combine.
    """

    def __init__(self, filters: list[AntibodyFilter]):
        if not filters:
            raise ValueError("At least one filter must be provided")
        self.filters = filters

    def accept(self, entry: dict) -> bool:
        return all(f.accept(entry) for f in self.filters)
