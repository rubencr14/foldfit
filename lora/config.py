"""LoRA configuration dataclass."""

from dataclasses import dataclass, field


@dataclass
class LoRAConfig:
    """Configuration for LoRA (Low-Rank Adaptation) adapters.

    Attributes:
        rank: Rank of the low-rank decomposition. Lower values use fewer
            parameters but have less capacity.
        alpha: Scaling factor for the LoRA update. The effective scaling
            is alpha / rank.
        dropout: Dropout probability applied to the input before the LoRA
            branch. Set to 0.0 to disable.
        target_modules: List of linear layer attribute names to adapt.
            These are matched against the leaf attribute name in the module
            tree (e.g., "linear_q", "linear_k", "linear_v", "linear_o").
        target_blocks: List of module name prefixes that restrict where
            LoRA is applied. Only linear layers whose fully-qualified name
            contains one of these prefixes will be adapted (e.g.,
            "pairformer_stack", "diffusion_module").
    """

    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    target_modules: list[str] = field(
        default_factory=lambda: ["linear_q", "linear_k", "linear_v", "linear_o"]
    )
    target_blocks: list[str] = field(
        default_factory=lambda: ["pairformer_stack", "diffusion_module"]
    )

    @property
    def scaling(self) -> float:
        """Effective scaling factor applied to the LoRA output."""
        return self.alpha / self.rank

    def __post_init__(self):
        if self.rank <= 0:
            raise ValueError(f"rank must be positive, got {self.rank}")
        if self.alpha <= 0:
            raise ValueError(f"alpha must be positive, got {self.alpha}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if not self.target_modules:
            raise ValueError("target_modules must not be empty")
        if not self.target_blocks:
            raise ValueError("target_blocks must not be empty")
