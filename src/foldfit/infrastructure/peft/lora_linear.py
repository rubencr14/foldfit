"""LoRA linear layer: drop-in replacement for nn.Linear with low-rank adaptation.

Mathematics: y = W_frozen @ x + (alpha / rank) * (B @ A) @ x
Where A is [rank, in_features] and B is [out_features, rank].
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Linear layer with Low-Rank Adaptation.

    Freezes the original weight matrix and adds trainable low-rank
    matrices A and B such that the effective weight is:
        W_effective = W_frozen + scaling * (B @ A)

    Args:
        original: The nn.Linear layer to adapt.
        rank: Low-rank dimension.
        alpha: Scaling factor (scaling = alpha / rank).
        dropout: Dropout rate applied to input before LoRA path.
    """

    def __init__(
        self,
        original: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        in_features = original.in_features
        out_features = original.out_features

        # Frozen base weight and bias
        self.weight = nn.Parameter(original.weight.data.clone(), requires_grad=False)
        self.bias: nn.Parameter | None = None
        if original.bias is not None:
            self.bias = nn.Parameter(original.bias.data.clone(), requires_grad=False)

        # Trainable low-rank matrices (same device as original weight)
        device = original.weight.device
        self.lora_A = nn.Parameter(torch.empty(rank, in_features, device=device))
        self.lora_B = nn.Parameter(torch.empty(out_features, rank, device=device))

        # Initialization: A with kaiming, B with zeros (so LoRA starts as identity)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        self.scaling = alpha / rank
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self._merged = False

    @property
    def in_features(self) -> int:
        return self.weight.shape[1]

    @property
    def out_features(self) -> int:
        return self.weight.shape[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: base linear + low-rank adaptation.

        Args:
            x: Input tensor of shape [..., in_features].

        Returns:
            Output tensor of shape [..., out_features].
        """
        if self._merged:
            return F.linear(x, self.weight, self.bias)

        base = F.linear(x, self.weight, self.bias)
        lora = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B)
        return base + lora * self.scaling

    def merge(self) -> None:
        """Merge LoRA weights into the base weight for efficient inference."""
        if self._merged:
            return
        self.weight.data += (self.lora_B @ self.lora_A) * self.scaling
        self._merged = True

    def unmerge(self) -> None:
        """Unmerge LoRA weights from the base weight."""
        if not self._merged:
            return
        self.weight.data -= (self.lora_B @ self.lora_A) * self.scaling
        self._merged = False
