"""LoRA linear layer wrapper."""

import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Wraps an existing Linear layer with a LoRA low-rank adaptation.

    The original layer is kept frozen. Two small parameter matrices
    (lora_A and lora_B) implement the low-rank update:

        output = original(x) + dropout(x) @ lora_A @ lora_B * scaling

    lora_B is zero-initialized so the adapter starts as an identity
    (zero delta), preserving the pretrained model's behavior at init.

    Args:
        original: The linear layer to wrap. Its parameters are frozen.
        rank: Rank of the low-rank decomposition.
        alpha: Scaling factor. Effective scaling is alpha / rank.
        dropout: Dropout probability on input before the LoRA branch.
    """

    def __init__(
        self,
        original: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.original = original
        self.rank = rank
        self.scaling = alpha / rank

        in_features = original.in_features
        out_features = original.out_features

        # Freeze the original layer
        for param in self.original.parameters():
            param.requires_grad = False

        # Low-rank decomposition matrices
        self.lora_A = nn.Parameter(torch.empty(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

        # Initialize A with Kaiming uniform (like nn.Linear default)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass combining original output with LoRA update."""
        base_output = self.original(input)
        lora_input = self.lora_dropout(input)
        lora_output = (lora_input @ self.lora_A @ self.lora_B) * self.scaling
        return base_output + lora_output.to(base_output.dtype)

    def merge(self) -> nn.Linear:
        """Merge LoRA weights into the original layer and return it.

        After merging, the original layer produces the same output as
        the combined original + LoRA forward pass, without the overhead
        of the separate LoRA computation.

        Returns:
            The original nn.Linear with LoRA weights folded in.
        """
        with torch.no_grad():
            delta = (self.lora_A @ self.lora_B) * self.scaling
            self.original.weight.add_(delta.T)
        return self.original

    @property
    def weight(self) -> torch.Tensor:
        """Proxy to the original layer's weight for compatibility."""
        return self.original.weight

    @property
    def bias(self) -> torch.Tensor | None:
        """Proxy to the original layer's bias for compatibility."""
        return self.original.bias

    def extra_repr(self) -> str:
        return (
            f"in_features={self.original.in_features}, "
            f"out_features={self.original.out_features}, "
            f"rank={self.rank}, scaling={self.scaling:.4f}"
        )
