"""
Gradient checkpointing (activation checkpointing) utilities.

Trades compute for memory: instead of keeping all intermediate activations
in VRAM for backward, re-computes them during the backward pass.  Reduces
peak memory by ~40-60% at the cost of ~25-35% slower training.

This is essential for long protein sequences (>512 residues) or deep
models (>8 blocks) on consumer GPUs (24 GB).

Usage::

    from molfun.training.checkpointing import apply_gradient_checkpointing

    # On a MolfunStructureModel
    apply_gradient_checkpointing(model.adapter)

    # With FSDP (handled automatically when activation_checkpointing=True)
    dist = FSDPStrategy(activation_checkpointing=True)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


def apply_gradient_checkpointing(
    module: nn.Module,
    block_types: set[type] | None = None,
    preserve_rng_state: bool = True,
) -> int:
    """
    Apply gradient checkpointing to submodules of the given module.

    Wraps the ``forward`` method of each qualifying submodule with
    ``torch.utils.checkpoint.checkpoint``, so intermediate activations
    are freed during forward and recomputed during backward.

    Args:
        module: The root module (typically ``model.adapter`` or a ``BuiltModel``).
        block_types: Explicit set of module types to checkpoint.
            If ``None``, auto-detects common block types by name
            (evoformer, pairformer, block, layer, transformer).
        preserve_rng_state: Preserve RNG state for reproducibility
            (slight overhead, recommended for training).

    Returns:
        Number of submodules that were checkpointed.
    """
    if block_types is None:
        block_types = _detect_block_types(module)

    if not block_types:
        return 0

    count = 0
    for child in module.modules():
        if type(child) in block_types and not _is_already_checkpointed(child):
            _wrap_forward(child, preserve_rng_state)
            count += 1

    return count


def remove_gradient_checkpointing(module: nn.Module) -> int:
    """
    Remove gradient checkpointing from previously wrapped submodules.

    Restores the original ``forward`` method.

    Returns:
        Number of submodules that were unwrapped.
    """
    count = 0
    for child in module.modules():
        if hasattr(child, "_original_forward"):
            child.forward = child._original_forward
            del child._original_forward
            count += 1
    return count


def estimate_memory_savings(
    module: nn.Module,
    seq_length: int = 256,
    batch_size: int = 1,
    dtype: torch.dtype = torch.float32,
) -> dict[str, float]:
    """
    Estimate memory savings from gradient checkpointing (heuristic).

    Returns dict with ``estimated_savings_mb`` and ``estimated_savings_pct``.
    This is a rough estimate based on parameter count and typical
    activation-to-parameter ratios for protein models.
    """
    total_params = sum(p.numel() for p in module.parameters())
    bytes_per_param = 4 if dtype == torch.float32 else 2

    param_memory_mb = total_params * bytes_per_param / 1e6

    activation_ratio = seq_length * batch_size * 0.001
    estimated_activation_mb = param_memory_mb * activation_ratio

    savings_pct = 0.45
    savings_mb = estimated_activation_mb * savings_pct

    return {
        "param_memory_mb": round(param_memory_mb, 1),
        "estimated_activation_mb": round(estimated_activation_mb, 1),
        "estimated_savings_mb": round(savings_mb, 1),
        "estimated_savings_pct": round(savings_pct * 100, 1),
    }


# ------------------------------------------------------------------
# Internal
# ------------------------------------------------------------------

_BLOCK_KEYWORDS = ("block", "layer", "evoformer", "pairformer", "transformer", "attention")


def _detect_block_types(module: nn.Module) -> set[type]:
    """Auto-detect checkpointable block types by class name patterns."""
    types: set[type] = set()
    for child in module.modules():
        name = type(child).__name__.lower()
        if any(kw in name for kw in _BLOCK_KEYWORDS):
            if _has_parameters(child) and not isinstance(child, (nn.Linear, nn.LayerNorm)):
                types.add(type(child))
    return types


def _has_parameters(module: nn.Module) -> bool:
    """Check if module has any parameters (avoids wrapping empty containers)."""
    return any(True for _ in module.parameters())


def _is_already_checkpointed(module: nn.Module) -> bool:
    return hasattr(module, "_original_forward")


def _wrap_forward(module: nn.Module, preserve_rng_state: bool) -> None:
    """Replace forward with a checkpointed version."""
    original_forward = module.forward
    module._original_forward = original_forward

    def checkpointed_forward(*args, **kwargs):
        if not torch.is_grad_enabled():
            return original_forward(*args, **kwargs)

        # torch.utils.checkpoint requires all tensor args to have requires_grad
        # for the backward to work, so we use use_reentrant=False
        def run(*a):
            return original_forward(*a, **kwargs)

        tensor_args = tuple(a for a in args if isinstance(a, torch.Tensor))
        tuple(a for a in args if not isinstance(a, torch.Tensor))

        if tensor_args:
            return checkpoint(
                run,
                *args,
                use_reentrant=False,
                preserve_rng_state=preserve_rng_state,
            )
        return original_forward(*args, **kwargs)

    module.forward = checkpointed_forward
