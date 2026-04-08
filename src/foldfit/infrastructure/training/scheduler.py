"""Learning rate schedulers with warmup support."""

from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def build_scheduler(
    optimizer: Optimizer,
    scheduler_type: str,
    warmup_steps: int,
    total_steps: int,
    min_lr: float = 1e-6,
) -> LambdaLR:
    """Build a learning rate scheduler with linear warmup.

    Args:
        optimizer: The optimizer to schedule.
        scheduler_type: One of "cosine", "linear", "constant".
        warmup_steps: Number of warmup steps (linear ramp from 0 to lr).
        total_steps: Total number of training steps.
        min_lr: Minimum learning rate after decay.

    Returns:
        A LambdaLR scheduler that applies warmup + decay.
    """
    base_lrs = [group["lr"] for group in optimizer.param_groups]

    def lr_lambda(current_step: int, group_idx: int = 0) -> float:
        base_lr = base_lrs[group_idx]
        if base_lr == 0:
            return 0.0

        # Warmup phase
        if current_step < warmup_steps:
            return current_step / max(warmup_steps, 1)

        # Decay phase
        decay_steps = total_steps - warmup_steps
        if decay_steps <= 0:
            return 1.0

        progress = (current_step - warmup_steps) / decay_steps

        if scheduler_type == "cosine":
            decay = 0.5 * (1 + math.cos(math.pi * progress))
        elif scheduler_type == "linear":
            decay = 1.0 - progress
        elif scheduler_type == "constant":
            return 1.0
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")

        # Scale decay so LR doesn't go below min_lr
        min_ratio = min_lr / base_lr
        return max(min_ratio, decay)

    # Create per-group lambdas to support different base LRs
    lambdas = [lambda step, i=i: lr_lambda(step, i) for i in range(len(base_lrs))]
    return LambdaLR(optimizer, lambdas)
