"""Learning rate schedulers."""

import math
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """Cosine decay LR schedule with linear warmup.

    Args:
        optimizer: The optimizer to schedule.
        warmup_steps: Number of steps for linear warmup.
        total_steps: Total number of training steps.
        min_lr_ratio: Minimum LR as a fraction of peak LR (default 10%).
    """

    def lr_lambda(current_step: int) -> float:
        # Linear warmup
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        # Cosine decay
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)


def get_wsd_schedule(
    optimizer: Optimizer,
    warmup_steps: int,
    stable_steps: int,
    decay_steps: int,
    min_lr_ratio: float = 0.1,
    decay_type: str = "cosine",
) -> LambdaLR:
    """Warmup-Stable-Decay LR schedule.

    Three phases:
      1. Linear warmup from 0 to peak LR
      2. Stable hold at peak LR (enables mid-training restarts without LR mismatch)
      3. Cosine or linear decay to min_lr_ratio * peak LR

    Args:
        optimizer: The optimizer to schedule.
        warmup_steps: Steps for linear warmup.
        stable_steps: Steps to hold at peak LR.
        decay_steps: Steps for decay to min_lr_ratio.
        min_lr_ratio: Minimum LR as fraction of peak (default 10%).
        decay_type: Decay shape — "cosine" or "linear".
    """
    decay_start = warmup_steps + stable_steps

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        if current_step < decay_start:
            return 1.0
        if decay_steps <= 0:
            return 1.0
        progress = min((current_step - decay_start) / decay_steps, 1.0)
        if decay_type == "linear":
            return min_lr_ratio + (1.0 - min_lr_ratio) * (1.0 - progress)
        # cosine (default)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)
