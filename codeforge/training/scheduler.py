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
