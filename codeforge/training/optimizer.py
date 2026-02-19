"""Optimizer utilities shared across trainers."""

import torch.nn as nn


def build_param_groups(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
    embed_lr_ratio: float = 1.0,
) -> list[dict]:
    """Build 3-group parameter list for AdamW.

    Groups:
      1. Decay params: weight tensors (not embedding, not bias/norm) with weight_decay
      2. No-decay params: bias, norm weights with weight_decay=0
      3. Embedding: tok_embeddings.weight with separate LR and weight_decay=0

    Weight tying: output.weight IS tok_embeddings.weight (same tensor).
    Identified by data_ptr() and placed in exactly one group.
    QK-norm frozen gains (requires_grad=False) are auto-excluded.

    Args:
        model: The CodeForgeModel.
        learning_rate: Base learning rate.
        weight_decay: Weight decay for group 1.
        embed_lr_ratio: Embedding LR = learning_rate * embed_lr_ratio.

    Returns:
        List of param group dicts for torch.optim.AdamW.
    """
    embed_param = model.tok_embeddings.weight  # type: ignore[union-attr]
    embed_ptr = embed_param.data_ptr()

    no_decay_names = {"bias", "norm"}
    decay_params: list[nn.Parameter] = []
    no_decay_params: list[nn.Parameter] = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.data_ptr() == embed_ptr:
            continue  # Embedding goes in its own group
        if any(nd in name for nd in no_decay_names):
            no_decay_params.append(p)
        else:
            decay_params.append(p)

    embed_lr = learning_rate * embed_lr_ratio
    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
        {"params": [embed_param], "lr": embed_lr, "weight_decay": 0.0},
    ]
