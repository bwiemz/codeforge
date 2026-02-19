import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Faster than LayerNorm — no mean subtraction, no bias.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


class FeedForward(nn.Module):
    """SwiGLU feed-forward network with optional Smooth-SwiGLU.

    Standard:  output = down_proj(SiLU(gate_proj(x)) * up_proj(x))
    Smooth:    output = down_proj(SiLU(gate_proj(x)) * (up_proj(x) * smooth_scale))

    Smooth-SwiGLU adds a per-channel learnable scale on the up_proj branch.
    This prevents gate/up weight alignment under L2 regularization that causes
    activation outliers during extended FP8 training (Intel/Habana ICLR 2025).
    Initializes to 1.0 (identity at init). Can be folded into up_proj.weight
    after training for zero inference cost.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        hidden_dim = config.ffn_hidden_dim
        self.gate_proj = nn.Linear(config.dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(config.dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, config.dim, bias=False)
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

        self.smooth_scale: nn.Parameter | None = None
        if config.use_smooth_swiglu:
            self.smooth_scale = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        if self.smooth_scale is not None:
            up = up * self.smooth_scale
        return self.dropout(self.down_proj(gate * up))
