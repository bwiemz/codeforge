import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


def precompute_rope_frequencies(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    scaling_type: Optional[str] = None,
    scaling_factor: float = 1.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Precompute sin/cos frequencies for real-valued RoPE.

    Uses real-valued sin/cos representation instead of complex numbers,
    enabling full torch.compile fusion (complex ops are not supported by inductor).

    Supports multiple scaling strategies for extended context:
    - None: Standard RoPE (up to training length)
    - "linear": Linear interpolation (scales positions down)
    - "ntk": NTK-aware scaling (scales theta up, preserves high-freq detail)
    - "yarn": YaRN (NTK + attention scaling, best quality for extended context)

    Returns a tensor of shape (max_seq_len, head_dim // 2, 2) with [cos, sin] values.
    """
    if scaling_type == "ntk":
        theta = theta * (scaling_factor ** (head_dim / (head_dim - 2)))
    elif scaling_type == "yarn":
        theta = theta * (scaling_factor ** (head_dim / (head_dim - 2)))

    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = torch.arange(max_seq_len, device=device).float()

    if scaling_type == "linear":
        positions = positions / scaling_factor

    angles = torch.outer(positions, freqs)

    if scaling_type == "yarn":
        dim_ratios = torch.arange(0, head_dim // 2, device=device).float() / (head_dim // 2)
        blend = torch.clamp((dim_ratios - 0.25) / 0.5, 0, 1)
        scale = (1 - blend) + blend / scaling_factor
        angles = angles * scale.unsqueeze(0)

    # Return [cos, sin] stacked — fully real-valued, no complex ops
    return torch.stack([angles.cos(), angles.sin()], dim=-1)


def apply_rotary_emb(
    xq: torch.Tensor, xk: torch.Tensor, freqs: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings using real-valued sin/cos rotation.

    This avoids torch.view_as_complex/torch.polar which torch.compile cannot fuse,
    using direct sin/cos multiplication instead (same mathematical result).

    Args:
        xq: Query tensor of shape (batch, seq_len, n_heads, head_dim)
        xk: Key tensor of shape (batch, seq_len, n_kv_heads, head_dim)
        freqs: Precomputed [cos, sin] of shape (seq_len, head_dim // 2, 2)
    """
    # Split cos/sin: each (seq_len, head_dim//2) -> (1, seq_len, 1, head_dim//2)
    cos = freqs[..., 0].unsqueeze(0).unsqueeze(2)
    sin = freqs[..., 1].unsqueeze(0).unsqueeze(2)

    # Split into even/odd pairs for rotation
    xq_even = xq[..., 0::2]
    xq_odd = xq[..., 1::2]
    xk_even = xk[..., 0::2]
    xk_odd = xk[..., 1::2]

    # Apply rotation: [x_even, x_odd] * [cos, sin] -> [x_even*cos - x_odd*sin, x_even*sin + x_odd*cos]
    xq_out = torch.stack([xq_even * cos - xq_odd * sin, xq_even * sin + xq_odd * cos], dim=-1).flatten(-2)
    xk_out = torch.stack([xk_even * cos - xk_odd * sin, xk_even * sin + xk_odd * cos], dim=-1).flatten(-2)

    return xq_out.type_as(xq), xk_out.type_as(xk)


class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention (GQA) with RoPE.

    Uses fewer KV heads than query heads for memory efficiency.
    When n_kv_heads == n_heads, this is standard MHA.
    When n_kv_heads == 1, this is Multi-Query Attention (MQA).
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.n_rep = self.n_heads // self.n_kv_heads

        self.wq = nn.Linear(config.dim, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, config.dim, bias=False)

        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        freqs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        batch, seq_len, _ = x.shape

        q = self.wq(x).view(batch, seq_len, self.n_heads, self.head_dim)
        k = self.wk(x).view(batch, seq_len, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(batch, seq_len, self.n_kv_heads, self.head_dim)

        # Apply rotary embeddings to Q and K
        q, k = apply_rotary_emb(q, k, freqs)

        # Handle KV cache for inference
        new_cache = None
        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            k = torch.cat([cached_k, k], dim=1)
            v = torch.cat([cached_v, v], dim=1)
            new_cache = (k, v)

        # Expand KV heads to match Q heads (GQA)
        if self.n_rep > 1:
            k = k.unsqueeze(3).expand(-1, -1, -1, self.n_rep, -1).reshape(
                batch, k.shape[1], self.n_heads, self.head_dim
            )
            v = v.unsqueeze(3).expand(-1, -1, -1, self.n_rep, -1).reshape(
                batch, v.shape[1], self.n_heads, self.head_dim
            )

        # Transpose to (batch, heads, seq, head_dim) for attention
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention (uses Flash Attention when available)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, is_causal=(mask is None and kv_cache is None)
        )

        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.dropout(self.wo(out)), new_cache
