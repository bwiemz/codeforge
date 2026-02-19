import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .attention import GroupedQueryAttention, precompute_rope_frequencies
from .config import ModelConfig
from .layers import FeedForward, RMSNorm


def chunked_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    vocab_size: int,
    chunk_size: int = 512,
) -> torch.Tensor:
    """Compute cross-entropy loss in chunks along the sequence dimension.

    Instead of materializing the full (batch*seq_len, vocab_size) tensor at once,
    this processes chunks of tokens at a time. For batch=8, seq=2048, vocab=32000:
    - Standard: allocates 8*2048*32000*2 = ~1GB contiguous tensor
    - Chunked (512): allocates 8*512*32000*2 = ~250MB at a time

    This reduces peak memory usage and improves cache locality.
    """
    batch, seq_len = targets.shape
    total_loss = torch.tensor(0.0, device=logits.device, dtype=torch.float32)
    total_tokens = 0

    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        chunk_logits = logits[:, start:end, :].reshape(-1, vocab_size)
        chunk_targets = targets[:, start:end].reshape(-1)

        # Count valid tokens (not ignore_index=-1)
        valid_mask = chunk_targets != -1
        n_valid = valid_mask.sum()
        if n_valid == 0:
            continue

        chunk_loss = F.cross_entropy(chunk_logits, chunk_targets, ignore_index=-1, reduction="sum")
        total_loss = total_loss + chunk_loss
        total_tokens += n_valid

    return total_loss / max(total_tokens, 1)


class TransformerBlock(nn.Module):
    """Single transformer block with configurable pre/post-norm architecture."""

    def __init__(self, config: ModelConfig, layer_idx: int = 0):
        super().__init__()
        self.use_post_norm = config.use_post_norm
        self.attn_norm = RMSNorm(config.dim, config.norm_eps)
        self.attn = GroupedQueryAttention(config)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        self.ffn = FeedForward(config)
        self.layer_idx = layer_idx

        # Depth-scaled residual: scale by 1/sqrt(2*n_layers) for stability
        if config.depth_scaled_init:
            self._residual_scale = 1.0 / math.sqrt(2 * config.n_layers)
        else:
            self._residual_scale = 1.0

    def forward(
        self,
        x: torch.Tensor,
        freqs: torch.Tensor,
        mask: torch.Tensor | None = None,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        if self.use_post_norm:
            # FOG post-norm: norm AFTER sublayer, BEFORE residual add
            attn_out, new_cache = self.attn(x, freqs, mask, kv_cache)
            x = x + self.attn_norm(attn_out) * self._residual_scale
            x = x + self.ffn_norm(self.ffn(x)) * self._residual_scale
        else:
            # Pre-norm (LLaMA-style): norm BEFORE sublayer
            attn_out, new_cache = self.attn(self.attn_norm(x), freqs, mask, kv_cache)
            x = x + attn_out * self._residual_scale
            x = x + self.ffn(self.ffn_norm(x)) * self._residual_scale

        # Kurtosis capture (zero overhead when not requested; flag cleared after capture)
        if hasattr(self, '_capture_kurtosis') and self._capture_kurtosis:
            self._last_output_for_kurtosis = x
            self._capture_kurtosis = False

        return x, new_cache


class CodeForgeModel(nn.Module):
    """CodeForge language model.

    Decoder-only transformer with:
    - RoPE (Rotary Position Embeddings) with optional scaling for extended context
    - Grouped Query Attention
    - RMSNorm
    - SwiGLU feed-forward
    - Weight-tied embedding/output projection
    - Optional gradient checkpointing for memory efficiency
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = config.use_gradient_checkpointing

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList([
            TransformerBlock(config, layer_idx=i) for i in range(config.n_layers)
        ])
        self.norm = RMSNorm(config.dim, config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        # Weight tying
        self.output.weight = self.tok_embeddings.weight

        # Precompute RoPE frequencies with optional scaling
        rope_seq_len = config.max_seq_len
        if config.rope_scaling_type and config.rope_scaling_factor > 1.0:
            rope_seq_len = int(config.max_seq_len * config.rope_scaling_factor)

        self.register_buffer(
            "rope_freqs",
            precompute_rope_frequencies(
                config.head_dim,
                rope_seq_len,
                config.rope_theta,
                scaling_type=config.rope_scaling_type,
                scaling_factor=config.rope_scaling_factor,
            ),
            persistent=False,
        )

        self.apply(self._init_weights)

        # Scale output projections by 1/sqrt(2*n_layers) for proper signal propagation
        # Targets: attn.wo (attention output) and ffn.down_proj (FFN output)
        output_scale = 1.0 / math.sqrt(2 * config.n_layers)
        with torch.no_grad():
            for layer in self.layers:
                layer.attn.wo.weight.mul_(output_scale)
                layer.ffn.down_proj.weight.mul_(output_scale)

    def _init_weights(self, module: nn.Module) -> None:
        # Skip output projection — its weight IS tok_embeddings.weight (tied),
        # and we want the Embedding init (N(0, 1/sqrt(dim))) to persist.
        if module is self.output:
            return
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=1.0 / math.sqrt(self.config.dim))

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        """Enable/disable gradient checkpointing at runtime."""
        self.gradient_checkpointing = enabled

    def forward(
        self,
        tokens: torch.Tensor,
        targets: torch.Tensor | None = None,
        kv_caches: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[tuple[torch.Tensor, torch.Tensor]] | None]:
        batch, seq_len = tokens.shape
        x = self.tok_embeddings(tokens)

        freqs = self.rope_freqs[start_pos : start_pos + seq_len]

        new_kv_caches = [] if kv_caches is not None else None
        for i, layer in enumerate(self.layers):
            layer_cache = kv_caches[i] if kv_caches is not None else None

            if self.gradient_checkpointing and self.training and layer_cache is None:
                # Gradient checkpointing: recompute activations during backward
                # Saves ~40% VRAM at cost of ~30% more compute
                x, new_cache = checkpoint(
                    layer, x, freqs, None, layer_cache, use_reentrant=False
                )
            else:
                x, new_cache = layer(x, freqs, kv_cache=layer_cache)

            if new_kv_caches is not None:
                new_kv_caches.append(new_cache)

        x = self.norm(x)
        logits = self.output(x)

        loss = None
        if targets is not None:
            ce_loss = chunked_cross_entropy(logits, targets, self.config.vocab_size)
            # Z-loss: penalizes large logit magnitudes to prevent FP8-amplified divergence
            # Chunked to avoid materializing full FP32 logits (vocab=49152 → ~1.6GB)
            if self.config.z_loss_alpha > 0.0:
                z_accum = torch.tensor(0.0, device=logits.device, dtype=torch.float32)
                z_count = 0
                for z_start in range(0, seq_len, 512):
                    z_end = min(z_start + 512, seq_len)
                    chunk = logits[:, z_start:z_end, :]
                    z_accum = z_accum + (torch.logsumexp(chunk.float(), dim=-1) ** 2).sum()
                    z_count += batch * (z_end - z_start)
                z_loss = self.config.z_loss_alpha * z_accum / max(z_count, 1)
                loss = ce_loss + z_loss
            else:
                loss = ce_loss

        return logits, loss, new_kv_caches

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def param_count_non_embedding(self) -> int:
        return sum(p.numel() for name, p in self.named_parameters() if "tok_embeddings" not in name)
