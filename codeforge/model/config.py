from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class ModelConfig:
    vocab_size: int = 49152
    dim: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: int = 4
    max_seq_len: int = 2048
    ffn_hidden_mult: float = 2.667
    norm_eps: float = 1e-6
    dropout: float = 0.0
    rope_theta: float = 10000.0

    # Phase 2: Extended context via RoPE scaling
    rope_scaling_type: Optional[str] = None  # None, "linear", "ntk", "yarn"
    rope_scaling_factor: float = 1.0  # scaling factor for extended context

    # Phase 2: Training efficiency
    use_gradient_checkpointing: bool = False
    depth_scaled_init: bool = True  # scale residual by 1/sqrt(2*n_layers)

    # Reclaimer Protocol: born-quantized architecture for FP8 training
    use_post_norm: bool = True       # FOG post-norm (norm after sublayer, before residual add)
    use_qk_norm: bool = True         # QK-RMSNorm with frozen gains before RoPE
    z_loss_alpha: float = 1e-4       # Z-loss coefficient (0.0 disables)

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads

    @property
    def ffn_hidden_dim(self) -> int:
        hidden = int(self.dim * self.ffn_hidden_mult)
        # Round to nearest multiple of 64 for GPU efficiency
        return ((hidden + 63) // 64) * 64

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ModelConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        with open(path, "w") as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)

    def param_count_estimate(self) -> int:
        """Rough parameter count estimate."""
        embed = self.vocab_size * self.dim
        attn_per_layer = (
            self.dim * self.dim  # Q
            + self.dim * (self.head_dim * self.n_kv_heads)  # K
            + self.dim * (self.head_dim * self.n_kv_heads)  # V
            + self.dim * self.dim  # output projection
        )
        ffn_per_layer = 3 * self.dim * self.ffn_hidden_dim  # gate, up, down
        norm_per_layer = 2 * self.dim  # 2x RMSNorm per layer
        qk_norm_per_layer = 2 * self.head_dim if self.use_qk_norm else 0
        total = (
            embed
            + self.n_layers * (attn_per_layer + ffn_per_layer + norm_per_layer + qk_norm_per_layer)
            + self.dim  # final norm
            # output projection is weight-tied with embedding
        )
        return total


# Preset configurations
PRESETS = {
    "150m": ModelConfig(
        dim=768,
        n_layers=12,
        n_heads=12,
        n_kv_heads=4,
        ffn_hidden_mult=2.667,
    ),
    "350m": ModelConfig(
        dim=1024,
        n_layers=24,
        n_heads=16,
        n_kv_heads=4,
        ffn_hidden_mult=2.667,
    ),
    "1b": ModelConfig(
        dim=2048,
        n_layers=24,
        n_heads=16,
        n_kv_heads=8,
        ffn_hidden_mult=2.667,
    ),
    "3b": ModelConfig(
        dim=3072,
        n_layers=26,
        n_heads=24,
        n_kv_heads=8,
        ffn_hidden_mult=2.667,
    ),
}


def get_preset(name: str) -> ModelConfig:
    if name not in PRESETS:
        raise ValueError(f"Unknown preset '{name}'. Available: {list(PRESETS.keys())}")
    return PRESETS[name]
