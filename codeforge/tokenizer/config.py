"""TokenizerConfig dataclass following TCFP's FP8Config pattern."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .constants import (
    BPE_VOCAB_SIZE,
    NUM_SPECIAL_TOKENS,
    TCFP_TILE_SIZE,
    VOCAB_SIZE,
)


@dataclass
class TokenizerConfig:
    """Configuration for TCFP tokenizer training and usage.

    Validates FP8 alignment invariants in ``__post_init__`` so that
    misconfigurations are caught before expensive BPE training.
    """

    # ── Vocabulary ─────────────────────────────────────────────────
    vocab_size: int = VOCAB_SIZE
    bpe_vocab_size: int = BPE_VOCAB_SIZE
    num_special_tokens: int = NUM_SPECIAL_TOKENS

    # ── Training corpus ────────────────────────────────────────────
    corpus_paths: list[Path] = field(default_factory=list)
    min_frequency: int = 2
    code_fraction: float = 0.70  # 70 % code, 30 % NL
    target_languages: tuple[str, ...] = (
        "python",
        "javascript",
        "typescript",
        "java",
        "cpp",
        "c",
        "go",
        "rust",
    )

    # ── BPE algorithm ──────────────────────────────────────────────
    add_prefix_space: bool = False
    show_progress: bool = True

    # ── Output ─────────────────────────────────────────────────────
    output_dir: Path = Path("tokenizer_output")
    tokenizer_filename: str = "tcfp_tokenizer.json"

    # ── Validation thresholds ──────────────────────────────────────
    compression_ratio_min: float = 3.5
    compression_ratio_max: float = 4.5

    def __post_init__(self) -> None:
        if self.vocab_size != self.bpe_vocab_size + self.num_special_tokens:
            raise ValueError(
                f"vocab_size ({self.vocab_size}) must equal "
                f"bpe_vocab_size ({self.bpe_vocab_size}) + "
                f"num_special_tokens ({self.num_special_tokens})"
            )
        if self.vocab_size % TCFP_TILE_SIZE != 0:
            raise ValueError(
                f"vocab_size ({self.vocab_size}) must be divisible by "
                f"{TCFP_TILE_SIZE} for TCFP tensor core tile alignment"
            )
        if not 0.0 <= self.code_fraction <= 1.0:
            raise ValueError(
                f"code_fraction must be in [0, 1], got {self.code_fraction}"
            )

    @property
    def output_path(self) -> Path:
        return self.output_dir / self.tokenizer_filename
