"""TCFP Tokenizer — code tokenizer aligned to FP8 tensor core training.

Vocabulary of 49,152 tokens (384 x 128) using byte-level BPE with a
GPT-4-derived pre-tokenization regex and individual digit splitting.
Designed for peak throughput with TCFP's FP8 dual-GEMM kernels.
"""

from __future__ import annotations

from .alignment import AlignmentReport, check_alignment, suggest_vocab_size
from .config import TokenizerConfig
from .constants import (
    CHATML_TEMPLATE,
    PRETOKENIZE_REGEX,
    SPECIAL_TOKENS,
    VOCAB_SIZE,
)
from .training import train_tokenizer, train_tokenizer_from_iterator
from .validation import ValidationReport, validate_tokenizer
from .wrapper import TCFPTokenizer

__all__ = [
    "AlignmentReport",
    "CHATML_TEMPLATE",
    "PRETOKENIZE_REGEX",
    "SPECIAL_TOKENS",
    "TCFPTokenizer",
    "TokenizerConfig",
    "VOCAB_SIZE",
    "ValidationReport",
    "check_alignment",
    "suggest_vocab_size",
    "train_tokenizer",
    "train_tokenizer_from_iterator",
    "validate_tokenizer",
]

__version__ = "0.1.0"
