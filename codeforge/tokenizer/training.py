"""BPE training pipeline using HuggingFace tokenizers.

Builds a byte-level BPE tokenizer with a GPT-4-style pre-tokenization
regex (individual digit splitting), trains on a user-provided corpus,
registers special tokens at exact end-of-vocabulary IDs, and saves the
result as a single JSON file.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator
from pathlib import Path

from tokenizers import Regex, Tokenizer, models, trainers
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.pre_tokenizers import ByteLevel, Sequence, Split
from tokenizers.processors import TemplateProcessing

from .config import TokenizerConfig
from .constants import PRETOKENIZE_REGEX, SPECIAL_TOKENS
from .special_tokens import register_special_tokens


def build_tokenizer_pipeline(
    config: TokenizerConfig,
) -> tuple[Tokenizer, trainers.BpeTrainer]:
    """Construct the tokenizer and trainer objects (before training).

    The pre-tokenizer is a two-stage sequence:

    1. **Split** with the GPT-4-derived regex — segments text into
       contractions, words, individual digits, punctuation runs, and
       whitespace groups.  ``behavior="isolated"`` prevents BPE merges
       across segment boundaries.

       Note: the regex is designed to be exhaustive for ASCII and common
       Unicode.  Any character not matched by the regex is silently
       dropped (``behavior="isolated"``).  This is fine for code
       (ASCII-dominant) but exotic Unicode may be lost.

    2. **ByteLevel** — maps every byte to a visible Unicode character so
       the BPE operates on a closed 256-symbol alphabet with no UNK.

    No normalizer is set.  This is intentional: normalisers (NFKC,
    lowercase, etc.) destroy code semantics like case-sensitive
    identifiers and operator distinctions.

    Returns
    -------
    tuple[Tokenizer, BpeTrainer]
        Ready to call ``tokenizer.train(files=..., trainer=trainer)``.
    """
    tokenizer = Tokenizer(models.BPE())

    # Pre-tokenizer: GPT-4 regex → byte-level encoding
    tokenizer.pre_tokenizer = Sequence(
        [
            Split(
                pattern=Regex(PRETOKENIZE_REGEX),
                behavior="isolated",
            ),
            ByteLevel(add_prefix_space=False, trim_offsets=True, use_regex=False),
        ]
    )

    # Decoder must mirror the pre-tokenizer's byte-level encoding
    tokenizer.decoder = ByteLevelDecoder()

    # BPE trainer — trains exactly bpe_vocab_size tokens (256 base + merges).
    # Special tokens are registered *after* training to guarantee exact IDs.
    trainer = trainers.BpeTrainer(
        vocab_size=config.bpe_vocab_size,
        min_frequency=config.min_frequency,
        show_progress=config.show_progress,
        initial_alphabet=ByteLevel.alphabet(),
        special_tokens=[],
    )

    return tokenizer, trainer


def _configure_post_training(tokenizer: Tokenizer) -> None:
    """Configure post-processor and padding after special tokens are registered.

    Sets up:
    - **TemplateProcessing** — ``encode(text, add_special_tokens=True)``
      wraps output in ``<|bos|> ... <|eos|>`` automatically.
    - **Padding** — enables ``pad_id`` / ``pad_token`` so batch encoding
      and HF DataCollators work out of the box.

    Truncation is intentionally NOT set here because max sequence
    length is model-dependent, not tokenizer-dependent.
    """
    tokenizer.post_processor = TemplateProcessing(
        single="<|bos|> $A <|eos|>",
        pair="<|bos|> $A <|eos|> $B <|eos|>",
        special_tokens=[
            ("<|bos|>", SPECIAL_TOKENS["<|bos|>"]),
            ("<|eos|>", SPECIAL_TOKENS["<|eos|>"]),
        ],
    )

    tokenizer.enable_padding(
        pad_id=SPECIAL_TOKENS["<|pad|>"],
        pad_token="<|pad|>",
        direction="right",  # right-pad for decoder-only (GPT-style) models
    )


def _save_tokenizer(tokenizer: Tokenizer, config: TokenizerConfig) -> Path:
    """Atomically save tokenizer to disk (temp file + rename)."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = config.output_dir / config.tokenizer_filename
    fd, tmp_path = tempfile.mkstemp(
        dir=str(config.output_dir), suffix=".tmp"
    )
    try:
        os.close(fd)
        tokenizer.save(tmp_path)
        Path(tmp_path).replace(output_path)
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise
    return output_path


def train_tokenizer(
    config: TokenizerConfig,
    corpus_files: list[Path] | None = None,
) -> Tokenizer:
    """Full training pipeline: build -> train on files -> register specials -> save.

    Parameters
    ----------
    config
        Tokenizer configuration (vocab size, output path, etc.).
    corpus_files
        Override for ``config.corpus_paths``.  Useful when the caller
        already has resolved file paths.

    Returns
    -------
    Tokenizer
        The trained tokenizer with all 49,152 tokens.
    """
    tokenizer, trainer = build_tokenizer_pipeline(config)

    files = corpus_files if corpus_files is not None else config.corpus_paths
    if not files:
        raise ValueError(
            "No corpus files provided.  Pass corpus_files or set "
            "config.corpus_paths before calling train_tokenizer()."
        )

    str_files = [str(p) for p in files]
    tokenizer.train(files=str_files, trainer=trainer)

    register_special_tokens(tokenizer)
    _configure_post_training(tokenizer)
    _save_tokenizer(tokenizer, config)
    return tokenizer


def train_tokenizer_from_iterator(
    config: TokenizerConfig,
    iterator: Iterator[str],
    length: int | None = None,
) -> Tokenizer:
    """Full training pipeline using a text iterator (e.g. HF dataset stream).

    Parameters
    ----------
    config
        Tokenizer configuration (vocab size, output path, etc.).
    iterator
        Yields text strings for BPE training.
    length
        Optional hint for progress bar (total number of items).

    Returns
    -------
    Tokenizer
        The trained tokenizer with all 49,152 tokens.
    """
    tokenizer, trainer = build_tokenizer_pipeline(config)
    tokenizer.train_from_iterator(iterator, trainer=trainer, length=length)

    register_special_tokens(tokenizer)
    _configure_post_training(tokenizer)
    _save_tokenizer(tokenizer, config)
    return tokenizer
