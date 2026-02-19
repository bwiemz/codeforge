"""User-facing TCFPTokenizer class for integration with TCFP training.

Wraps a trained HuggingFace tokenizer and guarantees FP8 tensor core
alignment.  Provides property accessors for special token IDs that
model code needs (pad, bos, eos, FIM tokens, etc.).

Implements HF-compatible duck-typing (``__len__``, ``__call__``,
``convert_tokens_to_ids``, ``convert_ids_to_tokens``, ``get_vocab``)
so that standard HF training loops and ``datasets`` pipelines work
without requiring ``transformers`` as a dependency.

Usage with a TCFP model::

    tok = TCFPTokenizer("tokenizer_output/tcfp_tokenizer.json")
    model = nn.Sequential(
        nn.Embedding(tok.vocab_size, d_model),  # 49_152 — aligned
        ...
        nn.Linear(d_model, tok.vocab_size),      # logits projection
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import overload

from tokenizers import Tokenizer

from .alignment import AlignmentReport, check_alignment
from .constants import SPECIAL_TOKENS


class TCFPTokenizer:
    """Ready-to-use tokenizer for TCFP FP8 training pipelines.

    Validates tensor core alignment on construction and exposes
    special token IDs as properties for model/data-pipeline code.

    Raises
    ------
    ValueError
        If the loaded tokenizer's vocab size is not aligned for peak
        TCFP throughput.
    """

    def __init__(self, tokenizer_path: str | Path) -> None:
        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self._alignment = check_alignment(self._tokenizer.get_vocab_size())

        if not self._alignment.peak_throughput:
            raise ValueError(
                f"Tokenizer vocab size {self._tokenizer.get_vocab_size()} "
                f"is not aligned for peak TCFP throughput.  "
                f"Expected a size divisible by 128.  "
                f"Use alignment.suggest_vocab_size() to find a valid size."
            )

        # Verify special token IDs match compile-time constants
        vocab = self._tokenizer.get_vocab()
        for token_str, expected_id in SPECIAL_TOKENS.items():
            actual_id = vocab.get(token_str)
            if actual_id is not None and actual_id != expected_id:
                raise ValueError(
                    f"Special token {token_str!r} is at ID {actual_id} "
                    f"in the loaded tokenizer, expected {expected_id}.  "
                    f"The tokenizer file may have been built with an "
                    f"incompatible version."
                )

    # ── Vocabulary info ────────────────────────────────────────────

    @property
    def vocab_size(self) -> int:
        """Total vocabulary size (BPE + special tokens)."""
        return self._tokenizer.get_vocab_size()

    @property
    def alignment_report(self) -> AlignmentReport:
        """FP8 tensor core alignment details."""
        return self._alignment

    # ── Special token IDs ──────────────────────────────────────────

    @property
    def pad_id(self) -> int:
        return SPECIAL_TOKENS["<|pad|>"]

    @property
    def bos_id(self) -> int:
        return SPECIAL_TOKENS["<|bos|>"]

    @property
    def eos_id(self) -> int:
        return SPECIAL_TOKENS["<|eos|>"]

    @property
    def unk_id(self) -> int:
        return SPECIAL_TOKENS["<|unk|>"]

    @property
    def fim_prefix_id(self) -> int:
        return SPECIAL_TOKENS["<|fim_prefix|>"]

    @property
    def fim_middle_id(self) -> int:
        return SPECIAL_TOKENS["<|fim_middle|>"]

    @property
    def fim_suffix_id(self) -> int:
        return SPECIAL_TOKENS["<|fim_suffix|>"]

    @property
    def fim_pad_id(self) -> int:
        return SPECIAL_TOKENS["<|fim_pad|>"]

    @property
    def repo_name_id(self) -> int:
        return SPECIAL_TOKENS["<|repo_name|>"]

    @property
    def file_sep_id(self) -> int:
        return SPECIAL_TOKENS["<|file_sep|>"]

    @property
    def im_start_id(self) -> int:
        return SPECIAL_TOKENS["<|im_start|>"]

    @property
    def im_end_id(self) -> int:
        return SPECIAL_TOKENS["<|im_end|>"]

    @property
    def code_start_id(self) -> int:
        return SPECIAL_TOKENS["<|code_start|>"]

    @property
    def code_end_id(self) -> int:
        return SPECIAL_TOKENS["<|code_end|>"]

    # ── Uppercase aliases (used by dataset, inference, eval code) ─

    @property
    def EOS_ID(self) -> int:  # noqa: N802
        return self.eos_id

    @property
    def PAD_ID(self) -> int:  # noqa: N802
        return self.pad_id

    @property
    def BOS_ID(self) -> int:  # noqa: N802
        return self.bos_id

    @property
    def ENDOFCODE_ID(self) -> int:  # noqa: N802
        return self.code_end_id

    # ── Encode / decode ────────────────────────────────────────────

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        """Encode *text* to token IDs."""
        return self._tokenizer.encode(text, add_special_tokens=add_special_tokens).ids

    def decode(self, ids: list[int], *, skip_special_tokens: bool = False) -> str:
        """Decode token *ids* back to text."""
        return self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def encode_batch(
        self, texts: list[str], *, add_special_tokens: bool = False
    ) -> list[list[int]]:
        """Encode a batch of texts to token IDs."""
        encodings = self._tokenizer.encode_batch(
            texts, add_special_tokens=add_special_tokens
        )
        return [e.ids for e in encodings]

    def decode_batch(
        self, id_lists: list[list[int]], *, skip_special_tokens: bool = False
    ) -> list[str]:
        """Decode a batch of token ID lists back to texts."""
        return self._tokenizer.decode_batch(
            id_lists, skip_special_tokens=skip_special_tokens
        )

    # ── Access to underlying tokenizer ─────────────────────────────

    @property
    def inner(self) -> Tokenizer:
        """The underlying HuggingFace ``Tokenizer`` instance."""
        return self._tokenizer

    # ── HF-compatible duck-typing ───────────────────────────────────

    def __len__(self) -> int:
        """Return vocab size (expected by HF training loops)."""
        return self._tokenizer.get_vocab_size()

    @overload
    def __call__(
        self, text: str, *, add_special_tokens: bool = ...
    ) -> dict[str, list[int]]: ...
    @overload
    def __call__(
        self, text: list[str], *, add_special_tokens: bool = ...
    ) -> dict[str, list[list[int]]]: ...

    def __call__(
        self,
        text: str | list[str],
        *,
        add_special_tokens: bool = False,
    ) -> dict[str, list[int]] | dict[str, list[list[int]]]:
        """Encode text(s) and return a dict with ``input_ids``.

        Matches the call signature HF ``datasets.map()`` and training
        loops expect.  Returns ``{"input_ids": ...}`` where the value
        is a flat list for a single string or a list of lists for a
        batch.
        """
        if isinstance(text, str):
            ids = self.encode(text, add_special_tokens=add_special_tokens)
            return {"input_ids": ids}
        ids_batch = self.encode_batch(text, add_special_tokens=add_special_tokens)
        return {"input_ids": ids_batch}

    def get_vocab(self) -> dict[str, int]:
        """Return the full token-to-ID mapping."""
        return self._tokenizer.get_vocab()

    @overload
    def convert_tokens_to_ids(self, tokens: str) -> int | None: ...
    @overload
    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int | None]: ...

    def convert_tokens_to_ids(
        self, tokens: str | list[str]
    ) -> int | None | list[int | None]:
        """Convert token string(s) to their vocabulary IDs."""
        if isinstance(tokens, str):
            return self._tokenizer.token_to_id(tokens)
        return [self._tokenizer.token_to_id(t) for t in tokens]

    @overload
    def convert_ids_to_tokens(self, ids: int) -> str | None: ...
    @overload
    def convert_ids_to_tokens(self, ids: list[int]) -> list[str | None]: ...

    def convert_ids_to_tokens(
        self, ids: int | list[int]
    ) -> str | None | list[str | None]:
        """Convert token ID(s) to their string representations."""
        if isinstance(ids, int):
            return self._tokenizer.id_to_token(ids)
        return [self._tokenizer.id_to_token(i) for i in ids]
