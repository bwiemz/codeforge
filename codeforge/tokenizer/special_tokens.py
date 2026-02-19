"""Register special tokens at exact end-of-vocabulary IDs.

After BPE training, this module pads the vocabulary up to 49,088 (if
the corpus was too small to learn all merges) and then adds 64 special
tokens to fill IDs 49088-49151.  Named tokens land at their designated
IDs; gaps are filled with ``<|reserved_N|>`` placeholders so the total
vocabulary is always exactly 49,152.
"""

from __future__ import annotations

from tokenizers import AddedToken, Tokenizer

from .constants import (
    SPECIAL_TOKEN_START_ID,
    SPECIAL_TOKENS,
    VOCAB_SIZE,
)


def _build_token_list() -> list[AddedToken]:
    """Build the ordered list of 64 special tokens (named + reserved).

    Gap slots use ``<|reserved_N|>`` where *N* is the absolute token ID,
    making the token self-documenting (e.g. ``<|reserved_49095|>`` is
    clearly in the special-token block at ID 49095).
    """
    # Detect duplicate IDs in SPECIAL_TOKENS at runtime (defense-in-depth;
    # test_constants also checks this, but this catches misconfigurations
    # even if tests aren't run).
    id_to_name: dict[int, str] = {}
    for name, token_id in SPECIAL_TOKENS.items():
        if token_id in id_to_name:
            raise ValueError(
                f"Duplicate special token ID {token_id}: "
                f"{id_to_name[token_id]!r} and {name!r}"
            )
        id_to_name[token_id] = name

    tokens: list[AddedToken] = []
    for token_id in range(SPECIAL_TOKEN_START_ID, VOCAB_SIZE):
        name = id_to_name.get(token_id, f"<|reserved_{token_id}|>")
        tokens.append(AddedToken(name, special=True, normalized=False))
    return tokens


def register_special_tokens(tokenizer: Tokenizer) -> None:
    """Pad vocab to 49,088 then add 64 special tokens at IDs 49088-49151.

    If BPE training produced fewer than 49,088 tokens (small corpus),
    unused BPE slots are filled with ``<|reserved_N|>`` padding tokens
    (where *N* is the absolute token ID) before the special tokens are
    added.  This guarantees the final vocab size is always exactly
    49,152 and uses one unified naming scheme for all placeholder tokens.

    Raises
    ------
    ValueError
        If the current vocab size exceeds ``SPECIAL_TOKEN_START_ID``.
    RuntimeError
        If the final vocab size or any named token ID is wrong.
    """
    current_size = tokenizer.get_vocab_size()

    if current_size > SPECIAL_TOKEN_START_ID:
        raise ValueError(
            f"BPE vocab size {current_size} exceeds the "
            f"target {SPECIAL_TOKEN_START_ID}.  Reduce vocab_size in "
            f"the BpeTrainer config."
        )

    # ── Pad unused BPE slots (small-corpus case) ──────────────────
    if current_size < SPECIAL_TOKEN_START_ID:
        padding: list[AddedToken] = [
            AddedToken(
                f"<|reserved_{current_size + i}|>", special=True, normalized=False
            )
            for i in range(SPECIAL_TOKEN_START_ID - current_size)
        ]
        tokenizer.add_tokens(padding)

    # ── Add 64 special tokens ─────────────────────────────────────
    token_list = _build_token_list()
    tokenizer.add_special_tokens(token_list)

    # ── Verify final state ────────────────────────────────────────
    final_size = tokenizer.get_vocab_size()
    if final_size != VOCAB_SIZE:
        raise RuntimeError(
            f"Final vocab size is {final_size}, expected {VOCAB_SIZE}"
        )

    vocab = tokenizer.get_vocab()
    for token_str, expected_id in SPECIAL_TOKENS.items():
        actual_id = vocab.get(token_str)
        if actual_id != expected_id:
            raise RuntimeError(
                f"Special token {token_str!r} is at ID {actual_id}, "
                f"expected {expected_id}"
            )
