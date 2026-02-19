"""Tests for special token registration and ID placement."""

from __future__ import annotations

import pytest

from codeforge.tokenizer.constants import (
    NUM_SPECIAL_TOKENS,
    SPECIAL_TOKEN_START_ID,
    SPECIAL_TOKENS,
    VOCAB_SIZE,
)
from codeforge.tokenizer.special_tokens import _build_token_list

# These tests don't need the tokenizers library installed
# (they test the token list builder, not the registration itself).


class TestBuildTokenList:
    def test_produces_64_tokens(self) -> None:
        tokens = _build_token_list()
        assert len(tokens) == NUM_SPECIAL_TOKENS

    def test_named_tokens_in_list(self) -> None:
        tokens = _build_token_list()
        names = [t.content for t in tokens]
        for special_name in SPECIAL_TOKENS:
            assert special_name in names, f"{special_name} missing from token list"

    def test_gaps_filled_with_reserved(self) -> None:
        tokens = _build_token_list()
        names = [t.content for t in tokens]
        named_count = len(SPECIAL_TOKENS)
        reserved_count = sum(1 for n in names if n.startswith("<|reserved_"))
        assert named_count + reserved_count == NUM_SPECIAL_TOKENS

    def test_all_tokens_are_special(self) -> None:
        tokens = _build_token_list()
        for t in tokens:
            assert t.special is True

    def test_reserved_tokens_use_pipe_format(self) -> None:
        tokens = _build_token_list()
        for t in tokens:
            if t.content.startswith("<|reserved_"):
                assert t.content.endswith("|>")

    def test_reserved_tokens_use_absolute_ids(self) -> None:
        """Reserved tokens in the special block use absolute token IDs."""
        tokens = _build_token_list()
        for i, t in enumerate(tokens):
            if t.content.startswith("<|reserved_"):
                expected_id = SPECIAL_TOKEN_START_ID + i
                assert t.content == f"<|reserved_{expected_id}|>"

    def test_token_order_matches_id_order(self) -> None:
        """Tokens must be in sequential ID order for correct placement."""
        tokens = _build_token_list()
        # First token should be at SPECIAL_TOKEN_START_ID (49088)
        # which is <|pad|>
        assert tokens[0].content == "<|pad|>"
        # Token at offset 12 (49100) should be <|fim_prefix|>
        assert tokens[12].content == "<|fim_prefix|>"


class TestRegistrationGuard:
    """Test that register_special_tokens validates preconditions.

    These tests require the tokenizers library; skip if unavailable.
    """

    @pytest.fixture()
    def _skip_if_no_tokenizers(self) -> None:
        pytest.importorskip("tokenizers")

    @pytest.mark.usefixtures("_skip_if_no_tokenizers")
    def test_rejects_oversized_vocab(self) -> None:
        from tokenizers import Tokenizer, models

        from codeforge.tokenizer.special_tokens import register_special_tokens

        # Create a tokenizer and manually add too many tokens
        tok = Tokenizer(models.BPE())
        # Add enough dummy tokens to exceed the target
        tok.add_tokens([f"tok_{i}" for i in range(SPECIAL_TOKEN_START_ID + 1)])

        with pytest.raises(ValueError, match="exceeds the target"):
            register_special_tokens(tok)

    @pytest.mark.usefixtures("_skip_if_no_tokenizers")
    def test_pads_small_vocab(self) -> None:
        from tokenizers import Tokenizer, models, trainers
        from tokenizers.pre_tokenizers import ByteLevel

        from codeforge.tokenizer.special_tokens import register_special_tokens

        # Train a tiny tokenizer — will have far fewer than 49088 tokens
        tok = Tokenizer(models.BPE())
        tok.pre_tokenizer = ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=300,
            initial_alphabet=ByteLevel.alphabet(),
            special_tokens=[],
        )
        tok.train_from_iterator(iter([""]), trainer=trainer)

        # Capture actual size before padding (may differ from alphabet size
        # if BPE learned any merges)
        pre_pad_size = tok.get_vocab_size()

        # Should pad without error and produce exactly VOCAB_SIZE
        register_special_tokens(tok)
        assert tok.get_vocab_size() == VOCAB_SIZE

        # Padding tokens use <|reserved_N|> with absolute IDs
        vocab = tok.get_vocab()
        first_pad = f"<|reserved_{pre_pad_size}|>"
        assert first_pad in vocab
