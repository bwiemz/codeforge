"""Tests for vocabulary geometry and special token invariants."""

from __future__ import annotations

from codeforge.tokenizer.constants import (
    ALIGNMENT_FACTORS,
    BPE_VOCAB_SIZE,
    NUM_SPECIAL_TOKENS,
    SPECIAL_TOKEN_START_ID,
    SPECIAL_TOKENS,
    VOCAB_SIZE,
)


class TestVocabGeometry:
    def test_vocab_size_composition(self) -> None:
        assert VOCAB_SIZE == BPE_VOCAB_SIZE + NUM_SPECIAL_TOKENS

    def test_special_token_start_id(self) -> None:
        assert SPECIAL_TOKEN_START_ID == BPE_VOCAB_SIZE

    def test_divisibility_by_all_alignment_factors(self) -> None:
        for factor in ALIGNMENT_FACTORS:
            assert VOCAB_SIZE % factor == 0, (
                f"VOCAB_SIZE ({VOCAB_SIZE}) not divisible by {factor}"
            )

    def test_specific_factorizations(self) -> None:
        assert VOCAB_SIZE == 384 * 128
        assert VOCAB_SIZE == 192 * 256
        assert VOCAB_SIZE == 48 * 1024

    def test_num_special_tokens(self) -> None:
        assert NUM_SPECIAL_TOKENS == 64


class TestSpecialTokens:
    def test_all_ids_in_range(self) -> None:
        for token, token_id in SPECIAL_TOKENS.items():
            assert SPECIAL_TOKEN_START_ID <= token_id < VOCAB_SIZE, (
                f"{token} ID {token_id} out of range "
                f"[{SPECIAL_TOKEN_START_ID}, {VOCAB_SIZE})"
            )

    def test_no_duplicate_ids(self) -> None:
        ids = list(SPECIAL_TOKENS.values())
        assert len(ids) == len(set(ids)), "Duplicate special token IDs"

    def test_no_duplicate_names(self) -> None:
        names = list(SPECIAL_TOKENS.keys())
        assert len(names) == len(set(names)), "Duplicate special token names"

    def test_pipe_delimited_format(self) -> None:
        for token in SPECIAL_TOKENS:
            assert token.startswith("<|") and token.endswith("|>"), (
                f"Token {token!r} not in <|name|> format"
            )

    def test_basic_control_tokens_present(self) -> None:
        required = ["<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>"]
        for tok in required:
            assert tok in SPECIAL_TOKENS

    def test_fim_tokens_present(self) -> None:
        required = ["<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>", "<|fim_pad|>"]
        for tok in required:
            assert tok in SPECIAL_TOKENS

    def test_chatml_tokens_present(self) -> None:
        assert "<|im_start|>" in SPECIAL_TOKENS
        assert "<|im_end|>" in SPECIAL_TOKENS

    def test_expected_ids(self) -> None:
        assert SPECIAL_TOKENS["<|pad|>"] == 49_088
        assert SPECIAL_TOKENS["<|bos|>"] == 49_089
        assert SPECIAL_TOKENS["<|eos|>"] == 49_090
        assert SPECIAL_TOKENS["<|fim_prefix|>"] == 49_100
        assert SPECIAL_TOKENS["<|im_start|>"] == 49_120
        assert SPECIAL_TOKENS["<|code_start|>"] == 49_130
