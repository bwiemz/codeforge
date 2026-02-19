"""Tests for the TCFPTokenizer wrapper class."""

from __future__ import annotations

from pathlib import Path

import pytest

tokenizers = pytest.importorskip("tokenizers")

from codeforge.tokenizer.config import TokenizerConfig
from codeforge.tokenizer.constants import SPECIAL_TOKENS, VOCAB_SIZE
from codeforge.tokenizer.training import train_tokenizer
from codeforge.tokenizer.wrapper import TCFPTokenizer

# Minimal corpus for training a test tokenizer
_MINI_CORPUS = "def foo():\n    return 42\n" * 500


@pytest.fixture()
def trained_tokenizer_path(tmp_path: Path) -> Path:
    """Train a small tokenizer and return the path to the JSON file."""
    corpus_file = tmp_path / "corpus.txt"
    corpus_file.write_text(_MINI_CORPUS, encoding="utf-8")

    config = TokenizerConfig(
        output_dir=tmp_path / "tok",
        corpus_paths=[corpus_file],
    )
    train_tokenizer(config)
    return config.output_path


class TestTCFPTokenizer:
    def test_construction(self, trained_tokenizer_path: Path) -> None:
        tok = TCFPTokenizer(trained_tokenizer_path)
        assert tok.vocab_size == VOCAB_SIZE

    def test_alignment_report(self, trained_tokenizer_path: Path) -> None:
        tok = TCFPTokenizer(trained_tokenizer_path)
        assert tok.alignment_report.peak_throughput is True

    def test_special_token_ids(self, trained_tokenizer_path: Path) -> None:
        tok = TCFPTokenizer(trained_tokenizer_path)
        assert tok.pad_id == SPECIAL_TOKENS["<|pad|>"]
        assert tok.bos_id == SPECIAL_TOKENS["<|bos|>"]
        assert tok.eos_id == SPECIAL_TOKENS["<|eos|>"]
        assert tok.unk_id == SPECIAL_TOKENS["<|unk|>"]
        assert tok.fim_prefix_id == SPECIAL_TOKENS["<|fim_prefix|>"]
        assert tok.fim_middle_id == SPECIAL_TOKENS["<|fim_middle|>"]
        assert tok.fim_suffix_id == SPECIAL_TOKENS["<|fim_suffix|>"]
        assert tok.im_start_id == SPECIAL_TOKENS["<|im_start|>"]
        assert tok.im_end_id == SPECIAL_TOKENS["<|im_end|>"]

    def test_encode_decode(self, trained_tokenizer_path: Path) -> None:
        tok = TCFPTokenizer(trained_tokenizer_path)
        text = "def foo():\n    return 42"
        ids = tok.encode(text)
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)
        decoded = tok.decode(ids)
        assert decoded == text

    def test_encode_batch(self, trained_tokenizer_path: Path) -> None:
        tok = TCFPTokenizer(trained_tokenizer_path)
        texts = ["hello world", "def foo():", "return 42"]
        batch_ids = tok.encode_batch(texts)
        assert len(batch_ids) == 3
        for ids in batch_ids:
            assert isinstance(ids, list)

    def test_decode_batch(self, trained_tokenizer_path: Path) -> None:
        tok = TCFPTokenizer(trained_tokenizer_path)
        # Unequal-length inputs to exercise padding in batch encoding
        texts = ["hi", "def foo():"]
        batch_ids = tok.encode_batch(texts)
        # skip_special_tokens strips pad tokens added by batch padding
        decoded = tok.decode_batch(batch_ids, skip_special_tokens=True)
        assert decoded == texts

    def test_inner_property(self, trained_tokenizer_path: Path) -> None:
        tok = TCFPTokenizer(trained_tokenizer_path)
        assert isinstance(tok.inner, tokenizers.Tokenizer)

    # ── HF duck-typing tests ───────────────────────────────────────

    def test_len(self, trained_tokenizer_path: Path) -> None:
        tok = TCFPTokenizer(trained_tokenizer_path)
        assert len(tok) == VOCAB_SIZE

    def test_call_single(self, trained_tokenizer_path: Path) -> None:
        tok = TCFPTokenizer(trained_tokenizer_path)
        result = tok("def foo():")
        assert "input_ids" in result
        assert isinstance(result["input_ids"], list)
        assert all(isinstance(i, int) for i in result["input_ids"])

    def test_call_batch(self, trained_tokenizer_path: Path) -> None:
        tok = TCFPTokenizer(trained_tokenizer_path)
        result = tok(["hello", "world"])
        assert "input_ids" in result
        assert isinstance(result["input_ids"], list)
        assert len(result["input_ids"]) == 2
        # Each element must be a list of ints (not a flat int)
        for inner in result["input_ids"]:
            assert isinstance(inner, list)
            assert all(isinstance(tok_id, int) for tok_id in inner)

    def test_get_vocab(self, trained_tokenizer_path: Path) -> None:
        tok = TCFPTokenizer(trained_tokenizer_path)
        vocab = tok.get_vocab()
        assert isinstance(vocab, dict)
        assert len(vocab) == VOCAB_SIZE
        assert "<|pad|>" in vocab

    def test_convert_tokens_to_ids(self, trained_tokenizer_path: Path) -> None:
        tok = TCFPTokenizer(trained_tokenizer_path)
        pad_id = tok.convert_tokens_to_ids("<|pad|>")
        assert pad_id == SPECIAL_TOKENS["<|pad|>"]
        # Batch conversion
        ids = tok.convert_tokens_to_ids(["<|pad|>", "<|eos|>"])
        assert ids == [SPECIAL_TOKENS["<|pad|>"], SPECIAL_TOKENS["<|eos|>"]]

    def test_convert_ids_to_tokens(self, trained_tokenizer_path: Path) -> None:
        tok = TCFPTokenizer(trained_tokenizer_path)
        token = tok.convert_ids_to_tokens(SPECIAL_TOKENS["<|pad|>"])
        assert token == "<|pad|>"
        # Batch conversion
        tokens = tok.convert_ids_to_tokens(
            [SPECIAL_TOKENS["<|pad|>"], SPECIAL_TOKENS["<|eos|>"]]
        )
        assert tokens == ["<|pad|>", "<|eos|>"]

    # ── Post-processor and padding tests ───────────────────────────

    def test_add_special_tokens_wraps_bos_eos(
        self, trained_tokenizer_path: Path
    ) -> None:
        """encode(add_special_tokens=True) should wrap with BOS/EOS."""
        tok = TCFPTokenizer(trained_tokenizer_path)
        ids = tok.encode("hello", add_special_tokens=True)
        assert ids[0] == tok.bos_id
        assert ids[-1] == tok.eos_id

    def test_add_special_tokens_false_no_bos_eos(
        self, trained_tokenizer_path: Path
    ) -> None:
        """encode(add_special_tokens=False) should NOT add BOS/EOS."""
        tok = TCFPTokenizer(trained_tokenizer_path)
        ids = tok.encode("hello", add_special_tokens=False)
        assert ids[0] != tok.bos_id
        assert ids[-1] != tok.eos_id

    def test_batch_padding(self, trained_tokenizer_path: Path) -> None:
        """Batch encoding should pad shorter sequences to uniform length."""
        tok = TCFPTokenizer(trained_tokenizer_path)
        texts = ["short", "this is a longer sentence with more tokens"]
        batch_ids = tok.encode_batch(texts)
        # All sequences should be same length (padded)
        assert len(batch_ids[0]) == len(batch_ids[1])
        # Shorter sequence should contain pad tokens
        assert tok.pad_id in batch_ids[0]
