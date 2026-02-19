"""Tests for FP8 tensor core alignment verification."""

from __future__ import annotations

from codeforge.tokenizer.alignment import check_alignment, suggest_vocab_size
from codeforge.tokenizer.constants import VOCAB_SIZE


class TestCheckAlignment:
    def test_49152_is_fully_aligned(self) -> None:
        report = check_alignment(VOCAB_SIZE)
        assert report.peak_throughput is True
        assert report.scaled_mm_aligned is True
        assert report.block_size_aligned is True
        assert report.tile_128_aligned is True
        assert report.tile_256_aligned is True
        assert report.waste_per_tile_128 == 0
        assert report.n_tiles_128 == 384

    def test_50000_not_tile_aligned(self) -> None:
        report = check_alignment(50_000)
        assert report.tile_128_aligned is False
        assert report.peak_throughput is False
        assert report.waste_per_tile_128 > 0

    def test_50257_gpt2_poor_alignment(self) -> None:
        report = check_alignment(50_257)
        assert report.scaled_mm_aligned is False
        assert report.block_size_aligned is False
        assert report.tile_128_aligned is False
        assert report.peak_throughput is False

    def test_32768_is_aligned(self) -> None:
        report = check_alignment(32_768)
        assert report.peak_throughput is True
        assert report.tile_256_aligned is True

    def test_all_factors_dict(self) -> None:
        report = check_alignment(VOCAB_SIZE)
        for factor, aligned in report.all_factors.items():
            assert aligned is True, f"49152 not divisible by {factor}"

    def test_summary_is_string(self) -> None:
        report = check_alignment(VOCAB_SIZE)
        s = report.summary()
        assert isinstance(s, str)
        assert "49152" in s


class TestSuggestVocabSize:
    def test_exact_fit(self) -> None:
        # 49088 + 64 = 49152, already aligned to 256
        assert suggest_vocab_size(49_088, 64) == 49_152

    def test_rounds_up(self) -> None:
        # 49000 + 64 = 49064, next multiple of 256 = 49152
        assert suggest_vocab_size(49_000, 64) == 49_152

    def test_small_input(self) -> None:
        result = suggest_vocab_size(100, 64)
        assert result % 256 == 0
        assert result >= 164

    def test_large_input(self) -> None:
        result = suggest_vocab_size(100_000, 64)
        assert result % 256 == 0
        assert result >= 100_064
