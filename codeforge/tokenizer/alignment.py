"""FP8 tensor core alignment verification for vocabulary sizes.

The critical matmul in any LLM is the output logits projection::

    [batch*seq, hidden_dim]  x  [hidden_dim, vocab_size]

``vocab_size`` appears as the N dimension.  TCFP's Triton kernels tile
this dimension at BLOCK_N = 128 (most configs) or 256 (largest config).
A non-aligned vocab wastes elements at every tile edge.

This module checks whether a given vocab size satisfies all TCFP
alignment requirements and can suggest the nearest valid size.
"""

from __future__ import annotations

from dataclasses import dataclass

from .constants import (
    ALIGNMENT_FACTORS,
    TCFP_BLOCK_SIZE,
    TCFP_MAX_TILE,
    TCFP_SCALED_MM_ALIGN,
    TCFP_TILE_SIZE,
)


@dataclass(frozen=True)
class AlignmentReport:
    """Results of FP8 tensor core alignment verification."""

    vocab_size: int
    scaled_mm_aligned: bool  # divisible by 16  (torch._scaled_mm hard req)
    block_size_aligned: bool  # divisible by 32  (TCFP block scaling)
    tile_128_aligned: bool  # divisible by 128 (primary BLOCK_N)
    tile_256_aligned: bool  # divisible by 256 (largest BLOCK_N config)
    all_factors: dict[int, bool]  # every factor in ALIGNMENT_FACTORS
    waste_per_tile_128: int  # unused elements per 128-wide tile edge
    n_tiles_128: int  # number of complete 128-wide tiles
    peak_throughput: bool  # True when all critical checks pass

    def summary(self) -> str:
        lines = [
            f"Vocab size: {self.vocab_size}",
            f"  _scaled_mm aligned (÷16):  {self.scaled_mm_aligned}",
            f"  Block-scale aligned (÷32): {self.block_size_aligned}",
            f"  Tile-128 aligned (÷128):   {self.tile_128_aligned}",
            f"  Tile-256 aligned (÷256):   {self.tile_256_aligned}",
            f"  128-tile waste:            {self.waste_per_tile_128} elements",
            f"  Complete 128-tiles:        {self.n_tiles_128}",
            f"  Peak throughput:           {self.peak_throughput}",
        ]
        return "\n".join(lines)


def check_alignment(vocab_size: int) -> AlignmentReport:
    """Verify FP8 tensor core alignment properties for *vocab_size*."""
    factors = {f: vocab_size % f == 0 for f in ALIGNMENT_FACTORS}
    remainder = vocab_size % TCFP_TILE_SIZE
    waste = (TCFP_TILE_SIZE - remainder) % TCFP_TILE_SIZE

    return AlignmentReport(
        vocab_size=vocab_size,
        scaled_mm_aligned=vocab_size % TCFP_SCALED_MM_ALIGN == 0,
        block_size_aligned=vocab_size % TCFP_BLOCK_SIZE == 0,
        tile_128_aligned=remainder == 0,
        tile_256_aligned=vocab_size % TCFP_MAX_TILE == 0,
        all_factors=factors,
        waste_per_tile_128=waste,
        n_tiles_128=vocab_size // TCFP_TILE_SIZE,
        peak_throughput=(
            vocab_size % TCFP_SCALED_MM_ALIGN == 0
            and vocab_size % TCFP_BLOCK_SIZE == 0
            and remainder == 0
        ),
    )


def suggest_vocab_size(
    target_bpe_merges: int,
    num_special: int = 64,
) -> int:
    """Return the smallest TCFP-aligned vocab size >= *target_bpe_merges* + *num_special*.

    Rounds up to the next multiple of 256 (largest Triton BLOCK_N config)
    so every autotune variant tiles cleanly.
    """
    minimum = target_bpe_merges + num_special
    alignment = TCFP_MAX_TILE  # 256
    return ((minimum + alignment - 1) // alignment) * alignment
