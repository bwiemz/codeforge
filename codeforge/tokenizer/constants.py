"""Central constants tying the tokenizer to TCFP FP8 tensor core alignment.

Every numeric constant that affects vocabulary geometry, special-token
placement, or hardware alignment lives here so that all other modules
import from a single source of truth.
"""

from __future__ import annotations

# ── Vocabulary geometry (TCFP-aligned) ─────────────────────────────
# 49_152 = 384 * 128 = 192 * 256 = 48 * 1024
# Divisible by every TCFP tile size and block-scaling factor.
VOCAB_SIZE: int = 49_152
BPE_VOCAB_SIZE: int = 49_088  # 256 byte-level base + 48_832 learned merges
NUM_SPECIAL_TOKENS: int = 64
SPECIAL_TOKEN_START_ID: int = BPE_VOCAB_SIZE  # 49_088

# ── TCFP alignment factors ─────────────────────────────────────────
# Sourced from tcfp/core.py (DEFAULT_BLOCK_SIZE=32, VALID_BLOCK_SIZES),
# tcfp/nn/__init__.py (dims%16 for _scaled_mm),
# tcfp/kernels.py (autotune BLOCK_N configs: 64, 128, 256).
TCFP_BLOCK_SIZE: int = 32
TCFP_SCALED_MM_ALIGN: int = 16
TCFP_TILE_SIZE: int = 128
TCFP_MAX_TILE: int = 256
ALIGNMENT_FACTORS: tuple[int, ...] = (16, 32, 64, 128, 256, 512, 1024)

# ── Pre-tokenization regex ─────────────────────────────────────────
# GPT-4 cl100k_base pattern with \p{N} (individual digits) instead
# of \p{N}{1,3}.  Matches StarCoder2, Qwen2.5-Coder, CodeGemma.
#
# Components:
#   1. (?i:'s|'t|'re|'ve|'m|'ll|'d)  — English contractions
#   2. [^\r\n\p{L}\p{N}]?\p{L}+      — Words w/ optional leading punct
#   3. \p{N}                           — Individual digits
#   4.  ?[^\s\p{L}\p{N}]+[\r\n]*      — Punctuation + trailing newlines
#   5. \s*[\r\n]+                      — Whitespace ending in newline
#   6. \s+(?!\S)                       — Trailing whitespace
#   7. \s+                             — Fallback whitespace
PRETOKENIZE_REGEX: str = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    r"|[^\r\n\p{L}\p{N}]?\p{L}+"
    r"|\p{N}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)

# ── Special token definitions ──────────────────────────────────────
# Pipe-delimited <|name|> format (industry standard: Qwen2.5, CodeLlama).
# End-of-vocabulary placement with reserved gaps between groups.
SPECIAL_TOKENS: dict[str, int] = {
    # Basic control (49088-49091, reserved 49092-49099)
    "<|pad|>": 49_088,
    "<|bos|>": 49_089,
    "<|eos|>": 49_090,
    "<|unk|>": 49_091,
    # Fill-in-the-Middle (49100-49103, reserved 49104-49109)
    "<|fim_prefix|>": 49_100,
    "<|fim_middle|>": 49_101,
    "<|fim_suffix|>": 49_102,
    "<|fim_pad|>": 49_103,
    # Repository context (49110-49111, reserved 49112-49119)
    "<|repo_name|>": 49_110,
    "<|file_sep|>": 49_111,
    # ChatML (49120-49121, reserved 49122-49129)
    "<|im_start|>": 49_120,
    "<|im_end|>": 49_121,
    # Code block boundaries (49130-49131, reserved 49132-49151)
    "<|code_start|>": 49_130,
    "<|code_end|>": 49_131,
}

# ── Validation thresholds ──────────────────────────────────────────
COMPRESSION_RATIO_MIN: float = 3.5  # bytes/token minimum target
COMPRESSION_RATIO_MAX: float = 4.5  # bytes/token maximum target

# ── ChatML template ────────────────────────────────────────────────
CHATML_TEMPLATE: str = (
    "{% for message in messages %}"
    "<|im_start|>{{ message['role'] }}\n"
    "{{ message['content'] }}<|im_end|>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)
