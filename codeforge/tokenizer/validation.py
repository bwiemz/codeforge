"""Post-training validation for the TCFP tokenizer.

Checks compression ratio, roundtrip fidelity, indentation tokens,
keyword coverage, and operator coverage across target languages.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tokenizers import Tokenizer

from .constants import (
    COMPRESSION_RATIO_MAX,
    COMPRESSION_RATIO_MIN,
    SPECIAL_TOKENS,
    VOCAB_SIZE,
)

# ── Keyword / operator lists for validation ────────────────────────
CROSS_LANGUAGE_KEYWORDS: list[str] = [
    "def", "function", "fn", "func", "void", "int",
    "return", "class", "import", "use",
]

COMMON_OPERATORS: list[str] = [
    "==", "!=", ">=", "<=", "->", "=>", "::", "**", "//",
]

PYTHON_KEYWORDS: list[str] = [
    "def", "class", "import", "from", "return", "if", "else",
    "elif", "for", "while", "try", "except", "with", "as",
    "yield", "lambda", "pass", "break", "continue", "raise",
    "async", "await", "None", "True", "False", "self",
]

JS_KEYWORDS: list[str] = [
    "function", "const", "let", "var", "return", "if", "else",
    "for", "while", "class", "import", "export", "async", "await",
    "try", "catch", "throw", "new", "this", "typeof",
]

RUST_KEYWORDS: list[str] = [
    "fn", "let", "mut", "impl", "struct", "enum", "trait",
    "pub", "use", "mod", "match", "self", "async", "await",
]

GO_KEYWORDS: list[str] = [
    "func", "var", "const", "type", "struct", "interface",
    "return", "if", "else", "for", "range", "switch", "case",
    "go", "chan", "defer", "package", "import",
]


@dataclass
class ValidationReport:
    """Post-training tokenizer validation results."""

    vocab_size: int = 0
    vocab_size_correct: bool = False

    # Compression
    compression_ratio: float = 0.0
    compression_by_language: dict[str, float] = field(default_factory=dict)
    compression_in_range: bool = False

    # Roundtrip
    roundtrip_ok: bool = True
    roundtrip_failures: list[str] = field(default_factory=list)

    # Code tokens
    indentation_tokens: dict[str, bool] = field(default_factory=dict)
    keyword_single_tokens: dict[str, list[str]] = field(default_factory=dict)
    operator_single_tokens: list[str] = field(default_factory=list)
    space_prefixed_keywords: dict[str, list[str]] = field(default_factory=dict)
    space_prefixed_operators: list[str] = field(default_factory=list)

    # Special tokens
    special_tokens_correct: bool = False

    def summary(self) -> str:
        lines = [
            f"Vocab size: {self.vocab_size} (correct={self.vocab_size_correct})",
            f"Compression ratio: {self.compression_ratio:.2f} bytes/token "
            f"(in range={self.compression_in_range})",
            f"Roundtrip fidelity: {self.roundtrip_ok} "
            f"({len(self.roundtrip_failures)} failures)",
            f"Special tokens correct: {self.special_tokens_correct}",
        ]
        if self.indentation_tokens:
            lines.append("Indentation tokens:")
            for name, ok in self.indentation_tokens.items():
                lines.append(f"  {name}: {'single token' if ok else 'MULTI-TOKEN'}")
        if self.operator_single_tokens:
            lines.append(
                f"Operators as single tokens: {', '.join(self.operator_single_tokens)}"
            )
        if self.space_prefixed_operators:
            lines.append(
                f"Space-prefixed operators: {', '.join(self.space_prefixed_operators)}"
            )
        for lang, kws in self.keyword_single_tokens.items():
            lines.append(f"{lang} bare keywords: {', '.join(kws)}")
        if self.space_prefixed_keywords or self.space_prefixed_operators:
            lines.append(
                "Space-prefixed results below are more predictive of "
                "real-world quality"
            )
            lines.append(
                "(keywords in code almost always appear after "
                "indentation/spaces):"
            )
        for lang, kws in self.space_prefixed_keywords.items():
            lines.append(f"  {lang} ' keyword': {', '.join(kws)}")
        return "\n".join(lines)


# ── Individual check functions ─────────────────────────────────────


def compute_compression_ratio(tokenizer: Tokenizer, text: str) -> float:
    """Compute bytes-per-token for *text*."""
    encoded = tokenizer.encode(text)
    byte_len = len(text.encode("utf-8"))
    n_tokens = len(encoded.ids)
    if n_tokens == 0:
        return 0.0
    return byte_len / n_tokens


def check_roundtrip(
    tokenizer: Tokenizer,
    samples: list[str],
) -> tuple[bool, list[str]]:
    """Verify ``decode(encode(text)) == text`` for all *samples*."""
    failures: list[str] = []
    for sample in samples:
        encoded = tokenizer.encode(sample)
        decoded = tokenizer.decode(encoded.ids)
        if decoded != sample:
            failures.append(
                f"MISMATCH: {sample[:60]!r} -> {decoded[:60]!r}"
            )
    return len(failures) == 0, failures


def check_single_token(tokenizer: Tokenizer, text: str) -> bool:
    """Return True if *text* encodes to exactly one token."""
    return len(tokenizer.encode(text).ids) == 1


def check_indentation(tokenizer: Tokenizer) -> dict[str, bool]:
    """Check whether common indentation patterns are single tokens."""
    return {
        "4-space": check_single_token(tokenizer, "    "),
        "2-space": check_single_token(tokenizer, "  "),
        "tab": check_single_token(tokenizer, "\t"),
    }


def check_keywords(
    tokenizer: Tokenizer,
    keywords: list[str],
) -> list[str]:
    """Return keywords from *keywords* that encode as single tokens."""
    return [kw for kw in keywords if check_single_token(tokenizer, kw)]


def check_space_prefixed(
    tokenizer: Tokenizer,
    tokens: list[str],
) -> list[str]:
    """Return *tokens* whose space-prefixed form (`` def``) is a single token.

    In real code, keywords and operators almost always appear after
    indentation or other tokens, so the leading-space variant is
    the one the BPE actually learns most often.
    """
    return [t for t in tokens if check_single_token(tokenizer, f" {t}")]


def check_operators(tokenizer: Tokenizer) -> list[str]:
    """Return operators from ``COMMON_OPERATORS`` that are single tokens."""
    return check_keywords(tokenizer, COMMON_OPERATORS)


def check_special_tokens(tokenizer: Tokenizer) -> bool:
    """Verify every named special token is at its expected ID."""
    vocab = tokenizer.get_vocab()
    for token_str, expected_id in SPECIAL_TOKENS.items():
        if vocab.get(token_str) != expected_id:
            return False
    return True


# ── Orchestrator ───────────────────────────────────────────────────


def validate_tokenizer(
    tokenizer: Tokenizer,
    validation_texts: dict[str, str] | None = None,
    roundtrip_samples: list[str] | None = None,
) -> ValidationReport:
    """Run the full validation suite on a trained tokenizer.

    Parameters
    ----------
    tokenizer
        A trained HuggingFace ``Tokenizer`` instance.
    validation_texts
        Mapping of ``language_name -> code_text`` for per-language
        compression ratio measurement.
    roundtrip_samples
        Strings to test encode-decode roundtrip fidelity.

    Returns
    -------
    ValidationReport
    """
    report = ValidationReport()

    # Vocab size
    report.vocab_size = tokenizer.get_vocab_size()
    report.vocab_size_correct = report.vocab_size == VOCAB_SIZE

    # Compression ratio
    if validation_texts:
        all_text = "\n".join(validation_texts.values())
        report.compression_ratio = compute_compression_ratio(tokenizer, all_text)
        report.compression_by_language = {
            lang: compute_compression_ratio(tokenizer, text)
            for lang, text in validation_texts.items()
        }
        report.compression_in_range = (
            COMPRESSION_RATIO_MIN <= report.compression_ratio <= COMPRESSION_RATIO_MAX
        )

    # Roundtrip fidelity
    if roundtrip_samples:
        report.roundtrip_ok, report.roundtrip_failures = check_roundtrip(
            tokenizer, roundtrip_samples
        )

    # Code tokens
    report.indentation_tokens = check_indentation(tokenizer)
    report.operator_single_tokens = check_operators(tokenizer)

    report.keyword_single_tokens = {
        "cross-language": check_keywords(tokenizer, CROSS_LANGUAGE_KEYWORDS),
        "python": check_keywords(tokenizer, PYTHON_KEYWORDS),
        "javascript": check_keywords(tokenizer, JS_KEYWORDS),
        "rust": check_keywords(tokenizer, RUST_KEYWORDS),
        "go": check_keywords(tokenizer, GO_KEYWORDS),
    }

    # Space-prefixed variants (how keywords actually appear in indented code)
    report.space_prefixed_keywords = {
        "cross-language": check_space_prefixed(tokenizer, CROSS_LANGUAGE_KEYWORDS),
        "python": check_space_prefixed(tokenizer, PYTHON_KEYWORDS),
        "javascript": check_space_prefixed(tokenizer, JS_KEYWORDS),
        "rust": check_space_prefixed(tokenizer, RUST_KEYWORDS),
        "go": check_space_prefixed(tokenizer, GO_KEYWORDS),
    }
    report.space_prefixed_operators = check_space_prefixed(
        tokenizer, COMMON_OPERATORS
    )

    # Special tokens
    report.special_tokens_correct = check_special_tokens(tokenizer)

    return report
