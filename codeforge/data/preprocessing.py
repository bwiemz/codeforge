"""Code preprocessing utilities."""

import random
import re

# Map file extensions to language names
EXTENSION_TO_LANG = {
    ".py": "python", ".js": "javascript", ".ts": "typescript", ".tsx": "typescript",
    ".java": "java", ".cpp": "cpp", ".cc": "cpp", ".c": "c", ".h": "c",
    ".hpp": "cpp", ".cs": "csharp", ".go": "go", ".rs": "rust", ".rb": "ruby",
    ".php": "php", ".swift": "swift", ".kt": "kotlin", ".scala": "scala",
    ".lua": "lua", ".sh": "bash", ".bash": "bash", ".zsh": "bash",
    ".sql": "sql", ".r": "r", ".R": "r", ".m": "matlab", ".jl": "julia",
    ".html": "html", ".css": "css", ".scss": "css", ".xml": "xml",
    ".json": "json", ".yaml": "yaml", ".yml": "yaml", ".toml": "toml",
    ".md": "markdown", ".rst": "markdown",
}

# License header patterns to optionally strip
LICENSE_PATTERNS = [
    re.compile(r"^(/\*.*?\*/\s*\n)", re.DOTALL),  # C-style block comment at top
    re.compile(r"^(#[^\n]*license[^\n]*\n(?:#[^\n]*\n)*)", re.IGNORECASE),  # Python/shell
    re.compile(r"^(//[^\n]*license[^\n]*\n(?://[^\n]*\n)*)", re.IGNORECASE),  # C++ style
]


def detect_language(filename: str) -> str | None:
    """Detect programming language from filename extension."""
    for ext, lang in EXTENSION_TO_LANG.items():
        if filename.endswith(ext):
            return lang
    return None


def normalize_whitespace(code: str, tabs_to_spaces: int | None = 4) -> str:
    """Normalize whitespace in code."""
    if tabs_to_spaces is not None:
        code = code.replace("\t", " " * tabs_to_spaces)
    # Remove trailing whitespace per line
    lines = [line.rstrip() for line in code.split("\n")]
    # Remove excessive blank lines (max 2 consecutive)
    result = []
    blank_count = 0
    for line in lines:
        if line == "":
            blank_count += 1
            if blank_count <= 2:
                result.append(line)
        else:
            blank_count = 0
            result.append(line)
    return "\n".join(result).strip() + "\n"


def strip_license_header(code: str) -> str:
    """Strip common license/copyright headers from the top of files."""
    for pattern in LICENSE_PATTERNS:
        match = pattern.match(code)
        if match and "license" in match.group(1).lower():
            code = code[match.end():]
    return code.lstrip("\n")


def apply_fim_transform(
    code: str,
    fim_rate: float = 0.5,
    fim_prefix_token: str = "<|fim_prefix|>",
    fim_middle_token: str = "<|fim_middle|>",
    fim_suffix_token: str = "<|fim_suffix|>",
) -> str:
    """Apply Fill-in-the-Middle transformation.

    With probability fim_rate, splits code at a random point into
    prefix/middle/suffix and rearranges into FIM format.
    Otherwise returns the code unchanged.
    """
    if random.random() > fim_rate:
        return code

    lines = code.split("\n")
    if len(lines) < 3:
        return code

    # Pick a random span to be the "middle" (what the model must predict)
    n_lines = len(lines)
    start = random.randint(0, n_lines - 2)
    end = random.randint(start + 1, min(start + max(n_lines // 4, 2), n_lines))

    prefix = "\n".join(lines[:start])
    middle = "\n".join(lines[start:end])
    suffix = "\n".join(lines[end:])

    # PSM format (Prefix-Suffix-Middle): prefix, then suffix, then middle to generate
    return f"{fim_prefix_token}{prefix}{fim_suffix_token}{suffix}{fim_middle_token}{middle}"


def preprocess_code(
    code: str,
    filename: str | None = None,
    strip_license: bool = True,
    normalize_ws: bool = True,
    tabs_to_spaces: int | None = 4,
    max_line_length: int = 1000,
    min_lines: int = 3,
    max_lines: int = 10000,
) -> str | None:
    """Full preprocessing pipeline for a code sample.

    Returns None if the sample should be filtered out.
    """
    if not code or not code.strip():
        return None

    # Strip surrogate characters that cause UnicodeEncodeError downstream
    code = code.encode("utf-8", errors="surrogatepass").decode(
        "utf-8", errors="replace"
    )

    if strip_license:
        code = strip_license_header(code)

    if normalize_ws:
        code = normalize_whitespace(code, tabs_to_spaces)

    lines = code.split("\n")

    # Filter by line count
    if len(lines) < min_lines or len(lines) > max_lines:
        return None

    # Filter out files with excessively long lines (likely minified/generated)
    if any(len(line) > max_line_length for line in lines):
        return None

    return code
