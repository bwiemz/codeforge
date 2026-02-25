"""Code quality scoring and filtering.

Uses AST parsing (when available) and heuristic rules to score code quality.
Higher quality code should be sampled more frequently during training.
"""

import re


# Heuristic quality signals
def compute_quality_score(
    code: str,
    language: str | None = None,
    max_score: float = 1.0,
) -> float:
    """Compute a quality score for a code sample in [0, max_score].

    Scoring factors:
    - Syntactic validity (AST parse success)
    - Comment/docstring ratio (moderate is best)
    - Average line length (not too short, not too long)
    - Function/class presence (structured code scores higher)
    - Naming conventions (descriptive names score higher)
    - Repetition ratio (low repetition is better)
    """
    if not code or not code.strip():
        return 0.0

    lines = code.strip().split("\n")
    n_lines = len(lines)

    if n_lines < 3:
        return 0.0

    scores = []

    # 1. Line length distribution (prefer 20-100 char avg)
    avg_len = sum(len(ln) for ln in lines) / n_lines
    if 20 <= avg_len <= 100:
        scores.append(1.0)
    elif avg_len < 10 or avg_len > 200:
        scores.append(0.2)
    else:
        scores.append(0.6)

    # 2. Comment ratio (prefer 5-30% comment lines)
    comment_patterns = {
        "python": r"^\s*#",
        "javascript": r"^\s*//",
        "typescript": r"^\s*//",
        "java": r"^\s*//",
        "cpp": r"^\s*//",
        "c": r"^\s*//",
        "go": r"^\s*//",
        "rust": r"^\s*//",
    }
    pattern = comment_patterns.get(language, r"^\s*(#|//)")
    comment_lines = sum(1 for ln in lines if re.match(pattern, ln))
    comment_ratio = comment_lines / n_lines
    if 0.05 <= comment_ratio <= 0.30:
        scores.append(1.0)
    elif comment_ratio > 0.5:  # Over-commented
        scores.append(0.3)
    elif comment_ratio == 0:
        scores.append(0.6)  # No comments is okay for short code
    else:
        scores.append(0.7)

    # 3. Structure: presence of functions/classes
    struct_patterns = [
        r"\bdef\s+\w+",       # Python functions
        r"\bclass\s+\w+",     # Classes
        r"\bfunction\s+\w+",  # JS functions
        r"\bfn\s+\w+",        # Rust functions
        r"\bfunc\s+\w+",      # Go functions
        r"(public|private|protected)\s+\w+\s+\w+\s*\(",  # Java/C# methods
    ]
    has_structure = any(re.search(p, code) for p in struct_patterns)
    scores.append(1.0 if has_structure else 0.5)

    # 4. Repetition ratio (lines that appear more than twice)
    line_counts = {}
    for ln in lines:
        stripped = ln.strip()
        if stripped and len(stripped) > 5:  # Skip short/empty lines
            line_counts[stripped] = line_counts.get(stripped, 0) + 1
    repeated = sum(c - 1 for c in line_counts.values() if c > 2)
    repetition_ratio = repeated / max(n_lines, 1)
    scores.append(max(0.0, 1.0 - repetition_ratio * 3))

    # 5. Identifier quality: average identifier length > 3 chars
    identifiers = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", code)
    if identifiers:
        avg_id_len = sum(len(i) for i in identifiers) / len(identifiers)
        scores.append(min(1.0, avg_id_len / 5.0))
    else:
        scores.append(0.3)

    # 6. Syntax check for Python
    if language == "python":
        try:
            compile(code, "<string>", "exec")
            scores.append(1.0)
        except (SyntaxError, UnicodeEncodeError, ValueError):
            scores.append(0.1)  # Syntax error or bad encoding — very low quality

    # Weighted average
    return min(max_score, sum(scores) / len(scores))


def filter_by_quality(
    code: str,
    language: str | None = None,
    min_score: float = 0.3,
) -> str | None:
    """Return code if it passes quality threshold, else None."""
    score = compute_quality_score(code, language)
    return code if score >= min_score else None


def compute_complexity_score(code: str) -> float:
    """Estimate code complexity (higher = more complex/interesting).

    Simple proxy using control flow keywords and nesting depth.
    """
    complexity = 0

    # Control flow density
    control_keywords = [
        r"\bif\b", r"\bfor\b", r"\bwhile\b", r"\btry\b", r"\bexcept\b",
        r"\bswitch\b", r"\bcase\b", r"\bmatch\b", r"\basync\b", r"\bawait\b",
    ]
    for kw in control_keywords:
        complexity += len(re.findall(kw, code)) * 0.5

    # Nesting depth (indentation levels)
    max_indent = 0
    for line in code.split("\n"):
        stripped = line.lstrip()
        if stripped:
            indent = len(line) - len(stripped)
            max_indent = max(max_indent, indent)
    complexity += min(max_indent / 4, 5)  # Cap at 5 levels

    # Normalize to [0, 1]
    return min(1.0, complexity / 20.0)
