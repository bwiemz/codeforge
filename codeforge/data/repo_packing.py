"""Repository-level context packing.

Packs multiple files from the same repository into single training sequences,
preserving directory structure context. This teaches the model to understand
cross-file dependencies and project structure.
"""

import random
from pathlib import Path
from typing import Iterator, Optional


# Special tokens for repo-level packing
REPO_NAME_TOKEN = "<|repo_name|>"
FILE_SEP_TOKEN = "<|file_sep|>"


def format_repo_context(
    repo_name: str,
    files: list[tuple[str, str]],
) -> str:
    """Format multiple files from a repo into a single training sequence.

    Args:
        repo_name: Repository identifier (e.g., "owner/repo")
        files: List of (filepath, content) tuples

    Returns:
        Formatted string with repo structure and file contents
    """
    parts = [f"{REPO_NAME_TOKEN}{repo_name}"]

    for filepath, content in files:
        parts.append(f"{FILE_SEP_TOKEN}{filepath}")
        parts.append(content)

    return "\n".join(parts)


def pack_repo_files(
    repo_name: str,
    files: list[tuple[str, str]],
    max_tokens_estimate: int = 8192,
    chars_per_token: float = 3.5,
) -> Iterator[str]:
    """Pack repository files into sequences that fit within token budget.

    Groups files by directory proximity and packs related files together.
    Yields multiple packed sequences if the repo is large.

    Args:
        repo_name: Repository identifier
        files: All files in the repo as (path, content) tuples
        max_tokens_estimate: Approximate max tokens per packed sequence
        chars_per_token: Estimated chars per token for budget calculation
    """
    max_chars = int(max_tokens_estimate * chars_per_token)

    # Sort files by path to group related files
    sorted_files = sorted(files, key=lambda f: f[0])

    current_batch: list[tuple[str, str]] = []
    current_chars = len(f"{REPO_NAME_TOKEN}{repo_name}\n")

    for filepath, content in sorted_files:
        file_overhead = len(f"{FILE_SEP_TOKEN}{filepath}\n")
        file_chars = file_overhead + len(content)

        # If single file exceeds budget, yield it alone (truncated)
        if file_chars > max_chars:
            if current_batch:
                yield format_repo_context(repo_name, current_batch)
                current_batch = []
                current_chars = len(f"{REPO_NAME_TOKEN}{repo_name}\n")

            truncated = content[: max_chars - file_overhead - 100]
            yield format_repo_context(repo_name, [(filepath, truncated)])
            continue

        # If adding this file exceeds budget, yield current batch
        if current_chars + file_chars > max_chars and current_batch:
            yield format_repo_context(repo_name, current_batch)
            current_batch = []
            current_chars = len(f"{REPO_NAME_TOKEN}{repo_name}\n")

        current_batch.append((filepath, content))
        current_chars += file_chars

    if current_batch:
        yield format_repo_context(repo_name, current_batch)


def iter_local_repo(
    repo_path: str | Path,
    extensions: Optional[set[str]] = None,
    max_file_size: int = 100_000,
) -> Iterator[tuple[str, str]]:
    """Iterate over code files in a local repository.

    Args:
        repo_path: Path to repository root
        extensions: File extensions to include (default: common code files)
        max_file_size: Skip files larger than this (likely generated/minified)

    Yields:
        (relative_path, content) tuples
    """
    if extensions is None:
        extensions = {
            ".py", ".js", ".ts", ".tsx", ".java", ".cpp", ".c", ".h", ".hpp",
            ".go", ".rs", ".rb", ".php", ".cs", ".kt", ".swift", ".scala",
            ".sh", ".bash", ".sql", ".lua", ".jl", ".r",
        }

    repo_path = Path(repo_path)

    for file_path in sorted(repo_path.rglob("*")):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in extensions:
            continue
        if file_path.stat().st_size > max_file_size:
            continue

        # Skip common non-source directories
        parts = file_path.relative_to(repo_path).parts
        skip_dirs = {"node_modules", ".git", "__pycache__", "venv", ".venv",
                     "dist", "build", ".tox", ".eggs", "vendor"}
        if any(p in skip_dirs for p in parts):
            continue

        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            rel_path = str(file_path.relative_to(repo_path)).replace("\\", "/")
            yield rel_path, content
        except Exception:
            continue
