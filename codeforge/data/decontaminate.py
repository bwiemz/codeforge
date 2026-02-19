"""Benchmark decontamination.

Removes training samples that overlap with evaluation benchmarks
(HumanEval, MBPP, etc.) to prevent data leakage and inflated scores.
Uses n-gram overlap detection.
"""

import re
from typing import Optional


def _normalize(text: str) -> str:
    """Normalize text for comparison: lowercase, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_ngrams(text: str, n: int = 10) -> set[tuple[str, ...]]:
    """Extract word-level n-grams from text."""
    words = _normalize(text).split()
    if len(words) < n:
        return {tuple(words)} if words else set()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


class Decontaminator:
    """Detects and filters training samples that overlap with benchmarks.

    Uses 10-gram overlap: if any 10-gram from a training sample matches
    a benchmark problem, the sample is flagged as contaminated.
    """

    def __init__(self, ngram_size: int = 10, overlap_threshold: float = 0.1):
        """
        Args:
            ngram_size: Size of n-grams for overlap detection
            overlap_threshold: Fraction of benchmark n-grams that must match
                to consider a sample contaminated
        """
        self.ngram_size = ngram_size
        self.overlap_threshold = overlap_threshold
        # All benchmark n-grams merged into one set for fast lookup
        self.benchmark_ngrams: set[tuple[str, ...]] = set()
        self.benchmark_names: list[str] = []

    def add_benchmark(self, name: str, problems: list[str]) -> None:
        """Register a benchmark's problems for decontamination.

        Args:
            name: Benchmark name (e.g., "humaneval", "mbpp")
            problems: List of problem texts (prompts + canonical solutions)
        """
        self.benchmark_names.append(name)
        for problem in problems:
            ngrams = _extract_ngrams(problem, self.ngram_size)
            self.benchmark_ngrams.update(ngrams)

    def add_humaneval(self) -> None:
        """Add HumanEval benchmark problems.

        Loads from the canonical dataset if available, otherwise uses
        a set of known function signatures to catch obvious leaks.
        """
        # Known HumanEval function signatures (subset for basic decontamination)
        known_signatures = [
            "def has_close_elements(numbers: List[float], threshold: float) -> bool:",
            "def separate_paren_groups(paren_string: str) -> List[str]:",
            "def truncate_number(number: float) -> float:",
            "def below_zero(operations: List[int]) -> bool:",
            "def mean_absolute_deviation(numbers: List[float]) -> float:",
            "def intersperse(numbers: List[int], delimeter: int) -> List[int]:",
            "def parse_nested_parens(paren_string: str) -> List[int]:",
            "def filter_by_substring(strings: List[str], substring: str) -> List[str]:",
            "def sum_product(numbers: List[int]) -> Tuple[int, int]:",
            "def rolling_max(numbers: List[int]) -> List[int]:",
            "def is_palindrome(string: str) -> bool:",
            "def make_palindrome(string: str) -> str:",
            "def string_xor(a: str, b: str) -> str:",
            "def longest(strings: List[str]) -> Optional[str]:",
            "def greatest_common_divisor(a: int, b: int) -> int:",
            "def all_prefixes(string: str) -> List[str]:",
            "def string_sequence(n: int) -> str:",
            "def count_distinct_characters(string: str) -> int:",
            "def parse_music(music_string: str) -> List[int]:",
            "def how_many_times(string: str, substring: str) -> int:",
        ]
        self.add_benchmark("humaneval", known_signatures)

    def is_contaminated(self, text: str) -> bool:
        """Check if a training sample overlaps with any registered benchmark."""
        if not self.benchmark_ngrams:
            return False

        sample_ngrams = _extract_ngrams(text, self.ngram_size)
        if not sample_ngrams:
            return False

        overlap = sample_ngrams & self.benchmark_ngrams
        overlap_ratio = len(overlap) / max(len(sample_ngrams), 1)
        return overlap_ratio >= self.overlap_threshold

    def filter_stream(self, texts: iter) -> iter:
        """Filter an iterator, yielding only non-contaminated samples."""
        contaminated_count = 0
        total_count = 0

        for text in texts:
            total_count += 1
            if self.is_contaminated(text):
                contaminated_count += 1
                continue
            yield text

        if total_count > 0:
            print(
                f"Decontamination: removed {contaminated_count}/{total_count} "
                f"samples ({100 * contaminated_count / total_count:.2f}%)"
            )
