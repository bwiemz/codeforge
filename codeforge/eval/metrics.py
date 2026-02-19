"""Evaluation metrics for code generation."""

import math
import itertools
from typing import Optional


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator of pass@k.

    Computes the probability that at least one of k samples passes,
    given n total samples with c correct ones.

    Uses the formula: 1 - C(n-c, k) / C(n, k)
    where C is the binomial coefficient.

    Args:
        n: Total number of samples generated
        c: Number of correct samples
        k: k value for pass@k

    Returns:
        Estimated pass@k probability
    """
    if n - c < k:
        return 1.0
    return 1.0 - math.prod(range(n - c, n - c - k, -1)) / math.prod(range(n, n - k, -1))


def compute_pass_at_k(
    results: list[list[bool]],
    k_values: list[int] = [1, 10, 100],
) -> dict[str, float]:
    """Compute pass@k for a set of problems.

    Args:
        results: For each problem, a list of booleans (True=passed, False=failed)
        k_values: Which k values to compute

    Returns:
        Dict like {"pass@1": 0.35, "pass@10": 0.62, "pass@100": 0.80}
    """
    metrics = {}

    for k in k_values:
        scores = []
        for problem_results in results:
            n = len(problem_results)
            c = sum(problem_results)
            if n >= k:
                scores.append(pass_at_k(n, c, k))
        if scores:
            metrics[f"pass@{k}"] = sum(scores) / len(scores)

    return metrics
