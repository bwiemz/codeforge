"""HumanEval benchmark runner.

Loads HumanEval problems, generates completions, and evaluates them
by executing test cases in a sandboxed environment.
"""

import json
import subprocess
import sys
import tempfile
import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass


@dataclass
class HumanEvalProblem:
    task_id: str
    prompt: str
    canonical_solution: str
    test: str
    entry_point: str


def load_humaneval(path: Optional[str] = None) -> list[HumanEvalProblem]:
    """Load HumanEval problems from a JSONL file or HuggingFace.

    If no path is provided, attempts to download from HuggingFace.
    """
    problems = []

    if path and Path(path).exists():
        with open(path) as f:
            for line in f:
                data = json.loads(line)
                problems.append(HumanEvalProblem(**{
                    k: data[k] for k in HumanEvalProblem.__dataclass_fields__
                }))
    else:
        # Try loading from HuggingFace datasets
        try:
            from datasets import load_dataset
            ds = load_dataset("openai_humaneval", split="test")
            for item in ds:
                problems.append(HumanEvalProblem(
                    task_id=item["task_id"],
                    prompt=item["prompt"],
                    canonical_solution=item["canonical_solution"],
                    test=item["test"],
                    entry_point=item["entry_point"],
                ))
        except Exception as e:
            print(f"Could not load HumanEval: {e}")
            print("Download manually from: https://github.com/openai/human-eval")

    return problems


def execute_code_safely(
    code: str,
    timeout: int = 10,
) -> tuple[bool, str]:
    """Execute Python code in a subprocess with timeout.

    Args:
        code: Python code to execute
        timeout: Maximum execution time in seconds

    Returns:
        (success, output) tuple
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(code)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        success = result.returncode == 0
        output = result.stdout + result.stderr
        return success, output
    except subprocess.TimeoutExpired:
        return False, "Timeout"
    except Exception as e:
        return False, str(e)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def evaluate_completion(
    problem: HumanEvalProblem,
    completion: str,
    timeout: int = 10,
) -> bool:
    """Evaluate a single completion against HumanEval test cases.

    Args:
        problem: The HumanEval problem
        completion: Generated code completion (function body)
        timeout: Execution timeout in seconds

    Returns:
        True if all test cases pass
    """
    # Construct full program: prompt + completion + tests
    full_code = problem.prompt + completion + "\n\n" + problem.test + f"\n\ncheck({problem.entry_point})\n"

    success, output = execute_code_safely(full_code, timeout=timeout)
    return success


def run_humaneval(
    generator,
    problems: Optional[list[HumanEvalProblem]] = None,
    n_samples: int = 1,
    temperature: float = 0.8,
    max_tokens: int = 512,
    timeout: int = 10,
) -> dict:
    """Run the full HumanEval benchmark.

    Args:
        generator: CodeForgeGenerator instance
        problems: HumanEval problems (loads default if None)
        n_samples: Number of completions per problem
        temperature: Sampling temperature
        max_tokens: Max tokens per completion
        timeout: Execution timeout per test

    Returns:
        Dict with results and metrics
    """
    if problems is None:
        problems = load_humaneval()

    if not problems:
        return {"error": "No HumanEval problems loaded"}

    all_results = []
    detailed = []

    for i, problem in enumerate(problems):
        problem_results = []

        for s in range(n_samples):
            # Generate completion
            completion = generator.generate(
                problem.prompt,
                max_tokens=max_tokens,
                temperature=temperature if n_samples > 1 else 0.0,
                stop_tokens=[generator.tokenizer.EOS_ID],
            )

            # Extract just the completion (remove the prompt)
            completion = completion[len(problem.prompt):]

            # Stop at the next function definition or class (heuristic)
            lines = completion.split("\n")
            trimmed = []
            for line in lines:
                if line.strip().startswith(("def ", "class ")) and trimmed:
                    break
                trimmed.append(line)
            completion = "\n".join(trimmed)

            # Evaluate
            passed = evaluate_completion(problem, completion, timeout=timeout)
            problem_results.append(passed)

        all_results.append(problem_results)
        passed_count = sum(problem_results)
        detailed.append({
            "task_id": problem.task_id,
            "passed": passed_count,
            "total": n_samples,
        })

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(problems)}] {problem.task_id}: {passed_count}/{n_samples}")

    # Compute metrics
    from .metrics import compute_pass_at_k
    k_values = [k for k in [1, 10, 100] if k <= n_samples]
    metrics = compute_pass_at_k(all_results, k_values)

    return {
        "benchmark": "humaneval",
        "n_problems": len(problems),
        "n_samples": n_samples,
        "metrics": metrics,
        "detailed": detailed,
    }
