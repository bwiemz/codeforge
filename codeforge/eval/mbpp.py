"""MBPP (Mostly Basic Python Programming) benchmark runner."""

import json
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from .humaneval import execute_code_safely
from .metrics import compute_pass_at_k


@dataclass
class MBPPProblem:
    task_id: int
    text: str          # Natural language description
    code: str          # Canonical solution
    test_list: list    # Test assertions
    test_setup_code: str = ""


def load_mbpp(path: Optional[str] = None, split: str = "test") -> list[MBPPProblem]:
    """Load MBPP problems from file or HuggingFace."""
    problems = []

    if path and Path(path).exists():
        with open(path) as f:
            for line in f:
                data = json.loads(line)
                problems.append(MBPPProblem(
                    task_id=data["task_id"],
                    text=data["text"],
                    code=data["code"],
                    test_list=data["test_list"],
                    test_setup_code=data.get("test_setup_code", ""),
                ))
    else:
        try:
            from datasets import load_dataset
            ds = load_dataset("mbpp", split=split)
            for item in ds:
                problems.append(MBPPProblem(
                    task_id=item["task_id"],
                    text=item["text"],
                    code=item["code"],
                    test_list=item["test_list"],
                    test_setup_code=item.get("test_setup_code", ""),
                ))
        except Exception as e:
            print(f"Could not load MBPP: {e}")

    return problems


def evaluate_mbpp_completion(
    problem: MBPPProblem,
    completion: str,
    timeout: int = 10,
) -> bool:
    """Evaluate a completion against MBPP test cases."""
    # Build test program
    parts = []
    if problem.test_setup_code:
        parts.append(problem.test_setup_code)
    parts.append(completion)
    parts.extend(problem.test_list)

    full_code = "\n".join(parts)
    success, output = execute_code_safely(full_code, timeout=timeout)
    return success


def run_mbpp(
    generator,
    problems: Optional[list[MBPPProblem]] = None,
    n_samples: int = 1,
    temperature: float = 0.8,
    max_tokens: int = 512,
    timeout: int = 10,
) -> dict:
    """Run the full MBPP benchmark."""
    if problems is None:
        problems = load_mbpp()

    if not problems:
        return {"error": "No MBPP problems loaded"}

    all_results = []
    detailed = []

    for i, problem in enumerate(problems):
        prompt = f"# {problem.text}\n"
        problem_results = []

        for s in range(n_samples):
            completion = generator.generate(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature if n_samples > 1 else 0.0,
                stop_tokens=[generator.tokenizer.EOS_ID],
            )
            completion = completion[len(prompt):]

            # Trim at next function definition
            lines = completion.split("\n")
            trimmed = []
            found_def = False
            for line in lines:
                if line.strip().startswith("def "):
                    if found_def:
                        break
                    found_def = True
                trimmed.append(line)
            completion = "\n".join(trimmed)

            passed = evaluate_mbpp_completion(problem, completion, timeout=timeout)
            problem_results.append(passed)

        all_results.append(problem_results)
        detailed.append({
            "task_id": problem.task_id,
            "passed": sum(problem_results),
            "total": n_samples,
        })

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(problems)}]")

    k_values = [k for k in [1, 10, 100] if k <= n_samples]
    metrics = compute_pass_at_k(all_results, k_values)

    return {
        "benchmark": "mbpp",
        "n_problems": len(problems),
        "n_samples": n_samples,
        "metrics": metrics,
        "detailed": detailed,
    }
