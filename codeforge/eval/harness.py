"""Evaluation harness — orchestrates running benchmarks and collecting results."""

import json
import time
from pathlib import Path
from typing import Optional

from .humaneval import run_humaneval, load_humaneval
from .mbpp import run_mbpp, load_mbpp


class EvalHarness:
    """Orchestrates benchmark evaluation for CodeForge models."""

    def __init__(self, generator, output_dir: str = "eval_results"):
        self.generator = generator
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        benchmarks: list[str] = ["humaneval", "mbpp"],
        n_samples: int = 1,
        temperature: float = 0.8,
        max_tokens: int = 512,
    ) -> dict:
        """Run specified benchmarks and save results.

        Args:
            benchmarks: List of benchmark names to run
            n_samples: Number of completions per problem
            temperature: Sampling temperature
            max_tokens: Max tokens per completion

        Returns:
            Combined results dict
        """
        results = {}
        start = time.time()

        for bench_name in benchmarks:
            print(f"\n{'='*60}")
            print(f"Running {bench_name}...")
            print(f"{'='*60}")

            bench_start = time.time()

            if bench_name == "humaneval":
                result = run_humaneval(
                    self.generator,
                    n_samples=n_samples,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            elif bench_name == "mbpp":
                result = run_mbpp(
                    self.generator,
                    n_samples=n_samples,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            else:
                print(f"Unknown benchmark: {bench_name}")
                continue

            elapsed = time.time() - bench_start
            result["elapsed_seconds"] = elapsed
            results[bench_name] = result

            # Print summary
            if "metrics" in result:
                print(f"\n{bench_name} results ({elapsed:.1f}s):")
                for metric, value in result["metrics"].items():
                    print(f"  {metric}: {value:.4f} ({value*100:.2f}%)")

        # Save results
        total_elapsed = time.time() - start
        output = {
            "model_params": self.generator.model.param_count(),
            "total_elapsed_seconds": total_elapsed,
            "benchmarks": results,
        }

        output_path = self.output_dir / f"eval_{int(time.time())}.json"
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {output_path}")

        return output
