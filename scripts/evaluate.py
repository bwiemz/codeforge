"""Evaluation entry point for CodeForge."""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from codeforge.inference.generator import CodeForgeGenerator
from codeforge.eval.harness import EvalHarness


def main():
    parser = argparse.ArgumentParser(description="Evaluate CodeForge model")
    parser.add_argument("--checkpoint", "-c", required=True, help="Model checkpoint path")
    parser.add_argument("--tokenizer", "-t", required=True, help="Tokenizer JSON path")
    parser.add_argument(
        "--benchmarks", "-b", nargs="+", default=["humaneval"],
        choices=["humaneval", "mbpp"], help="Benchmarks to run",
    )
    parser.add_argument("--n-samples", "-n", type=int, default=1, help="Samples per problem")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--output-dir", "-o", default="eval_results")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else None
    print("Loading model...")
    generator = CodeForgeGenerator.from_checkpoint(
        args.checkpoint, args.tokenizer, device
    )
    print(f"Model loaded ({generator.model.param_count():,} params) on {generator.device}")

    harness = EvalHarness(generator, output_dir=args.output_dir)
    results = harness.run(
        benchmarks=args.benchmarks,
        n_samples=args.n_samples,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()
