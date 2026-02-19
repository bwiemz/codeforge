"""Interactive code generation with CodeForge."""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from codeforge.inference.generator import CodeForgeGenerator


def main():
    parser = argparse.ArgumentParser(description="Generate code with CodeForge")
    parser.add_argument(
        "--checkpoint", "-c",
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--tokenizer", "-t",
        required=True,
        help="Path to tokenizer JSON file",
    )
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        default=None,
        help="Input prompt (if not provided, enters interactive mode)",
    )
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument(
        "--fim",
        action="store_true",
        help="Fill-in-the-Middle mode (provide prefix and suffix)",
    )
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else None
    print("Loading model...")
    generator = CodeForgeGenerator.from_checkpoint(
        args.checkpoint, args.tokenizer, device
    )
    print(f"Model loaded ({generator.model.param_count():,} params) on {generator.device}")

    if args.prompt:
        # Single prompt mode
        if args.fim:
            # Expect prompt format: "prefix|||suffix"
            parts = args.prompt.split("|||")
            if len(parts) != 2:
                print("FIM mode requires prompt format: 'prefix|||suffix'")
                sys.exit(1)
            result = generator.generate_fim(
                parts[0], parts[1],
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            print(f"\n--- Infill ---\n{result}\n---")
        else:
            for token in generator.stream_generate(
                args.prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
            ):
                print(token, end="", flush=True)
            print()
    else:
        # Interactive mode
        print("\nCodeForge Interactive Mode")
        print("Type your prompt and press Enter. Type 'quit' to exit.")
        print("Use '/fim prefix|||suffix' for fill-in-the-middle.\n")

        while True:
            try:
                prompt = input(">>> ")
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if prompt.strip().lower() == "quit":
                break

            if not prompt.strip():
                continue

            if prompt.startswith("/fim "):
                fim_text = prompt[5:]
                parts = fim_text.split("|||")
                if len(parts) != 2:
                    print("Usage: /fim prefix|||suffix")
                    continue
                result = generator.generate_fim(
                    parts[0], parts[1],
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                )
                print(f"\n{result}\n")
            else:
                for token in generator.stream_generate(
                    prompt,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                ):
                    print(token, end="", flush=True)
                print("\n")


if __name__ == "__main__":
    main()
