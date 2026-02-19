"""Train the CodeForge BPE tokenizer on a code corpus."""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from codeforge.tokenizer.trainer import train_from_files, train_from_iterator


def main():
    parser = argparse.ArgumentParser(description="Train CodeForge tokenizer")
    parser.add_argument(
        "--input", "-i",
        nargs="+",
        default=None,
        help="Input files or directories containing code",
    )
    parser.add_argument(
        "--output", "-o",
        default="tokenizer/codeforge.json",
        help="Output path for trained tokenizer (default: tokenizer/codeforge.json)",
    )
    parser.add_argument(
        "--vocab-size", "-v",
        type=int,
        default=32000,
        help="Vocabulary size (default: 32000)",
    )
    parser.add_argument(
        "--from-hf",
        type=str,
        default=None,
        help="HuggingFace dataset name (e.g., 'bigcode/starcoderdata')",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=None,
        help="Languages to include (e.g., python javascript typescript java)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=500_000,
        help="Max samples for tokenizer training (default: 500000)",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace token for gated datasets (persisted after first use)",
    )
    args = parser.parse_args()

    # HuggingFace login if token provided
    if args.hf_token:
        from huggingface_hub import login
        login(token=args.hf_token)
        print("Logged in to HuggingFace Hub")

    if args.from_hf:
        # Train from HuggingFace dataset
        try:
            from datasets import load_dataset
        except ImportError:
            print("Error: `datasets` package required. Install with: pip install datasets")
            sys.exit(1)

        print(f"Loading dataset: {args.from_hf}")
        if args.languages:
            print(f"Languages: {', '.join(args.languages)}")
        print(f"Max samples: {args.num_samples:,}")

        # starcoderdata supports data_dir for per-language loading
        # Try loading per-language subsets if languages specified
        datasets = []
        if args.languages and "starcoderdata" in args.from_hf:
            for lang in args.languages:
                try:
                    ds = load_dataset(
                        args.from_hf, data_dir=lang,
                        split="train", streaming=True,
                    )
                    datasets.append((lang, ds))
                    print(f"  Loaded stream for: {lang}")
                except Exception as e:
                    print(f"  Warning: Could not load '{lang}': {e}")

        if not datasets:
            # Fallback: load full dataset and filter by lang column
            ds = load_dataset(args.from_hf, split="train", streaming=True)
            datasets.append(("all", ds))
            print("  Loaded full dataset stream")

        languages_lower = [l.lower() for l in args.languages] if args.languages else None

        def text_iterator():
            count = 0
            # Round-robin across language datasets for diversity
            iters = [(lang, iter(ds)) for lang, ds in datasets]
            exhausted = set()

            while count < args.num_samples and len(exhausted) < len(iters):
                for idx, (lang, it) in enumerate(iters):
                    if idx in exhausted:
                        continue
                    try:
                        sample = next(it)
                    except StopIteration:
                        exhausted.add(idx)
                        print(f"  Stream '{lang}' exhausted at {count:,} total samples")
                        continue

                    # Filter by language if needed (for non-per-language datasets)
                    if languages_lower and lang == "all":
                        sample_lang = (
                            sample.get("lang") or sample.get("language") or ""
                        ).lower()
                        if sample_lang not in languages_lower:
                            continue

                    text = (
                        sample.get("content")
                        or sample.get("code")
                        or sample.get("text", "")
                    )
                    if text and len(text.strip()) > 50:
                        yield text
                        count += 1
                        if count % 10_000 == 0:
                            print(f"  Processed {count:,} / {args.num_samples:,} samples...")
                        if count >= args.num_samples:
                            break

        print("Training tokenizer...")
        train_from_iterator(text_iterator(), args.output, args.vocab_size)

    elif args.input:
        # Collect all code files from input paths
        files = []
        extensions = {".py", ".js", ".ts", ".java", ".cpp", ".c", ".h", ".go", ".rs", ".rb"}
        for path_str in args.input:
            path = Path(path_str)
            if path.is_file():
                files.append(path)
            elif path.is_dir():
                for ext in extensions:
                    files.extend(path.rglob(f"*{ext}"))

        if not files:
            print(f"Error: No code files found in {args.input}")
            sys.exit(1)

        print(f"Training tokenizer on {len(files)} files...")
        train_from_files(files, args.output, args.vocab_size)
    else:
        print("Error: Provide --input paths or --from-hf dataset name")
        sys.exit(1)


if __name__ == "__main__":
    main()
