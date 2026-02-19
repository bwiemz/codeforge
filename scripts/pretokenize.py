"""Pre-tokenize dataset to memmap files for fast training.

Runs the full data pipeline (HF streaming → DataMixer quality/dedup/decontam/FIM
→ tokenization) and saves token IDs to a flat uint16 memmap file + sample index.
Training then loads from these files instead of streaming from HuggingFace.

Usage:
  python scripts/pretokenize.py \
    --config configs/pretrain_150m.yaml \
    --output data/pretokenized/150m \
    --max-tokens 30000000000

Output files:
  tokens.bin    — flat uint16 memmap of all token IDs
  index.npy     — (N, 2) int64 array of (offset, length) per sample
  metadata.json — config snapshot, filtering stats, creation info
"""

import argparse
import json
import struct
import sys
import time
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from codeforge.data.mix import DataMixer
from codeforge.tokenizer.tokenizer import CodeForgeTokenizer


def create_language_streams(
    hf_dataset: str,
    languages: list[str],
    hf_dataset_kwargs: dict | None = None,
) -> dict[str, Iterator[str]]:
    """Create per-language iterators from HuggingFace dataset.

    Mirrors MixedCodeDataset._create_language_streams exactly.
    """
    from datasets import load_dataset

    hf_dataset_kwargs = hf_dataset_kwargs or {}
    streams: dict[str, Iterator[str]] = {}

    for lang in languages:
        try:
            ds = load_dataset(
                hf_dataset,
                data_dir=lang,
                split="train",
                streaming=True,
                **hf_dataset_kwargs,
            )

            def make_iterator(dataset):  # noqa: ANN001
                for sample in dataset:
                    content = (
                        sample.get("content")
                        or sample.get("code")
                        or sample.get("text", "")
                    )
                    if content and len(content.strip()) > 50:
                        yield content

            streams[lang] = make_iterator(ds)
        except Exception as e:
            print(f"  Warning: Could not load '{lang}' via data_dir: {e}")

    if not streams:
        print("  Falling back to full dataset with lang column filtering...")
        for lang in languages:
            ds = load_dataset(
                hf_dataset,
                split="train",
                streaming=True,
                **hf_dataset_kwargs,
            )

            def make_filtered_iter(dataset, target_lang: str):  # noqa: ANN001
                for sample in dataset:
                    sample_lang = (
                        sample.get("lang")
                        or sample.get("language")
                        or ""
                    ).lower()
                    if sample_lang != target_lang:
                        continue
                    content = (
                        sample.get("content")
                        or sample.get("code")
                        or sample.get("text", "")
                    )
                    if content and len(content.strip()) > 50:
                        yield content

            streams[lang] = make_filtered_iter(ds, lang.lower())

    return streams


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-tokenize dataset to memmap files"
    )
    parser.add_argument(
        "--config", "-c", type=str, required=True,
        help="Training config YAML (same as train.py uses)",
    )
    parser.add_argument(
        "--output", "-o", type=str, required=True,
        help="Output directory for memmap files",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=30_000_000_000,
        help="Stop after accumulating this many tokens (default: 30B)",
    )
    parser.add_argument(
        "--hf-token", type=str, default=None,
        help="HuggingFace token for gated datasets",
    )
    args = parser.parse_args()

    # HuggingFace login
    if args.hf_token:
        from huggingface_hub import login
        login(token=args.hf_token)
        print("Logged in to HuggingFace Hub")

    # Load config
    with open(args.config) as f:
        raw_config = yaml.safe_load(f)
    data_config = raw_config.get("data", {})

    # Init tokenizer
    tokenizer_path = raw_config.get("tokenizer_path")
    if not tokenizer_path:
        print("Error: No tokenizer_path in config")
        sys.exit(1)

    tokenizer_path = Path(tokenizer_path)
    if not tokenizer_path.exists():
        print(f"Error: Tokenizer not found at {tokenizer_path}")
        sys.exit(1)

    tokenizer = CodeForgeTokenizer(str(tokenizer_path))

    # Init DataMixer (identical to MixedCodeDataset)
    language_weights = data_config.get("language_weights")
    mixer = DataMixer(
        language_weights=language_weights,
        quality_threshold=data_config.get("quality_threshold", 0.3),
        enable_dedup=data_config.get("enable_dedup", True),
        dedup_threshold=data_config.get("dedup_threshold", 0.8),
        enable_decontamination=data_config.get("enable_decontamination", True),
        fim_rate=data_config.get("fim_rate", 0.5),
        seed=data_config.get("seed", 42),
    )

    # Create language streams
    hf_dataset = data_config.get("hf_dataset", "bigcode/starcoderdata")
    languages = data_config.get("languages", [
        "python", "javascript", "typescript", "java", "cpp", "c", "go", "rust",
    ])

    print("\nPre-tokenization config:")
    print(f"  Dataset: {hf_dataset}")
    print(f"  Languages: {', '.join(languages)}")
    print(f"  Quality threshold: {data_config.get('quality_threshold', 0.3)}")
    print(f"  FIM rate: {data_config.get('fim_rate', 0.5)}")
    print(f"  Max tokens: {args.max_tokens:,}")
    print(f"  Output: {args.output}")
    print(f"  Tokenizer: {tokenizer_path} (vocab={tokenizer.vocab_size})")

    streams = create_language_streams(hf_dataset, languages)
    if not streams:
        print("Error: No language streams could be created")
        sys.exit(1)
    print(f"  Streams: {', '.join(streams.keys())}")

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokens_path = output_dir / "tokens.bin"
    index_path = output_dir / "index.npy"
    meta_path = output_dir / "metadata.json"

    # Stream, filter, tokenize, write
    print("\nPre-tokenizing...")
    start_time = time.time()

    mixed = mixer.mix_streams(streams)
    total_tokens = 0
    total_samples = 0
    index_entries: list[tuple[int, int]] = []

    with open(tokens_path, "wb") as f:
        for code in mixed:
            ids = tokenizer.encode(code, add_special_tokens=False)
            if len(ids) == 0:
                continue

            # Write token IDs as uint16
            offset = total_tokens
            length = len(ids)
            f.write(struct.pack(f"<{length}H", *ids))

            index_entries.append((offset, length))
            total_tokens += length
            total_samples += 1

            # Progress reporting
            if total_samples % 100_000 == 0:
                elapsed = time.time() - start_time
                tps = total_tokens / elapsed if elapsed > 0 else 0
                gb_written = (total_tokens * 2) / (1024**3)
                print(
                    f"  {total_samples:>10,} samples | "
                    f"{total_tokens:>13,} tokens | "
                    f"{gb_written:.2f} GB | "
                    f"{tps:,.0f} tok/s | "
                    f"{elapsed:.0f}s"
                )

            if total_tokens >= args.max_tokens:
                print(f"\n  Reached max-tokens limit ({args.max_tokens:,})")
                break

    elapsed = time.time() - start_time

    # Save index (ensure (N, 2) shape even when empty)
    if index_entries:
        index_array = np.array(index_entries, dtype=np.int64)
    else:
        index_array = np.empty((0, 2), dtype=np.int64)
    np.save(index_path, index_array)

    # Save metadata
    metadata = {
        "total_tokens": total_tokens,
        "total_samples": total_samples,
        "vocab_size": tokenizer.vocab_size,
        "tokenizer_path": str(tokenizer_path),
        "config_file": args.config,
        "hf_dataset": hf_dataset,
        "languages": languages,
        "data_config": data_config,
        "max_tokens_limit": args.max_tokens,
        "elapsed_seconds": round(elapsed, 1),
        "created": datetime.now(timezone.utc).isoformat(),
        "mixer_stats": mixer.stats,
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Print summary
    gb_written = (total_tokens * 2) / (1024**3)
    tps = total_tokens / elapsed if elapsed > 0 else 0
    print("\nPre-tokenization complete:")
    print(f"  Samples:  {total_samples:,}")
    print(f"  Tokens:   {total_tokens:,}")
    print(f"  Size:     {gb_written:.2f} GB")
    print(f"  Time:     {elapsed:.0f}s ({tps:,.0f} tok/s)")
    print(f"  Output:   {output_dir}")

    # DataMixer stats
    mixer.print_stats()


if __name__ == "__main__":
    main()
