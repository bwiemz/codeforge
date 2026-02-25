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
import random
import signal
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
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from a previous incomplete run (reads existing metadata/index)",
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

    # -- Helpers for saving progress ------------------------------------------

    def save_index_and_metadata(
        index_entries: list[tuple[int, int]],
        total_tokens: int,
        total_samples: int,
        elapsed: float,
        *,
        complete: bool = False,
    ) -> None:
        """Write index.npy + metadata.json (safe to call at any time)."""
        if index_entries:
            index_array = np.array(index_entries, dtype=np.int64)
        else:
            index_array = np.empty((0, 2), dtype=np.int64)
        np.save(index_path, index_array)

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
            "complete": complete,
            "mixer_stats": {
                k: mixer.stats[k] + stats_offset.get(k, 0)
                for k in mixer.stats
            },
        }
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # -- Graceful shutdown on Ctrl+C / SIGTERM --------------------------------

    shutdown_requested = False

    def _signal_handler(signum: int, frame: object) -> None:
        nonlocal shutdown_requested
        if shutdown_requested:
            print("\n  Forced exit — saving may be incomplete")
            sys.exit(1)
        shutdown_requested = True
        print("\n  Shutdown requested — finishing current sample then saving...")

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # -- Resume handling -------------------------------------------------------

    total_tokens = 0
    total_samples = 0
    index_entries: list[tuple[int, int]] = []
    elapsed_before = 0.0
    file_mode = "wb"
    stats_offset: dict[str, int] = {}

    if args.resume:
        if not meta_path.exists():
            print("Error: No metadata.json found — nothing to resume from")
            sys.exit(1)
        prev_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if prev_meta.get("complete", False):
            print("Previous run already complete — nothing to resume.")
            sys.exit(0)

        resume_samples = prev_meta["total_samples"]
        resume_tokens = prev_meta["total_tokens"]
        elapsed_before = prev_meta.get("elapsed_seconds", 0.0)
        prev_index = np.load(index_path)
        index_entries = [tuple(row) for row in prev_index.tolist()]
        mixer_stats = prev_meta.get("mixer_stats", {})
        total_seen = mixer_stats.get("total_seen", resume_samples)

        print("\nResuming from checkpoint:")
        print(f"  Existing: {resume_samples:,} samples, {resume_tokens:,} tokens")
        print(f"  Total raw samples seen: {total_seen:,}")

        # Step 1: Replay language selection RNG to count per-language raw samples
        # This is pure math — no data download, runs in seconds
        print("  Replaying language selection to estimate per-language counts...")
        replay_rng = random.Random(data_config.get("seed", 42))
        lang_list = list(streams.keys())
        lang_weights = [
            (language_weights or {}).get(lang, 0.01) for lang in lang_list
        ]
        total_w = sum(lang_weights)
        lang_weights = [w / total_w for w in lang_weights]

        per_lang_raw: dict[str, int] = {lang: 0 for lang in lang_list}
        for _ in range(total_seen):
            lang = replay_rng.choices(lang_list, weights=lang_weights, k=1)[0]
            per_lang_raw[lang] += 1

        print("  Estimated per-language raw counts:")
        for lang, count in per_lang_raw.items():
            print(f"    {lang}: {count:,}")

        # Step 2: Skip raw samples in each language stream (fast — no quality/dedup)
        print("  Advancing language streams...")
        ff_start = time.time()
        for lang, count in per_lang_raw.items():
            if count == 0 or lang not in streams:
                continue
            skipped = 0
            stream_iter = streams[lang]
            for _ in stream_iter:
                skipped += 1
                if skipped >= count:
                    break
            print(f"    {lang}: skipped {skipped:,} / {count:,}")
        ff_elapsed = time.time() - ff_start
        print(f"  Stream advance done in {ff_elapsed:.0f}s")

        # Step 3: Create fresh mixer (dedup state lost — was only 0.006%, negligible)
        # Re-create mixer to get clean state for continued processing
        mixer = DataMixer(
            language_weights=language_weights,
            quality_threshold=data_config.get("quality_threshold", 0.3),
            enable_dedup=data_config.get("enable_dedup", True),
            dedup_threshold=data_config.get("dedup_threshold", 0.8),
            enable_decontamination=data_config.get(
                "enable_decontamination", True
            ),
            fim_rate=data_config.get("fim_rate", 0.5),
            seed=data_config.get("seed", 42),
        )

        # Preserve cumulative stats so metadata reflects full history
        stats_offset = mixer_stats

        total_tokens = resume_tokens
        total_samples = resume_samples
        file_mode = "ab"

    mixed = mixer.mix_streams(streams)

    # -- Stream, filter, tokenize, write --------------------------------------

    print("\nPre-tokenizing...")
    start_time = time.time()
    last_checkpoint = time.time()
    checkpoint_interval = 300  # Save index every 5 minutes

    try:
        with open(tokens_path, file_mode) as f:
            for code in mixed:
                if shutdown_requested:
                    print("  Graceful shutdown — saving progress...")
                    break

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

                # Periodic index checkpoint
                now = time.time()
                if now - last_checkpoint >= checkpoint_interval:
                    save_index_and_metadata(
                        index_entries, total_tokens, total_samples,
                        elapsed_before + (now - start_time), complete=False,
                    )
                    last_checkpoint = now
                    print(f"  [Checkpoint] Saved index ({total_samples:,} samples)")

                if total_tokens >= args.max_tokens:
                    print(f"\n  Reached max-tokens limit ({args.max_tokens:,})")
                    break

    except Exception as e:
        elapsed = elapsed_before + (time.time() - start_time)
        print(f"\n  ERROR: {type(e).__name__}: {e}")
        print(f"  Saving partial progress ({total_samples:,} samples, "
              f"{total_tokens:,} tokens)...")
        save_index_and_metadata(
            index_entries, total_tokens, total_samples, elapsed, complete=False,
        )
        mixer.print_stats()
        raise

    elapsed = elapsed_before + (time.time() - start_time)
    complete = not shutdown_requested
    save_index_and_metadata(
        index_entries, total_tokens, total_samples, elapsed, complete=complete,
    )

    # Print summary
    gb_written = (total_tokens * 2) / (1024**3)
    tps = total_tokens / elapsed if elapsed > 0 else 0
    status = "complete" if complete else "partial (interrupted)"
    print(f"\nPre-tokenization {status}:")
    print(f"  Samples:  {total_samples:,}")
    print(f"  Tokens:   {total_tokens:,}")
    print(f"  Size:     {gb_written:.2f} GB")
    print(f"  Time:     {elapsed:.0f}s ({tps:,.0f} tok/s)")
    print(f"  Output:   {output_dir}")

    mixer.print_stats()


if __name__ == "__main__":
    main()
