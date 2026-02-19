"""Train the TCFP tokenizer on bigcode/starcoderdata.

Streams code from HuggingFace with weighted language sampling to match
the pretraining data distribution, then trains a 49,152-token byte-level
BPE tokenizer aligned to FP8 tensor core tile boundaries.

Usage:
    python -m scripts.train_tokenizer
    python -m scripts.train_tokenizer --num-samples 1000000
    python -m scripts.train_tokenizer --from-files data/corpus/*.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from codeforge.tokenizer.config import TokenizerConfig
from codeforge.tokenizer.training import train_tokenizer, train_tokenizer_from_iterator
from codeforge.tokenizer.validation import validate_tokenizer
from codeforge.tokenizer.wrapper import TCFPTokenizer

# Language weights matching pretrain_150m.yaml
LANGUAGE_WEIGHTS: dict[str, float] = {
    "python": 0.35,
    "javascript": 0.12,
    "typescript": 0.12,
    "java": 0.10,
    "cpp": 0.08,
    "c": 0.05,
    "go": 0.07,
    "rust": 0.07,
}

# Validation text samples per language
VALIDATION_TEXTS: dict[str, str] = {
    "python": (
        "def fibonacci(n: int) -> list[int]:\n"
        "    result = [0, 1]\n"
        "    for i in range(2, n):\n"
        "        result.append(result[i-1] + result[i-2])\n"
        "    return result\n"
    ),
    "javascript": (
        "async function fetchData(url) {\n"
        "    const response = await fetch(url);\n"
        "    if (!response.ok) throw new Error(`HTTP ${response.status}`);\n"
        "    return response.json();\n"
        "}\n"
    ),
    "typescript": (
        "interface Config {\n"
        "    readonly host: string;\n"
        "    port: number;\n"
        "    debug?: boolean;\n"
        "}\n"
        "const config: Config = { host: 'localhost', port: 8080 };\n"
    ),
    "java": (
        "public class HashMap<K, V> {\n"
        "    private Entry<K, V>[] table;\n"
        "    public V get(Object key) {\n"
        "        int hash = key.hashCode() & 0x7fffffff;\n"
        "        return table[hash % table.length].value;\n"
        "    }\n"
        "}\n"
    ),
    "rust": (
        "fn parse_config(path: &Path) -> Result<Config, Box<dyn Error>> {\n"
        "    let contents = fs::read_to_string(path)?;\n"
        "    let config: Config = serde_json::from_str(&contents)?;\n"
        "    Ok(config)\n"
        "}\n"
    ),
    "go": (
        "func (s *Server) handleRequest(w http.ResponseWriter, r *http.Request) {\n"
        "    ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)\n"
        "    defer cancel()\n"
        "    result, err := s.db.QueryContext(ctx, query)\n"
        "}\n"
    ),
    "cpp": (
        "template<typename T>\n"
        "class Vector {\n"
        "    T* data_;\n"
        "    size_t size_, capacity_;\n"
        "public:\n"
        "    void push_back(const T& value) {\n"
        "        if (size_ == capacity_) reserve(capacity_ * 2);\n"
        "        data_[size_++] = value;\n"
        "    }\n"
        "};\n"
    ),
}


def weighted_stream_iterator(
    dataset_name: str,
    language_weights: dict[str, float],
    num_samples: int,
) -> tuple[int, "Iterator[str]"]:  # noqa: F821
    """Stream weighted code samples from HuggingFace.

    Opens one streaming dataset per language (via data_dir for starcoderdata),
    then yields samples in proportion to the specified weights.

    Returns (total_count, iterator) where total_count is for progress tracking.
    """
    from datasets import load_dataset

    # Compute per-language sample counts
    total_weight = sum(language_weights.values())
    lang_counts: dict[str, int] = {}
    for lang, weight in language_weights.items():
        lang_counts[lang] = int(num_samples * weight / total_weight)

    # Adjust rounding to hit exact total
    assigned = sum(lang_counts.values())
    if assigned < num_samples:
        # Give remainder to highest-weight language
        top_lang = max(language_weights, key=language_weights.get)  # type: ignore[arg-type]
        lang_counts[top_lang] += num_samples - assigned

    print(f"Target samples per language:")
    for lang, count in sorted(lang_counts.items(), key=lambda x: -x[1]):
        print(f"  {lang:>12}: {count:>7,} ({language_weights[lang]*100:.0f}%)")
    print(f"  {'TOTAL':>12}: {num_samples:>7,}")

    # Open streaming datasets
    streams: dict[str, "Iterator"] = {}  # noqa: F821
    for lang in language_weights:
        try:
            ds = load_dataset(dataset_name, data_dir=lang, split="train", streaming=True)
            streams[lang] = iter(ds)
            print(f"  Stream opened: {lang}")
        except Exception as e:
            print(f"  WARNING: Could not open stream for '{lang}': {e}")

    if not streams:
        raise RuntimeError(f"No language streams could be opened from {dataset_name}")

    def _iterator():
        counts: dict[str, int] = {lang: 0 for lang in streams}
        skips: dict[str, int] = {lang: 0 for lang in streams}
        max_skips_per_lang = 10_000  # guard against spinning on empty streams
        total = 0
        exhausted: set[str] = set()
        t0 = time.time()

        while total < num_samples and len(exhausted) < len(streams):
            for lang, it in streams.items():
                if lang in exhausted:
                    continue
                if counts[lang] >= lang_counts.get(lang, 0):
                    exhausted.add(lang)
                    continue

                try:
                    sample = next(it)
                except StopIteration:
                    exhausted.add(lang)
                    print(f"  Stream '{lang}' exhausted at {counts[lang]:,} samples")
                    continue

                text = sample.get("content") or sample.get("code") or sample.get("text", "")
                if not text or len(text.strip()) < 50:
                    skips[lang] += 1
                    if skips[lang] >= max_skips_per_lang:
                        exhausted.add(lang)
                        print(f"  Stream '{lang}' skipped too many short samples, marking exhausted")
                    continue

                skips[lang] = 0  # reset on success
                yield text
                counts[lang] += 1
                total += 1

                if total % 25_000 == 0:
                    elapsed = time.time() - t0
                    rate = total / elapsed
                    eta = (num_samples - total) / rate if rate > 0 else 0
                    print(
                        f"  {total:>7,}/{num_samples:,} samples "
                        f"({total/num_samples*100:.1f}%) "
                        f"[{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining]"
                    )

                if total >= num_samples:
                    break

        elapsed = time.time() - t0
        print(f"\nStreaming complete: {total:,} samples in {elapsed:.1f}s")
        for lang in sorted(counts):
            target = lang_counts.get(lang, 0)
            actual = counts[lang]
            pct = actual / target * 100 if target > 0 else 0
            print(f"  {lang:>12}: {actual:>7,} / {target:>7,} ({pct:.0f}%)")

    return num_samples, _iterator()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the TCFP tokenizer (49,152 tokens, FP8-aligned)"
    )
    parser.add_argument(
        "--from-files",
        nargs="+",
        default=None,
        help="Train from local code files/directories instead of HuggingFace",
    )
    parser.add_argument(
        "--dataset",
        default="bigcode/starcoderdata",
        help="HuggingFace dataset (default: bigcode/starcoderdata)",
    )
    parser.add_argument(
        "--output", "-o",
        default="tokenizer",
        help="Output directory (default: tokenizer/)",
    )
    parser.add_argument(
        "--num-samples", "-n",
        type=int,
        default=500_000,
        help="Total samples for tokenizer training (default: 500,000)",
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=2,
        help="Minimum BPE merge frequency (default: 2)",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip post-training validation",
    )
    args = parser.parse_args()

    config = TokenizerConfig(
        output_dir=Path(args.output),
        tokenizer_filename="codeforge.json",
        min_frequency=args.min_frequency,
    )

    print("=" * 60)
    print("TCFP Tokenizer Training")
    print("=" * 60)
    print(f"  Vocab size:     {config.vocab_size:,} ({config.vocab_size // 128} x 128 tiles)")
    print(f"  BPE merges:     {config.bpe_vocab_size - 256:,} + 256 byte alphabet")
    print(f"  Special tokens: {config.num_special_tokens}")
    print(f"  Output:         {config.output_path}")
    print(f"  Min frequency:  {config.min_frequency}")
    print()

    t_start = time.time()

    if args.from_files:
        # File-based training
        extensions = {".py", ".js", ".ts", ".java", ".cpp", ".c", ".h", ".go", ".rs"}
        files: list[Path] = []
        for path_str in args.from_files:
            path = Path(path_str)
            if path.is_file():
                files.append(path)
            elif path.is_dir():
                for ext in extensions:
                    files.extend(path.rglob(f"*{ext}"))

        if not files:
            print(f"ERROR: No code files found in {args.from_files}")
            sys.exit(1)

        print(f"Training on {len(files):,} local files...")
        tokenizer = train_tokenizer(config, corpus_files=files)
    else:
        # HuggingFace streaming (default)
        print(f"Streaming from: {args.dataset}")
        print(f"Total samples:  {args.num_samples:,}")
        print()

        length, iterator = weighted_stream_iterator(
            args.dataset,
            LANGUAGE_WEIGHTS,
            args.num_samples,
        )

        print(f"\nTraining BPE tokenizer...")
        tokenizer = train_tokenizer_from_iterator(config, iterator, length=length)

    t_train = time.time() - t_start
    print(f"\nTraining complete in {t_train:.1f}s")
    print(f"Saved to: {config.output_path}")

    # Validation
    if not args.skip_validation:
        print("\n" + "=" * 60)
        print("Validation")
        print("=" * 60)

        report = validate_tokenizer(
            tokenizer,
            validation_texts=VALIDATION_TEXTS,
            roundtrip_samples=[
                "    if x == 0:\n        return None\n",
                "const arr = [1, 2, 3].map(x => x * 2);",
                "fn main() { println!(\"Hello, world!\"); }",
                "port = 8080\nhttp://localhost:8080/api/v2",
                "  \t  \n\n  ",  # whitespace edge case
                "x = 3.14159265358979323846",
            ],
        )
        print(report.summary())

        # Quick demo
        print("\n" + "=" * 60)
        print("Quick Demo")
        print("=" * 60)
        wrapper = TCFPTokenizer(config.output_path)
        print(f"Vocab size: {wrapper.vocab_size:,}")
        print(f"Pad ID: {wrapper.pad_id}, BOS: {wrapper.bos_id}, EOS: {wrapper.eos_id}")

        demo = 'def hello(name: str) -> str:\n    return f"Hello, {name}!"'
        ids = wrapper.encode(demo)
        decoded = wrapper.decode(ids)
        print(f"\nInput:   {demo!r}")
        print(f"Tokens:  {len(ids)} IDs")
        print(f"Decoded: {decoded!r}")
        print(f"Match:   {demo == decoded}")

        # Digit splitting check
        digit_demo = "port = 8080"
        digit_ids = wrapper.encode(digit_demo)
        digit_tokens = [wrapper.decode([i]) for i in digit_ids]
        print(f"\nDigit splitting: {digit_demo!r}")
        print(f"Tokens: {digit_tokens}")

    print(f"\nTotal time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
