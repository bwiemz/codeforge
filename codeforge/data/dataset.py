"""Dataset classes for code training data."""

import random
from pathlib import Path
from typing import Iterator, Optional

import torch
from torch.utils.data import IterableDataset, DataLoader

from ..tokenizer.tokenizer import CodeForgeTokenizer
from .preprocessing import preprocess_code, apply_fim_transform
from .mix import DataMixer


class CodeDataset(IterableDataset):
    """Streaming dataset that tokenizes and packs code samples.

    Supports loading from:
    - Local text/code files
    - HuggingFace datasets (streamed)
    """

    def __init__(
        self,
        tokenizer: CodeForgeTokenizer,
        max_seq_len: int = 2048,
        fim_rate: float = 0.5,
        sources: Optional[list[str]] = None,
        local_files: Optional[list[str | Path]] = None,
        languages: Optional[list[str]] = None,
        seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.fim_rate = fim_rate
        self.sources = sources or []
        self.local_files = [Path(f) for f in (local_files or [])]
        self.languages = set(languages) if languages else None
        self.seed = seed

    def _iterate_local_files(self) -> Iterator[str]:
        """Yield code content from local files."""
        for path in self.local_files:
            if path.is_file():
                try:
                    yield path.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue
            elif path.is_dir():
                for file in sorted(path.rglob("*")):
                    if file.is_file() and file.suffix in {
                        ".py", ".js", ".ts", ".java", ".cpp", ".c", ".h",
                        ".go", ".rs", ".rb", ".php", ".cs", ".kt", ".swift",
                    }:
                        try:
                            yield file.read_text(encoding="utf-8", errors="ignore")
                        except Exception:
                            continue

    def _iterate_hf_datasets(self) -> Iterator[str]:
        """Yield code content from HuggingFace datasets (streamed)."""
        try:
            from datasets import load_dataset
        except ImportError:
            return

        for source in self.sources:
            try:
                ds = load_dataset(source, split="train", streaming=True)
                for sample in ds:
                    # Handle common dataset column names
                    content = sample.get("content") or sample.get("code") or sample.get("text", "")
                    if content:
                        yield content
            except Exception as e:
                print(f"Warning: Failed to load dataset '{source}': {e}")
                continue

    def _raw_samples(self) -> Iterator[str]:
        """Yield raw code strings from all configured sources."""
        yield from self._iterate_local_files()
        yield from self._iterate_hf_datasets()

    def _pack_tokens(self, token_stream: Iterator[list[int]]) -> Iterator[dict[str, torch.Tensor]]:
        """Pack tokenized samples into fixed-length sequences.

        Concatenates multiple samples (separated by EOS) into sequences
        of exactly max_seq_len for efficient training.
        """
        buffer = []
        for ids in token_stream:
            buffer.extend(ids)
            buffer.append(self.tokenizer.EOS_ID)

            while len(buffer) >= self.max_seq_len + 1:
                chunk = buffer[: self.max_seq_len + 1]
                buffer = buffer[self.max_seq_len + 1 :]

                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                targets = torch.tensor(chunk[1:], dtype=torch.long)
                yield {"input_ids": input_ids, "targets": targets}

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        rng = random.Random(self.seed)

        def token_stream():
            for code in self._raw_samples():
                processed = preprocess_code(code)
                if processed is None:
                    continue

                # Apply FIM transform with configured rate
                if self.fim_rate > 0 and rng.random() < self.fim_rate:
                    processed = apply_fim_transform(processed, fim_rate=1.0)

                ids = self.tokenizer.encode(processed, add_special_tokens=False)
                if len(ids) > 0:
                    yield ids

        yield from self._pack_tokens(token_stream())


class SyntheticCodeDataset(IterableDataset):
    """Generates synthetic data for smoke testing the training loop."""

    def __init__(self, vocab_size: int, seq_len: int = 2048, num_samples: int = 1000):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        for _ in range(self.num_samples):
            ids = torch.randint(0, self.vocab_size, (self.seq_len + 1,))
            yield {
                "input_ids": ids[:-1],
                "targets": ids[1:],
            }


class MixedCodeDataset(IterableDataset):
    """Production dataset that integrates DataMixer for quality-filtered,
    deduplicated, language-balanced streaming from HuggingFace datasets.

    Bridges: HF streaming -> DataMixer (quality + dedup + decontam + FIM) -> token packing.
    """

    def __init__(
        self,
        tokenizer: CodeForgeTokenizer,
        max_seq_len: int = 2048,
        hf_dataset: str = "bigcode/starcoderdata",
        languages: Optional[list[str]] = None,
        language_weights: Optional[dict[str, float]] = None,
        quality_threshold: float = 0.3,
        enable_dedup: bool = True,
        dedup_threshold: float = 0.8,
        enable_decontamination: bool = True,
        fim_rate: float = 0.5,
        seed: int = 42,
        hf_dataset_kwargs: Optional[dict] = None,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.hf_dataset = hf_dataset
        self.languages = languages or [
            "python", "javascript", "typescript", "java",
            "cpp", "c", "go", "rust",
        ]
        self.hf_dataset_kwargs = hf_dataset_kwargs or {}

        self.mixer = DataMixer(
            language_weights=language_weights,
            quality_threshold=quality_threshold,
            enable_dedup=enable_dedup,
            dedup_threshold=dedup_threshold,
            enable_decontamination=enable_decontamination,
            fim_rate=fim_rate,
            seed=seed,
        )

    def _create_language_streams(self) -> dict[str, Iterator[str]]:
        """Create per-language iterators from the HF dataset."""
        from datasets import load_dataset

        streams = {}

        # Try per-language loading via data_dir (starcoderdata format)
        for lang in self.languages:
            try:
                ds = load_dataset(
                    self.hf_dataset,
                    data_dir=lang,
                    split="train",
                    streaming=True,
                    **self.hf_dataset_kwargs,
                )

                def make_iterator(dataset):
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

        # Fallback: if no per-language streams worked, load full and filter
        if not streams:
            print("  Falling back to full dataset with lang column filtering...")
            ds = load_dataset(
                self.hf_dataset,
                split="train",
                streaming=True,
                **self.hf_dataset_kwargs,
            )
            languages_lower = {l.lower() for l in self.languages}

            # Partition the single stream into per-language buckets
            # by iterating once and dispatching
            for lang in self.languages:

                def make_filtered_iter(dataset, target_lang):
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
                # Only the first language can use this approach with a single stream
                # For multiple languages, we'd need separate loads
                ds = load_dataset(
                    self.hf_dataset,
                    split="train",
                    streaming=True,
                    **self.hf_dataset_kwargs,
                )

        return streams

    def _pack_tokens(
        self, token_stream: Iterator[list[int]]
    ) -> Iterator[dict[str, torch.Tensor]]:
        """Pack tokenized samples into fixed-length sequences."""
        buffer = []
        for ids in token_stream:
            buffer.extend(ids)
            buffer.append(self.tokenizer.EOS_ID)

            while len(buffer) >= self.max_seq_len + 1:
                chunk = buffer[: self.max_seq_len + 1]
                buffer = buffer[self.max_seq_len + 1 :]

                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                targets = torch.tensor(chunk[1:], dtype=torch.long)
                yield {"input_ids": input_ids, "targets": targets}

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        streams = self._create_language_streams()

        if not streams:
            raise RuntimeError(
                "No language streams could be created. "
                "Check dataset name and language names."
            )

        print(f"  Created streams for: {', '.join(streams.keys())}")

        # Mix through DataMixer (quality + dedup + decontam + FIM)
        mixed = self.mixer.mix_streams(streams)

        def token_stream():
            for code in mixed:
                ids = self.tokenizer.encode(code, add_special_tokens=False)
                if len(ids) > 0:
                    yield ids

        yield from self._pack_tokens(token_stream())

    def print_stats(self):
        """Print DataMixer filtering statistics."""
        self.mixer.print_stats()


class EvalCodeDataset(IterableDataset):
    """Evaluation dataset that draws a fixed number of held-out samples.

    Uses a different seed and skips the first `skip_samples` items in each
    language stream so that eval data does not overlap with training data.
    The dataset is finite (produces exactly `num_eval_samples` packed sequences).
    """

    def __init__(
        self,
        tokenizer: CodeForgeTokenizer,
        max_seq_len: int = 2048,
        hf_dataset: str = "bigcode/starcoderdata",
        languages: Optional[list[str]] = None,
        quality_threshold: float = 0.3,
        fim_rate: float = 0.0,
        num_eval_samples: int = 200,
        skip_samples: int = 50_000,
        seed: int = 99999,
        hf_dataset_kwargs: Optional[dict] = None,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.hf_dataset = hf_dataset
        self.languages = languages or [
            "python", "javascript", "typescript", "java",
            "cpp", "c", "go", "rust",
        ]
        self.quality_threshold = quality_threshold
        self.fim_rate = fim_rate
        self.num_eval_samples = num_eval_samples
        self.skip_samples = skip_samples
        self.seed = seed
        self.hf_dataset_kwargs = hf_dataset_kwargs or {}

    def _create_eval_streams(self) -> dict[str, Iterator[str]]:
        """Create per-language iterators, skipping the first N samples."""
        from datasets import load_dataset

        streams = {}
        for lang in self.languages:
            try:
                ds = load_dataset(
                    self.hf_dataset,
                    data_dir=lang,
                    split="train",
                    streaming=True,
                    **self.hf_dataset_kwargs,
                )

                def make_iterator(dataset, skip=self.skip_samples):
                    for i, sample in enumerate(dataset):
                        if i < skip:
                            continue
                        content = (
                            sample.get("content")
                            or sample.get("code")
                            or sample.get("text", "")
                        )
                        if content and len(content.strip()) > 50:
                            yield content

                streams[lang] = make_iterator(ds)
            except Exception:
                continue
        return streams

    def _pack_tokens(
        self, token_stream: Iterator[list[int]]
    ) -> Iterator[dict[str, torch.Tensor]]:
        """Pack tokenized samples into fixed-length sequences."""
        buffer = []
        for ids in token_stream:
            buffer.extend(ids)
            buffer.append(self.tokenizer.EOS_ID)

            while len(buffer) >= self.max_seq_len + 1:
                chunk = buffer[: self.max_seq_len + 1]
                buffer = buffer[self.max_seq_len + 1 :]

                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                targets = torch.tensor(chunk[1:], dtype=torch.long)
                yield {"input_ids": input_ids, "targets": targets}

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        from .preprocessing import preprocess_code
        from .quality import compute_quality_score

        streams = self._create_eval_streams()
        if not streams:
            raise RuntimeError("No eval language streams could be created.")

        rng = random.Random(self.seed)

        # Simple round-robin sampling across languages for eval
        active_iters = {lang: iter(s) for lang, s in streams.items()}
        exhausted = set()
        lang_list = list(active_iters.keys())

        def raw_samples():
            idx = 0
            while len(exhausted) < len(lang_list):
                lang = lang_list[idx % len(lang_list)]
                idx += 1
                if lang in exhausted:
                    continue
                try:
                    code = next(active_iters[lang])
                    code = preprocess_code(code)
                    if code is None:
                        continue
                    quality = compute_quality_score(code, lang)
                    if quality < self.quality_threshold:
                        continue
                    yield code
                except StopIteration:
                    exhausted.add(lang)

        def token_stream():
            for code in raw_samples():
                ids = self.tokenizer.encode(code, add_special_tokens=False)
                if len(ids) > 0:
                    yield ids

        count = 0
        for sample in self._pack_tokens(token_stream()):
            yield sample
            count += 1
            if count >= self.num_eval_samples:
                return
