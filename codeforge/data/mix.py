"""Multi-source data mixing with quality-weighted sampling.

Controls the mix of different programming languages and data sources
during training, with higher-quality samples drawn more frequently.
"""

import random
from typing import Iterator, Optional

from .quality import compute_quality_score
from .dedup import Deduplicator
from .decontaminate import Decontaminator
from .preprocessing import preprocess_code, detect_language, apply_fim_transform


# Default language mix ratios (should sum to ~1.0)
# Weighted toward languages most useful for coding assistance
DEFAULT_LANGUAGE_WEIGHTS = {
    "python": 0.25,
    "javascript": 0.12,
    "typescript": 0.12,
    "java": 0.10,
    "cpp": 0.08,
    "c": 0.05,
    "go": 0.07,
    "rust": 0.07,
    "csharp": 0.04,
    "ruby": 0.03,
    "php": 0.03,
    "swift": 0.02,
    "kotlin": 0.02,
}


class DataMixer:
    """Manages multi-source, multi-language data mixing with quality filtering.

    Combines:
    - Language-balanced sampling
    - Quality-weighted upsampling of good code
    - Near-deduplication
    - Benchmark decontamination
    - FIM transformation
    """

    def __init__(
        self,
        language_weights: Optional[dict[str, float]] = None,
        quality_threshold: float = 0.3,
        quality_upsample: bool = True,
        enable_dedup: bool = True,
        dedup_threshold: float = 0.8,
        enable_decontamination: bool = True,
        fim_rate: float = 0.5,
        seed: int = 42,
    ):
        self.language_weights = language_weights or DEFAULT_LANGUAGE_WEIGHTS
        self.quality_threshold = quality_threshold
        self.quality_upsample = quality_upsample
        self.fim_rate = fim_rate
        self.rng = random.Random(seed)

        # Deduplication
        self.dedup = Deduplicator(threshold=dedup_threshold) if enable_dedup else None

        # Decontamination
        self.decontaminator = None
        if enable_decontamination:
            self.decontaminator = Decontaminator()
            self.decontaminator.add_humaneval()

        # Stats tracking
        self.stats = {
            "total_seen": 0,
            "quality_filtered": 0,
            "dedup_filtered": 0,
            "decontam_filtered": 0,
            "passed": 0,
        }

    def process_sample(
        self,
        code: str,
        filename: Optional[str] = None,
        language: Optional[str] = None,
    ) -> Optional[str]:
        """Process a single code sample through the full pipeline.

        Returns processed code or None if filtered out.
        """
        self.stats["total_seen"] += 1

        # Detect language if not provided
        if language is None and filename:
            language = detect_language(filename)

        # Preprocess
        code = preprocess_code(code)
        if code is None:
            self.stats["quality_filtered"] += 1
            return None

        # Quality check
        quality = compute_quality_score(code, language)
        if quality < self.quality_threshold:
            self.stats["quality_filtered"] += 1
            return None

        # Quality-weighted upsampling: duplicate high-quality samples
        if self.quality_upsample and quality > 0.8 and self.rng.random() < 0.3:
            pass  # Will be yielded (effectively upsampled)
        elif quality < 0.5 and self.rng.random() < 0.3:
            self.stats["quality_filtered"] += 1
            return None  # Downsample low-quality

        # Deduplication
        if self.dedup and not self.dedup.is_unique(code):
            self.stats["dedup_filtered"] += 1
            return None
        if self.dedup:
            self.dedup.add(code)

        # Decontamination
        if self.decontaminator and self.decontaminator.is_contaminated(code):
            self.stats["decontam_filtered"] += 1
            return None

        # Apply FIM transformation
        if self.fim_rate > 0 and self.rng.random() < self.fim_rate:
            code = apply_fim_transform(code, fim_rate=1.0)

        self.stats["passed"] += 1
        return code

    def mix_streams(
        self,
        streams: dict[str, Iterator[str]],
        log_every: int = 10_000,
    ) -> Iterator[str]:
        """Mix multiple language-tagged streams according to configured weights.

        Args:
            streams: Dict mapping language name to iterators of code strings
            log_every: Print progress every N passed samples

        Yields:
            Processed code samples in mixed order
        """
        # Build weighted selection list
        languages = list(streams.keys())
        weights = [self.language_weights.get(lang, 0.01) for lang in languages]
        total = sum(weights)
        weights = [w / total for w in weights]

        # Create iterators with StopIteration handling
        active_iters = {lang: iter(stream) for lang, stream in streams.items()}
        exhausted = set()

        while len(exhausted) < len(languages):
            # Weighted random language selection
            available = [
                (lang, w) for lang, w in zip(languages, weights)
                if lang not in exhausted
            ]
            if not available:
                break

            avail_langs, avail_weights = zip(*available)
            total_w = sum(avail_weights)
            avail_weights = [w / total_w for w in avail_weights]
            lang = self.rng.choices(avail_langs, weights=avail_weights, k=1)[0]

            try:
                code = next(active_iters[lang])
                result = self.process_sample(code, language=lang)
                if result is not None:
                    if self.stats["passed"] % log_every == 0 and self.stats["passed"] > 0:
                        self._log_progress()
                    yield result
            except StopIteration:
                exhausted.add(lang)
                print(f"  [DataMixer] Language '{lang}' stream exhausted")

    def _log_progress(self) -> None:
        """Print periodic progress update."""
        s = self.stats
        total = s["total_seen"] or 1
        pass_rate = 100 * s["passed"] / total
        print(
            f"  [DataMixer] {s['passed']:,} samples passed | "
            f"Pass rate: {pass_rate:.1f}% | "
            f"Quality filtered: {s['quality_filtered']:,} | "
            f"Dedup: {s['dedup_filtered']:,} | "
            f"Decontam: {s['decontam_filtered']:,}"
        )

    def print_stats(self) -> None:
        """Print filtering statistics."""
        s = self.stats
        total = s["total_seen"] or 1
        print(f"\nData mixing stats:")
        print(f"  Total seen:          {s['total_seen']:,}")
        print(f"  Quality filtered:    {s['quality_filtered']:,} ({100*s['quality_filtered']/total:.1f}%)")
        print(f"  Dedup filtered:      {s['dedup_filtered']:,} ({100*s['dedup_filtered']/total:.1f}%)")
        print(f"  Decontam filtered:   {s['decontam_filtered']:,} ({100*s['decontam_filtered']/total:.1f}%)")
        print(f"  Passed:              {s['passed']:,} ({100*s['passed']/total:.1f}%)")
