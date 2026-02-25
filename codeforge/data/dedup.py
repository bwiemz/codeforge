"""Near-deduplication using MinHash + LSH.

Removes near-duplicate code samples from training data to improve
model quality and reduce memorization.
"""

from __future__ import annotations

import array
import hashlib
import struct
from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

# Large prime for hash computation
_MERSENNE_PRIME = (1 << 61) - 1
_MAX_HASH = (1 << 32) - 1


def _ngrams(text: str, n: int = 5) -> list[str]:
    """Extract character n-grams from text."""
    tokens = text.split()
    if len(tokens) < n:
        return [" ".join(tokens)] if tokens else []
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _sha1_hash(text: str) -> int:
    """Compute a 32-bit hash from text using SHA-1."""
    return struct.unpack("<I", hashlib.sha1(text.encode("utf-8")).digest()[:4])[0]


class MinHasher:
    """MinHash signature computation for near-deduplication.

    Uses random hash functions to compute a fixed-size signature
    that approximates Jaccard similarity.
    """

    def __init__(self, num_perm: int = 128, ngram_size: int = 5, seed: int = 42):
        self.num_perm = num_perm
        self.ngram_size = ngram_size

        # Generate random hash function parameters: h(x) = (ax + b) % p
        import random
        rng = random.Random(seed)
        self.a = [rng.randint(1, _MERSENNE_PRIME - 1) for _ in range(num_perm)]
        self.b = [rng.randint(0, _MERSENNE_PRIME - 1) for _ in range(num_perm)]

    def signature(self, text: str) -> tuple[int, ...]:
        """Compute MinHash signature for a text."""
        grams = _ngrams(text, self.ngram_size)
        if not grams:
            return tuple([_MAX_HASH] * self.num_perm)

        # Hash each n-gram
        hashes = [_sha1_hash(g) for g in grams]

        # For each permutation, compute the minimum hash
        sig = []
        for i in range(self.num_perm):
            min_hash = _MAX_HASH
            for h in hashes:
                val = ((self.a[i] * h + self.b[i]) % _MERSENNE_PRIME) & _MAX_HASH
                if val < min_hash:
                    min_hash = val
            sig.append(min_hash)

        return tuple(sig)

    def jaccard_estimate(self, sig1: tuple[int, ...], sig2: tuple[int, ...]) -> float:
        """Estimate Jaccard similarity from two MinHash signatures."""
        return sum(a == b for a, b in zip(sig1, sig2, strict=True)) / self.num_perm


class LSHIndex:
    """Locality-Sensitive Hashing index for fast approximate nearest neighbor.

    Divides MinHash signatures into bands, hashes each band.
    Documents that share at least one band hash are candidate duplicates.
    """

    def __init__(self, num_bands: int = 16, rows_per_band: int = 8):
        self.num_bands = num_bands
        self.rows_per_band = rows_per_band
        self.buckets: list[dict[int, set[int]]] = [
            {} for _ in range(num_bands)
        ]
        self.num_docs = 0

    def insert(self, doc_id: int, signature: tuple[int, ...]) -> None:
        """Insert a document signature into the index."""
        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            band = signature[start : start + self.rows_per_band]
            band_hash = hash(band)
            bucket = self.buckets[band_idx].get(band_hash)
            if bucket is None:
                self.buckets[band_idx][band_hash] = {doc_id}
            else:
                bucket.add(doc_id)
        self.num_docs += 1

    def remove(self, doc_id: int, signature: tuple[int, ...]) -> None:
        """Remove a document from the index."""
        found = False
        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            band = signature[start : start + self.rows_per_band]
            band_hash = hash(band)
            bucket = self.buckets[band_idx].get(band_hash)
            if bucket is not None and doc_id in bucket:
                bucket.discard(doc_id)
                found = True
                if not bucket:
                    del self.buckets[band_idx][band_hash]
        if found:
            self.num_docs -= 1

    def query(self, signature: tuple[int, ...]) -> set[int]:
        """Find candidate duplicates for a signature."""
        candidates: set[int] = set()
        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            band = signature[start : start + self.rows_per_band]
            band_hash = hash(band)
            bucket = self.buckets[band_idx].get(band_hash)
            if bucket is not None:
                candidates.update(bucket)
        return candidates


class Deduplicator:
    """Full near-deduplication pipeline using MinHash + LSH.

    Uses a fixed-size ring buffer so memory stays bounded regardless of
    how many samples are processed.  Signatures are stored as compact
    ``array.array('I')`` (4 bytes per uint32) instead of Python tuples
    to cut per-entry memory by ~4x.

    Usage:
        dedup = Deduplicator(threshold=0.8)
        for text in corpus:
            if dedup.is_unique(text):
                dedup.add(text)
    """

    def __init__(
        self,
        threshold: float = 0.8,
        num_perm: int = 128,
        ngram_size: int = 5,
        num_bands: int = 16,
        max_entries: int = 500_000,
    ):
        self.threshold = threshold
        self.num_bands = num_bands
        self.num_perm = num_perm
        self.hasher = MinHasher(num_perm=num_perm, ngram_size=ngram_size)
        self.rows_per_band = num_perm // num_bands
        self.index = LSHIndex(num_bands=num_bands, rows_per_band=self.rows_per_band)
        self.max_entries = max_entries

        # Ring buffer: deque of (doc_id, signature) for O(1) eviction of oldest
        self._ring: deque[tuple[int, array.array[int]]] = deque()
        # Fast lookup: doc_id -> compact signature
        self._sigs: dict[int, array.array[int]] = {}
        self._next_id = 0

    def _compact_sig(self, sig: tuple[int, ...]) -> array.array[int]:
        """Store signature as a compact unsigned-int array (~4 bytes/element)."""
        a = array.array("I")
        a.fromlist(list(sig))
        return a

    def _evict_oldest(self) -> None:
        """Evict the single oldest entry (O(1) amortised)."""
        doc_id, sig = self._ring.popleft()
        # May already have been removed (shouldn't happen, but be safe)
        if doc_id in self._sigs:
            del self._sigs[doc_id]
            self.index.remove(doc_id, tuple(sig))

    def add(self, text: str) -> int:
        """Add a document and return its ID."""
        while len(self._sigs) >= self.max_entries and self._ring:
            self._evict_oldest()

        sig_tuple = self.hasher.signature(text)
        sig_compact = self._compact_sig(sig_tuple)
        doc_id = self._next_id
        self._next_id += 1
        self._sigs[doc_id] = sig_compact
        self._ring.append((doc_id, sig_compact))
        self.index.insert(doc_id, sig_tuple)
        return doc_id

    def is_unique(self, text: str) -> bool:
        """Check if text is sufficiently different from all indexed documents."""
        sig = self.hasher.signature(text)
        candidates = self.index.query(sig)

        for cand_id in candidates:
            cand_sig = self._sigs.get(cand_id)
            if cand_sig is None:
                continue  # Evicted between query and check
            similarity = self.hasher.jaccard_estimate(sig, tuple(cand_sig))
            if similarity >= self.threshold:
                return False

        return True

    @property
    def num_entries(self) -> int:
        return len(self._sigs)

    def deduplicate_stream(self, texts: Iterator[str]) -> Iterator[str]:
        """Filter an iterator of texts, yielding only unique ones."""
        for text in texts:
            if self.is_unique(text):
                self.add(text)
                yield text
