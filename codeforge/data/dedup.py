"""Near-deduplication using MinHash + LSH.

Removes near-duplicate code samples from training data to improve
model quality and reduce memorization.
"""

import hashlib
import struct
from typing import Iterator, Optional

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
        return sum(a == b for a, b in zip(sig1, sig2)) / self.num_perm


class LSHIndex:
    """Locality-Sensitive Hashing index for fast approximate nearest neighbor.

    Divides MinHash signatures into bands, hashes each band.
    Documents that share at least one band hash are candidate duplicates.
    """

    def __init__(self, num_bands: int = 16, rows_per_band: int = 8):
        self.num_bands = num_bands
        self.rows_per_band = rows_per_band
        # Each band maps: band_hash -> set of document IDs
        self.buckets: list[dict[int, list[int]]] = [
            {} for _ in range(num_bands)
        ]
        self.num_docs = 0

    def insert(self, doc_id: int, signature: tuple[int, ...]) -> None:
        """Insert a document signature into the index."""
        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            band = signature[start : start + self.rows_per_band]
            band_hash = hash(band)
            if band_hash not in self.buckets[band_idx]:
                self.buckets[band_idx][band_hash] = []
            self.buckets[band_idx][band_hash].append(doc_id)
        self.num_docs += 1

    def query(self, signature: tuple[int, ...]) -> set[int]:
        """Find candidate duplicates for a signature."""
        candidates = set()
        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            band = signature[start : start + self.rows_per_band]
            band_hash = hash(band)
            bucket = self.buckets[band_idx].get(band_hash, [])
            candidates.update(bucket)
        return candidates


class Deduplicator:
    """Full near-deduplication pipeline using MinHash + LSH.

    Usage:
        dedup = Deduplicator(threshold=0.8)
        for text in corpus:
            if dedup.is_unique(text):
                # Keep this sample
                dedup.add(text)
    """

    def __init__(
        self,
        threshold: float = 0.8,
        num_perm: int = 128,
        ngram_size: int = 5,
        num_bands: int = 16,
        max_entries: int = 1_000_000,
    ):
        self.threshold = threshold
        self.num_bands = num_bands
        self.hasher = MinHasher(num_perm=num_perm, ngram_size=ngram_size)
        self.rows_per_band = num_perm // num_bands
        self.index = LSHIndex(num_bands=num_bands, rows_per_band=self.rows_per_band)
        self.signatures: dict[int, tuple[int, ...]] = {}
        self.next_id = 0
        self.max_entries = max_entries

    def _evict_old_entries(self) -> None:
        """Evict oldest 50% of entries when capacity is exceeded."""
        keep_from = self.next_id // 2
        old_count = len(self.signatures)
        self.signatures = {k: v for k, v in self.signatures.items() if k >= keep_from}
        self.index = LSHIndex(
            num_bands=self.num_bands, rows_per_band=self.rows_per_band
        )
        for doc_id, sig in self.signatures.items():
            self.index.insert(doc_id, sig)
        print(f"  [Dedup] Evicted {old_count - len(self.signatures):,} entries, "
              f"kept {len(self.signatures):,}")

    def add(self, text: str) -> int:
        """Add a document and return its ID."""
        if len(self.signatures) >= self.max_entries:
            self._evict_old_entries()

        sig = self.hasher.signature(text)
        doc_id = self.next_id
        self.next_id += 1
        self.signatures[doc_id] = sig
        self.index.insert(doc_id, sig)
        return doc_id

    def is_unique(self, text: str) -> bool:
        """Check if text is sufficiently different from all indexed documents."""
        sig = self.hasher.signature(text)
        candidates = self.index.query(sig)

        for cand_id in candidates:
            cand_sig = self.signatures[cand_id]
            similarity = self.hasher.jaccard_estimate(sig, cand_sig)
            if similarity >= self.threshold:
                return False

        return True

    def deduplicate_stream(self, texts: Iterator[str]) -> Iterator[str]:
        """Filter an iterator of texts, yielding only unique ones."""
        for text in texts:
            if self.is_unique(text):
                self.add(text)
                yield text
