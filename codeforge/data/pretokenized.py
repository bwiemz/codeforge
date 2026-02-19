"""Pre-tokenized memmap dataset for fast training data loading.

Reads token IDs from a flat uint16 memmap file + sample index, eliminating
HuggingFace streaming, quality filtering, dedup, and tokenization overhead
during training. Pre-tokenization is done once by scripts/pretokenize.py.

Storage format:
  tokens.bin   — flat little-endian uint16 memmap of token IDs
  index.npy    — (N, 2) int64 array of (offset, length) per sample
  metadata.json — config snapshot, stats, tokenizer info
"""

import json
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import IterableDataset


class PreTokenizedDataset(IterableDataset):
    """Dataset that reads pre-tokenized samples from memmap files.

    Replaces MixedCodeDataset when pre-tokenized data is available.
    Samples are shuffled per-epoch (with advancing seed) and partitioned
    across DataLoader workers. Packing logic is identical to
    MixedCodeDataset._pack_tokens.
    """

    def __init__(
        self,
        data_dir: str | Path,
        max_seq_len: int = 2048,
        eos_id: int = 49_090,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.data_dir = Path(data_dir)
        self.max_seq_len = max_seq_len
        self.eos_id = eos_id
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0

        tokens_path = self.data_dir / "tokens.bin"
        index_path = self.data_dir / "index.npy"

        if not tokens_path.exists():
            raise FileNotFoundError(f"Token file not found: {tokens_path}")
        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")

        # Explicit little-endian uint16 for cross-platform compatibility
        self.tokens = np.memmap(tokens_path, dtype="<u2", mode="r")

        self.index = np.load(index_path)  # (N, 2) int64: (offset, length)
        if self.index.ndim == 1 and len(self.index) == 0:
            self.index = self.index.reshape(0, 2)
        if self.index.ndim != 2 or self.index.shape[1] != 2:
            raise ValueError(
                f"Malformed index: expected shape (N, 2), got {self.index.shape}"
            )

        # Load metadata if available
        meta_path = self.data_dir / "metadata.json"
        self.metadata: dict = {}
        if meta_path.exists():
            self.metadata = json.loads(meta_path.read_text(encoding="utf-8"))

    @property
    def total_tokens(self) -> int:
        return len(self.tokens)

    @property
    def num_samples(self) -> int:
        return len(self.index)

    def set_epoch(self, epoch: int) -> None:
        """Set epoch for shuffle seed advancement (call before each epoch)."""
        self._epoch = epoch

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        if self.shuffle:
            epoch_seed = self.seed + self._epoch
            order = np.random.RandomState(epoch_seed).permutation(len(self.index))
            self._epoch += 1
        else:
            order = np.arange(len(self.index))

        # Partition across DataLoader workers for parallel loading
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            order = order[worker_info.id :: worker_info.num_workers]

        yield from self._pack_tokens(self._token_stream(order))

    def _token_stream(
        self, order: np.ndarray
    ) -> Iterator[list[int]]:
        """Yield token ID lists for each sample in the given order."""
        for i in order:
            offset, length = int(self.index[i, 0]), int(self.index[i, 1])
            ids = self.tokens[offset : offset + length].tolist()
            if len(ids) > 0:
                yield ids

    def _pack_tokens(
        self, token_stream: Iterator[list[int]]
    ) -> Iterator[dict[str, torch.Tensor]]:
        """Pack token sequences into fixed-length training samples.

        Identical to MixedCodeDataset._pack_tokens: concatenates samples
        with EOS separators, slices into (max_seq_len + 1) chunks, yields
        input_ids/targets pairs shifted by one position.
        """
        buffer: list[int] = []
        for ids in token_stream:
            buffer.extend(ids)
            buffer.append(self.eos_id)

            while len(buffer) >= self.max_seq_len + 1:
                chunk = buffer[: self.max_seq_len + 1]
                buffer = buffer[self.max_seq_len + 1 :]

                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                targets = torch.tensor(chunk[1:], dtype=torch.long)
                yield {"input_ids": input_ids, "targets": targets}
