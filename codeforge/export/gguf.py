"""GGUF format export for llama.cpp / Ollama compatibility.

GGUF (GGML Universal Format) is the standard format used by
llama.cpp, ollama, and other local inference engines.
"""

import struct
import numpy as np
from pathlib import Path
from typing import Optional

import torch

from ..model.config import ModelConfig
from ..model.transformer import CodeForgeModel

# GGUF magic number and version
GGUF_MAGIC = 0x46554747  # "GGUF" in little-endian
GGUF_VERSION = 3

# GGUF value types
GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12

# GGML tensor types
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q4_1 = 3
GGML_TYPE_Q8_0 = 8


class GGUFWriter:
    """Writes a GGUF file from a CodeForge model."""

    def __init__(self):
        self.metadata: list[tuple[str, int, bytes]] = []
        self.tensors: list[tuple[str, np.ndarray, int]] = []

    def _encode_string(self, s: str) -> bytes:
        encoded = s.encode("utf-8")
        return struct.pack("<Q", len(encoded)) + encoded

    def add_string(self, key: str, value: str) -> None:
        data = self._encode_string(value)
        self.metadata.append((key, GGUF_TYPE_STRING, data))

    def add_uint32(self, key: str, value: int) -> None:
        data = struct.pack("<I", value)
        self.metadata.append((key, GGUF_TYPE_UINT32, data))

    def add_uint64(self, key: str, value: int) -> None:
        data = struct.pack("<Q", value)
        self.metadata.append((key, GGUF_TYPE_UINT64, data))

    def add_float32(self, key: str, value: float) -> None:
        data = struct.pack("<f", value)
        self.metadata.append((key, GGUF_TYPE_FLOAT32, data))

    def add_bool(self, key: str, value: bool) -> None:
        data = struct.pack("<B", int(value))
        self.metadata.append((key, GGUF_TYPE_BOOL, data))

    def add_tensor(self, name: str, tensor: np.ndarray, dtype: int = GGML_TYPE_F16) -> None:
        if dtype == GGML_TYPE_F16:
            tensor = tensor.astype(np.float16)
        elif dtype == GGML_TYPE_F32:
            tensor = tensor.astype(np.float32)
        self.tensors.append((name, tensor, dtype))

    def write(self, path: str | Path) -> None:
        """Write the GGUF file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            # Header
            f.write(struct.pack("<I", GGUF_MAGIC))
            f.write(struct.pack("<I", GGUF_VERSION))
            f.write(struct.pack("<Q", len(self.tensors)))  # n_tensors
            f.write(struct.pack("<Q", len(self.metadata)))  # n_kv

            # Metadata KV pairs
            for key, vtype, data in self.metadata:
                f.write(self._encode_string(key))
                f.write(struct.pack("<I", vtype))
                f.write(data)

            # Tensor info
            tensor_data_list = []
            offset = 0

            for name, tensor, dtype in self.tensors:
                f.write(self._encode_string(name))
                f.write(struct.pack("<I", len(tensor.shape)))  # n_dims
                for dim in tensor.shape:
                    f.write(struct.pack("<Q", dim))
                f.write(struct.pack("<I", dtype))
                f.write(struct.pack("<Q", offset))

                raw = tensor.tobytes()
                tensor_data_list.append(raw)
                offset += len(raw)

                # Alignment padding
                padding = (32 - (len(raw) % 32)) % 32
                offset += padding

            # Alignment before tensor data
            current = f.tell()
            padding = (32 - (current % 32)) % 32
            f.write(b"\x00" * padding)

            # Tensor data
            for i, raw in enumerate(tensor_data_list):
                f.write(raw)
                padding = (32 - (len(raw) % 32)) % 32
                f.write(b"\x00" * padding)


# llama.cpp name mapping for CodeForge tensors
TENSOR_NAME_MAP = {
    "tok_embeddings.weight": "token_embd.weight",
    "norm.weight": "output_norm.weight",
    "output.weight": "output.weight",
}


def _layer_tensor_name(layer_idx: int, suffix: str) -> str:
    """Map CodeForge layer tensor names to llama.cpp names."""
    mappings = {
        "attn_norm.weight": f"blk.{layer_idx}.attn_norm.weight",
        "attn.wq.weight": f"blk.{layer_idx}.attn_q.weight",
        "attn.wk.weight": f"blk.{layer_idx}.attn_k.weight",
        "attn.wv.weight": f"blk.{layer_idx}.attn_v.weight",
        "attn.wo.weight": f"blk.{layer_idx}.attn_output.weight",
        "attn.q_norm.weight": f"blk.{layer_idx}.attn_q_norm.weight",
        "attn.k_norm.weight": f"blk.{layer_idx}.attn_k_norm.weight",
        "ffn_norm.weight": f"blk.{layer_idx}.ffn_norm.weight",
        "ffn.gate_proj.weight": f"blk.{layer_idx}.ffn_gate.weight",
        "ffn.up_proj.weight": f"blk.{layer_idx}.ffn_up.weight",
        "ffn.down_proj.weight": f"blk.{layer_idx}.ffn_down.weight",
    }
    return mappings.get(suffix, f"blk.{layer_idx}.{suffix}")


def export_to_gguf(
    model: CodeForgeModel,
    output_path: str | Path,
    dtype: str = "f16",
) -> None:
    """Export a CodeForge model to GGUF format.

    Args:
        model: The model to export
        output_path: Path for the output .gguf file
        dtype: Weight dtype ("f16" or "f32")
    """
    config = model.config
    writer = GGUFWriter()

    # Architecture metadata
    writer.add_string("general.architecture", "llama")
    writer.add_string("general.name", "CodeForge")
    writer.add_uint32("llama.context_length", config.max_seq_len)
    writer.add_uint32("llama.embedding_length", config.dim)
    writer.add_uint32("llama.block_count", config.n_layers)
    writer.add_uint32("llama.feed_forward_length", config.ffn_hidden_dim)
    writer.add_uint32("llama.attention.head_count", config.n_heads)
    writer.add_uint32("llama.attention.head_count_kv", config.n_kv_heads)
    writer.add_float32("llama.attention.layer_norm_rms_epsilon", config.norm_eps)
    writer.add_float32("llama.rope.freq_base", config.rope_theta)
    writer.add_uint32("llama.vocab_size", config.vocab_size)
    writer.add_uint32("general.file_type", 1 if dtype == "f16" else 0)

    ggml_dtype = GGML_TYPE_F16 if dtype == "f16" else GGML_TYPE_F32

    # Export tensors
    state_dict = model.state_dict()

    for key, tensor in state_dict.items():
        np_tensor = tensor.cpu().float().numpy()

        # Map to llama.cpp tensor names
        if key in TENSOR_NAME_MAP:
            gguf_name = TENSOR_NAME_MAP[key]
        elif key.startswith("layers."):
            parts = key.split(".", 2)
            layer_idx = int(parts[1])
            suffix = parts[2]
            gguf_name = _layer_tensor_name(layer_idx, suffix)
        else:
            gguf_name = key

        writer.add_tensor(gguf_name, np_tensor, ggml_dtype)

    writer.write(output_path)
    print(f"GGUF exported to {output_path}")
    print(f"  Format: {dtype}, Tensors: {len(state_dict)}")
