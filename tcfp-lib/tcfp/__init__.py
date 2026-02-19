"""
TCFP — Tensor Core Floating Point
==================================

Near-BF16 quality at FP8 speed via dual-GEMM residual decomposition.

**TCFP-12** decomposes each weight matrix into two FP8 components:
``W ≈ W_hi + W_lo``, then computes ``output = (A @ W_hi) + (A @ W_lo)``
using two FP8 GEMMs.  The result is ~12.5 effective bits of precision
at better-than-BF16 throughput on FP8 tensor cores.

Quickstart::

    import torch
    from tcfp import convert_to_tcfp

    model = YourModel().cuda()
    convert_to_tcfp(model, use_tensor_cores=True)
    # Train as usual — optimizer sees FP32 master weights

Public API (most users need only these):

- :func:`convert_to_tcfp` — convert any model's nn.Linear/LayerNorm/Embedding
- :func:`diagnose` — preflight check: dimension compatibility, VRAM estimates, SNR
- :func:`export` — strip TCFP wrappers for inference deployment
- :class:`TCFPLinear` — drop-in nn.Linear replacement (for manual construction)
- :class:`TCFPMode` — precision mode enum
"""

from __future__ import annotations

# Core primitives
from .core import (
    ErrorFeedbackState,
    FP8Config,
    TCFPMode,
    ensure_column_major,
    fp8_matmul,
    to_fp8_e4m3,
    to_fp8_e4m3_nf_aware,
    to_fp8_e4m3_sr,
    to_fp8_e5m2,
)

# Drop-in neural network modules (the main user-facing API)
from .nn import (
    TCFPEmbedding,
    TCFPLayerNorm,
    TCFPLinear,
    convert_to_tcfp,
    diagnose,
    export,
)

__all__ = [
    # High-level API (most users)
    "convert_to_tcfp",
    "diagnose",
    "export",
    "TCFPLinear",
    "TCFPLayerNorm",
    "TCFPEmbedding",
    "TCFPMode",
    # Core primitives (advanced users)
    "ErrorFeedbackState",
    "FP8Config",
    "ensure_column_major",
    "fp8_matmul",
    "to_fp8_e4m3",
    "to_fp8_e4m3_nf_aware",
    "to_fp8_e4m3_sr",
    "to_fp8_e5m2",
]

__version__ = "0.1.0"
