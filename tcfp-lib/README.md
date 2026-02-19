# TCFP — Tensor Core Floating Point

**Near-BF16 quality at FP8 speed.**

TCFP-12 decomposes each weight matrix into two FP8 components (`W = W_hi + W_lo`) and computes both GEMMs in a single fused kernel pass. The result is ~12.5 effective bits of precision with throughput that matches or exceeds BF16 on FP8 tensor cores (SM89+).

```
pip install tcfp                  # core (torch >= 2.4)
pip install tcfp[triton]          # + fused Triton kernels (recommended)
```

## Quickstart

```python
import torch
from tcfp import convert_to_tcfp, diagnose

model = YourTransformer().cuda()

# Check compatibility before converting
report = diagnose(model, scale_block_size=128)
print(f"{report.tc_eligible_layers}/{report.total_linear_layers} layers TC-eligible")
print(f"Estimated VRAM savings: {report.estimated_savings_pct:.0f}%")

# Convert — replaces nn.Linear/LayerNorm/Embedding in-place
convert_to_tcfp(
    model,
    use_tensor_cores=True,      # real FP8 GEMMs (not fake-quantize)
    scale_block_size=128,       # per-block scaling (finer than per-tensor)
    error_feedback=True,        # accumulate quantization error across steps
)

# Train as usual — optimizer sees FP32 master weights
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for batch in dataloader:
    loss = model(batch).loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

## What TCFP-12 Does

Standard FP8 (E4M3) has only 3 mantissa bits — fine for inference, lossy for training. TCFP-12 recovers the lost precision:

1. **Quantize** `W` to FP8 -> `W_hi` (captures the main signal)
2. **Compute residual** `R = W - dequant(W_hi)`
3. **Quantize** `R` to FP8 -> `W_lo` (captures what FP8 missed)
4. **Forward pass**: `output = (X @ W_hi) + (X @ W_lo)` via two FP8 GEMMs

The two GEMMs together give ~12.5 bits of effective precision — enough to match BF16 training quality.

## Compute Dispatch

TCFP uses a 3-tier dispatch hierarchy for the dual-GEMM (highest priority first):

| Backend | Requirement | What it does |
|---------|-------------|-------------|
| **cuBLASLt C++ ext** | CUDA Toolkit + `nvcc` | beta=1 accumulation — second GEMM adds directly to first, zero intermediate buffer |
| **Triton fused kernel** | `triton >= 3.0` | Loads activation tile once, computes both dot products, ~25% bandwidth savings |
| **PyTorch fallback** | `torch >= 2.4` | Two `torch._scaled_mm` calls + add (always available) |

All three produce identical numerical results. The dispatch is automatic — TCFP picks the fastest available backend.

## Features

| Feature | Flag | Description |
|---------|------|-------------|
| **Tensor core GEMMs** | `use_tensor_cores=True` | Real FP8 matmuls via `torch._scaled_mm` |
| **Per-block scaling** | `scale_block_size=128` | Finer-grained scales than per-tensor (requires Triton) |
| **Error feedback** | `error_feedback=True` | Accumulates quantization error across optimizer steps |
| **Delayed scaling** | `delayed_scaling=True` | EMA-based scale prediction (skips one amax reduction) |
| **ABD** | `abd=True` | Asymmetric backward — drops `W_lo` from dgrad (saves 1 GEMM/layer/backward) |
| **SRR** | `srr=True` | Stochastic residual rounding — eliminates error feedback buffers |
| **HP grad weight** | `hp_grad_weight=True` | Compute weight gradient in FP32 instead of FP8 |
| **Warmup** | `warmup_steps=100` | Skip quantization for N initial steps |

## Per-Block Scaling

Per-tensor scaling uses a single scale factor for the entire weight matrix. Per-block scaling computes a scale factor per `block_size` chunk along the K dimension, capturing local dynamic range:

```python
convert_to_tcfp(model, use_tensor_cores=True, scale_block_size=64)
```

Block sizes: 32, 64, or 128. Requires Triton for the custom block-scaled GEMM kernel.

## Diagnostics

Run `diagnose()` before conversion to check dimension compatibility and estimate VRAM:

```python
from tcfp import diagnose

report = diagnose(model, scale_block_size=128, sample_batch=dummy_input)

for layer in report.layer_details:
    status = "OK" if layer.tc_eligible else f"SKIP: {layer.fallback_reason}"
    snr = f"{layer.snr_db:.1f} dB" if layer.snr_db else "n/a"
    print(f"  {layer.name}: {status} (SNR={snr})")
```

Layers with SNR below 30 dB are flagged automatically.

## Export for Inference

Strip TCFP wrappers to get vanilla PyTorch modules for deployment:

```python
from tcfp import export

vanilla_model = export(model)  # TCFPLinear -> nn.Linear (shared weights)
torch.save(vanilla_model.state_dict(), "model_inference.pt")
```

## Training Utilities

The `tcfp.training` submodule provides production training tools:

```python
from tcfp.training import (
    TCFPCheckpointer,       # checkpoint save/resume with EF state
    TCFPMonitor,            # loss/gradient health tracking
    ProgressiveQuantizer,   # staged quantization curriculum
    TrainingPreset,         # battle-tested config presets
    apply_highway_routing,  # per-layer sensitivity-based routing
)
```

## Hardware Requirements

- **GPU**: NVIDIA Ada Lovelace (SM89), Hopper (SM90), or Blackwell (SM100+)
- **PyTorch**: >= 2.4.0 (for `torch._scaled_mm` with FP8)
- **Triton**: >= 3.0.0 (optional, for fused kernels and per-block scaling)
- **CUDA Toolkit**: optional, for cuBLASLt C++ extension

CPU and older GPUs fall back to the fake-quantize path (STE + F.linear).

## Comparison with Other Approaches

| Method | Effective Bits | Tensor Cores | Training | Library |
|--------|---------------|-------------|----------|---------|
| BF16 | 16 | Yes | Yes | PyTorch native |
| FP8 (E4M3) | 8 | Yes | Lossy | `torch._scaled_mm` |
| INT4 (NF4) | 4 | No | QLoRA only | bitsandbytes |
| **TCFP-12** | **~12.5** | **Yes** | **Full** | **this library** |

TCFP-12 fills the gap between FP8 (too lossy for training) and BF16 (too slow for FP8 hardware).

## License

Apache-2.0
