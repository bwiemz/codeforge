# CodeForge

**A code-generation language model trained from scratch in FP8.**

CodeForge is a decoder-only transformer designed for code synthesis and understanding, built on a custom quantization format (TCFP) that enables efficient training on consumer GPUs. The architecture is *born-quantized*: it trains natively in FP8 from step 0, eliminating the precision loss typically introduced by post-training quantization.

---

## Key ideas

| Concept | What it does |
|---|---|
| **TCFP-12** | Two-GEMM residual FP8 decomposition achieving ~12 effective bits from two 8-bit matmuls. 19% faster than BF16 on FP8-capable hardware (Hopper, Ada Lovelace, Blackwell). |
| **Reclaimer Protocol** | Born-quantized training regime: QK-RMSNorm with frozen gains, post-norm (FOG) placement, z-loss stabilization, Smooth-SwiGLU activations, depth-scaled init, separate embedding LR. No full-precision warmup required. |
| **ZSFTP** | Zero-Sync Fused Training Pipeline. GPU-resident loss accumulation, fused AdamW, chunked cross-entropy (75% peak VRAM reduction), `torch.compile` with Inductor, non-blocking prefetch. |
| **Pre-tokenization** | Optional offline pipeline that runs the full data stack (quality filtering, dedup, decontamination, FIM) once and saves token IDs to a flat memmap file. Training then reads sequential uint16 tokens with near-zero I/O overhead. |

## Model presets

| Preset | Dim | Layers | Heads | KV Heads | Approx. params |
|--------|-----|--------|-------|----------|----------------|
| `150m` | 768 | 12 | 12 | 4 | 150 M |
| `350m` | 1024 | 24 | 16 | 4 | 350 M |
| `1b` | 2048 | 24 | 16 | 8 | 1 B |
| `3b` | 3072 | 26 | 24 | 8 | 3 B |

All presets use Grouped Query Attention, RoPE (with optional linear/NTK/YaRN scaling), SwiGLU feed-forward (Smooth-SwiGLU for stability at scale), RMSNorm, and weight-tied embeddings.

## Architecture

```
Input tokens
    │
    ▼
Embedding (vocab 49 152 × dim, weight-tied with output head)
    │
    ▼
┌──────────────────────────────────────────────┐
│  TransformerBlock  ×  n_layers               │
│  ┌────────────────────────────────────────┐  │
│  │ Grouped Query Attention (RoPE, GQA)    │  │
│  │   + QK-RMSNorm (frozen gains)          │  │
│  │   + optional KV-cache for inference    │  │
│  └────────────────────────────────────────┘  │
│  Post-Norm (RMSNorm)                         │
│  Residual add                                │
│  ┌────────────────────────────────────────┐  │
│  │ Smooth-SwiGLU Feed-Forward             │  │
│  │   gate, up, down projections           │  │
│  └────────────────────────────────────────┘  │
│  Post-Norm (RMSNorm)                         │
│  Residual add                                │
└──────────────────────────────────────────────┘
    │
    ▼
Final RMSNorm → Linear (weight-tied) → logits
```

---

## TCFP-12: Two-GEMM Residual FP8 Decomposition

### The problem: FP8 alone loses too much precision

FP8 E4M3 has only 3 mantissa bits, giving it ~8x larger quantization step sizes than BF16 (7 mantissa bits). When every linear layer in a transformer quantizes weights to FP8, the accumulated rounding error degrades training convergence. Naive FP8 training typically requires a warmup phase in higher precision, and even then loses 1-2% final accuracy.

TCFP-12 solves this by **decomposing each weight matrix into two FP8 components** and running two GEMMs instead of one:

```
W ≈ W_hi + W_lo

where:
  W_hi = dequant(quant_fp8(W))     — main FP8 component (captures bulk signal)
  W_lo = dequant(quant_fp8(R))     — residual FP8 component (captures what W_hi missed)
  R    = W − W_hi                  — quantization residual
```

The output of a linear layer `Y = X @ W^T` becomes:

```
Y = X @ W_hi^T + X @ W_lo^T       — two FP8 GEMMs, summed in FP32
```

This is analogous to **double-double arithmetic**: two lower-precision numbers carry a high-precision value, like storing `π ≈ 3.14 + 0.001593` as two separate floats.

### Why two GEMMs work

The key observation is that after quantizing W to FP8, the residual `R = W − dequant(quant(W))` is bounded by the unit-in-the-last-place (ULP) of the FP8 representation. For E4M3 with 3 mantissa bits, the worst-case rounding error is half the ULP: `max(|R|) ≤ max(|W|) / (2 × 2^3) = max(|W|) / 16`. This residual is small enough to fit cleanly into a second FP8 representation with its own dedicated scale. When the second FP8 quantization introduces its own rounding error, that error is bounded by `max(|R|) / 16 ≈ max(|W|) / 256`, yielding a total reconstruction error equivalent to roughly **5-6 effective mantissa bits** — up from 3 for a single FP8, and approaching BF16's 7.

### Forward pass — step by step

Every `TCFPLinear.forward()` call executes the following pipeline. Code references are to `tcfp-lib/tcfp/nn/__init__.py`, class `_TCFP12TensorCoreFunction` (per-tensor scaling path). A second implementation, `_TCFP12BlockScaledFunction`, uses per-block Triton GEMM kernels with fused error feedback — same mathematical decomposition, different scaling granularity and kernel dispatch.

**Step 1 — Error feedback injection** (line 344-347)

```
W' = W_master + E_{t-1}
```

Before quantizing, we add back the quantization error from the previous training step. Without this, quantization errors accumulate silently over thousands of steps, causing weight drift. Error feedback ensures that **any value lost to rounding in step t is recovered in step t+1**. Over time, `E[W_quantized] = W_master` — the quantized weights are an unbiased estimator of the true weights.

*Why not just use higher precision?* Because the error feedback buffer is FP32 but only stores the small quantization residual (typically 1-2 orders of magnitude smaller than the weight), the memory cost is fixed and independent of the number of GEMM operations.

**Step 2 — Quantize W_hi (main component)** (lines 350-361)

```
s_hi     = 448 / amax(|W'|)         — per-tensor scale (maps max value to FP8 range)
W_hi_fp8 = round_fp8(W' × s_hi)     — cast to FP8 E4M3 (3 mantissa bits)
s_hi_inv = 1 / s_hi                  — inverse scale for dequantization
```

The scale `s_hi` maps the weight tensor's dynamic range into FP8 E4M3's representable range [−448, +448]. With delayed scaling (see optimization section below), this uses an EMA-smoothed amax instead of a fresh reduction.

**Step 3 — Compute and quantize residual W_lo** (lines 363-372)

```
R        = W' − (W_hi_fp8 × s_hi_inv)   — what the main FP8 component missed
s_lo     = 448 / amax(|R|)               — scale for the residual
W_lo_fp8 = round_fp8(R × s_lo)           — residual in FP8
s_lo_inv = 1 / s_lo
```

The residual R contains the quantization error of step 2. It is typically 8-16x smaller than the original weight (bounded by the E4M3 quantization step), so it fits well into another FP8 tensor with its own scale.

**Step 4 — Update error feedback buffer** (lines 378-386)

```
W_reconstructed = (W_hi_fp8 × s_hi_inv) + (W_lo_fp8 × s_lo_inv)
E_t             = W' − W_reconstructed
```

The remaining error after both quantization stages is stored for the next step. This is the "residual of the residual" — typically 64x smaller than the original weight, making it negligible for any single step but important for long-term drift prevention.

**Step 5 — Quantize activations** (line 393)

```
A_fp8, s_a = quant_fp8(X)     — always fresh per-tensor scale (activations change every step)
```

Activations are quantized to FP8 E4M3 with a fresh scale computed from the current input. Unlike weights, activations have no error feedback because they are ephemeral — each forward pass produces new activations.

**Step 6 — Two FP8 GEMMs** (lines 396-424)

```
Y_hi = _scaled_mm(A_fp8, W_hi_fp8^T, scale_a=s_a_inv, scale_b=s_hi_inv)   — GEMM 1
Y_lo = _scaled_mm(A_fp8, W_lo_fp8^T, scale_a=s_a_inv, scale_b=s_lo_inv)   — GEMM 2
Y    = Y_hi + Y_lo                                                          — sum in FP32
```

Note: `torch._scaled_mm` takes **inverse scales** (dequantization factors) as arguments. Both `s_a_inv = 1/s_a` and `s_hi_inv = 1/s_hi` are the values returned by `to_fp8_e4m3()`. The hardware multiplies `(A_fp8 × s_a_inv) @ (W_fp8 × s_inv)` internally using FP8 tensor cores with FP32 accumulation (`use_fast_accum=False` to prevent FP16 accumulator rounding). The activation tensor is quantized once and reused for both GEMMs.

Three dispatch backends are tried in order:
1. **cuBLASLt C++ extension** — fused dual GEMM in a single CUDA kernel launch
2. **Triton fused kernel** — JIT-compiled dual GEMM with activation-quantization fusion
3. **2× `torch._scaled_mm`** — PyTorch's built-in FP8 matmul (fallback, always available)

**Step 7 — Add bias and reshape** (lines 427-430)

```
Y = Y + bias    (if present)
Y = reshape(Y, original_batch_shape)
```

### Backward pass — step by step

Class `_TCFP12TensorCoreFunction.backward()` (line 464).

**Step 1 — Quantize gradient to FP8 E5M2** (line 497)

```
G_fp8, s_g = quant_fp8_e5m2(dL/dY)
```

Gradients use **E5M2** (5 exponent bits, 2 mantissa bits) instead of E4M3 because gradients have a much wider dynamic range than weights. E5M2 covers ±57344 vs E4M3's ±448. The lower mantissa precision (2 bits vs 3) is acceptable because gradient direction matters more than magnitude for convergence.

*Why not delayed scaling for gradients?* Gradients change dramatically between steps (unlike weights which update incrementally), so an EMA-smoothed scale would be dangerously stale. Fresh per-tensor scaling is mandatory.

**Step 2 — Input gradient (dX): two FP8 GEMMs** (lines 499-540)

```
dX = G_fp8 @ W_hi + G_fp8 @ W_lo        — full path: 2 GEMMs
dX = G_fp8 @ W_hi                        — ABD path: 1 GEMM (approximate)
```

The input gradient needs the full weight to propagate correct gradients to earlier layers. The full (non-ABD) path runs two GEMMs just like the forward pass. The **ABD (Asymmetric Backward Decomposition)** optimization drops W_lo from this computation, saving one GEMM per layer at the cost of slightly noisier input gradients. This is acceptable because:
- W_lo is ~8x smaller than W_hi, so the gradient error is bounded
- SGD already introduces gradient noise via mini-batching, so a small additional approximation has minimal effect

**Step 3 — Weight gradient (dW): single GEMM** (lines 542-558)

```
dW = G^T @ A        — single GEMM (no residual decomposition needed)
```

The weight gradient does not need two-GEMM decomposition because it accumulates into the FP32 master weight via the optimizer. The result of this GEMM is always FP32 regardless of input precision.

Three paths exist depending on configuration:
- **`hp_grad_weight=True`**: Uses the full BF16/FP32 activation tensor saved during forward. Both inputs are high-precision — highest quality gradient at the cost of saving a larger activation tensor.
- **`hp_grad_weight=False`, aligned dims**: Uses `_scaled_mm` with the FP8 gradient and FP8 saved activation, accumulating to FP32. The inputs are lower precision but the accumulator is not.
- **`hp_grad_weight=False`, unaligned dims**: Dequantizes the FP8 activation to FP32 before a standard matmul (fallback for dimensions not divisible by 16).

**Step 4 — Bias gradient** (line 561)

```
dBias = sum(G, dim=batch)    — reduction over batch dimension, FP32
```

### GEMM count summary

| Path | Forward | dX (backward) | dW (backward) | Total per layer |
|------|---------|---------------|---------------|-----------------|
| Full (non-ABD) | 2 | 2 | 1 | 5 |
| ABD | 2 | 1 | 1 | 4 |
| BF16 baseline | 1 | 1 | 1 | 3 |

TCFP uses 67% more GEMMs than BF16 (full path) but each FP8 GEMM runs at roughly 2x the FLOPS throughput on tensor cores. The net wall-clock speedup depends on the compute-vs-memory-bandwidth balance of the specific GPU and layer dimensions; on Ada Lovelace (RTX 4090), the measured benefit is ~19% end-to-end including all non-GEMM overhead.

### Error feedback: why it matters

Without error feedback, the quantization error `E_t = W − Q(W)` is silently discarded every step. Over thousands of steps, this creates **systematic weight drift**: if a weight `w = 0.126` always rounds down to `0.125` in FP8, the optimizer keeps trying to push it to `0.126` but the quantized value never reaches it. The gradient signal is wasted.

Error feedback stores `E_t` and adds it back before quantizing at step `t+1`:

```
Step t:    Q(W_t)           = 0.125, error = 0.001
Step t+1:  Q(W_{t+1} + 0.001) = Q(0.127) = 0.125, error = 0.002
Step t+2:  Q(W_{t+2} + 0.002) = Q(0.128) = 0.125, error = 0.003
...
Step t+k:  Q(W_{t+k} + E)    = Q(0.130) = 0.1875  ← finally rounds up
```

The accumulated error eventually pushes the quantized value past the rounding boundary. Over many steps, `E[Q(W + E)] = W` — the quantization becomes unbiased.

The error buffer is FP32 and requires one buffer per tensor-core-eligible weight tensor (layers skipped by `skip_patterns` or with dimensions not divisible by 16 do not get EF buffers). For a 150M parameter model where most linear layers are TC-eligible, this adds up to ~600 MB of VRAM on top of the existing FP32 master weights. This is the primary memory cost of TCFP beyond vanilla FP8.

### Delayed scaling: eliminating global reductions

Computing `amax(|W|)` requires a global reduction over the entire weight tensor — an all-elements max that synchronizes the GPU. For a 768×2048 weight matrix, that's 1.5M comparisons per layer per step.

Delayed scaling replaces this with an **exponential moving average**:

```
amax_delayed_t = α × amax_delayed_{t-1} + (1 − α) × amax_fresh_t
```

With `α = 0.999`, the delayed amax tracks the true amax closely (weights change by ~0.1% per step, so the scale drifts by <0.001% per step). This eliminates the amax reduction for W_hi.

For W_lo, we go further with **predictive scaling**: since `amax(R) ≈ amax(W) / 8` (bounded by the FP8 quantization step), we maintain an EMA of the ratio `amax_lo / amax_hi` and predict the residual's amax from the main component's amax — eliminating a second reduction entirely.

The ratio is periodically recalibrated (every 100 steps by default) by computing the actual residual amax and updating the EMA.

### NF-aware scaling: optimizing for Gaussian distributions

Neural network weights are approximately Gaussian-distributed after initialization and remain so during training (per the central limit theorem applied to gradient updates). The standard scaling `s = 448 / max(|W|)` wastes dynamic range on outliers — a single extreme value forces the scale to accommodate it, reducing precision for the 99.7% of values that are smaller.

NF-aware scaling uses the standard deviation instead:

```
s = 448 / (3 × std(W))
```

This maps 99.7% of values (within 3 standard deviations) into the full FP8 range, giving them maximum precision. The ~0.3% of outliers beyond 3σ are clipped to ±448, which causes negligible MSE because they are rare and the clipping error is bounded by the outlier's distance from the clip boundary.

For the 150M model, this typically improves quantization SNR by 2-4 dB compared to max-based scaling.

### Straight-Through Estimator (STE): training through quantization

Quantization is a step function (not differentiable), so the backward pass cannot compute `∂Q(W)/∂W` directly. The **Straight-Through Estimator** solves this by defining:

```
Forward: W_q = Q(W)                — quantized weight used in computation
Backward: ∂L/∂W = ∂L/∂W_q × 1    — gradient passes through Q as if it were identity
```

This is a biased gradient estimator, but it works well in practice because the quantization error is small relative to the gradient magnitude. The optimizer receives a gradient that points in approximately the right direction, and the error feedback mechanism corrects any systematic bias over time.

### Born-quantized training

Traditional quantized training uses a warmup phase in FP32/BF16 before switching to lower precision. CodeForge trains in FP8 **from step 0** (`tcfp_warmup_steps=0`). This works because:

1. **Error feedback + delayed scaling** prevent the quantization from introducing catastrophic rounding in early training
2. **Post-norm placement** (Reclaimer Protocol) keeps activations bounded, preventing outliers that would destabilize FP8 scaling
3. **QK-RMSNorm** with frozen gains prevents attention logit growth, which is the primary failure mode of low-precision training
4. **z-loss** (`1e-4 × mean(logits²)`) penalizes large logits at the output layer, keeping the softmax numerically stable

The model learns "quantization-friendly" weight distributions from the start, making post-training quantization more accurate when exporting to INT8/INT4 for inference.

---

## TCFP-12 storage format (offline/checkpoint)

Each scalar is stored as an FP8 E4M3 main value plus a 4-bit residual correction with shared per-block scales:

```
Per block of B values:
  1.  S     = pow2 block scale               (8 bits)
  2.  x_hi  = fp8_e4m3(x / S)               (8 bits)
  3.  r     = x/S − dequant(x_hi)
  4.  S_r   = pow2 residual scale            (8 bits)
  5.  x_lo  = int4(r / S_r)                  (4 bits)

Reconstruction:  x ≈ S · (dequant(x_hi) + S_r · x_lo)
Storage:         12.5 bits/value (B = 32), 25 % less than BF16
```

Optional enhancements: NF4 residual codebook, sigma-based (NF-aware) block scales, stochastic rounding, per-parameter error feedback with EMA-smoothed amax.

---

## Repository layout

```
codeforge/
├── model/          Transformer architecture, config presets, GQA, RoPE
├── tcfp/           TCFP quantization core, Triton kernels, cuBLASLt ext
│   ├── tcfp12/     FP8 + 4-bit residual (TCFP-12)
│   ├── nn/         Quantized nn.Linear and related layers
│   └── training/   TCFP monitoring, presets, curriculum, checkpointing
├── tokenizer/      49K-vocab BPE tokenizer (code-optimized)
├── data/           Dataset pipeline: quality scoring, dedup, FIM, mixing,
│                   pre-tokenized memmap support
├── training/       Trainer (ZSFTP), SFT, DPO, schedulers, optimizers
├── inference/      Generator with KV-cache, speculative decoding
├── eval/           HumanEval, MBPP harnesses
└── export/         GGUF export, post-training INT8/INT4 quantization
configs/            YAML presets for model sizes and training runs
scripts/            Entry points: train, generate, evaluate, export,
                    pretokenize, train_sft, train_dpo, train_tokenizer
tests/              pytest suite (480+ tests: TCFP, tokenizer, model)
```

## Setup

Requires Python 3.10+ and PyTorch 2.4+.

```bash
# Core install
pip install -e .

# With Triton kernels, W&B tracking, and dev tools
pip install -e ".[all]"
```

Optional dependency groups: `triton`, `tracking`, `quantize`, `dev`.

## Usage

### Pre-tokenization (recommended)

Pre-tokenize once, then train from fast local memmap files instead of streaming from HuggingFace:

```bash
python scripts/pretokenize.py \
  --config configs/pretrain_150m.yaml \
  --output data/pretokenized/150m \
  --max-tokens 30000000000
```

This runs the full data pipeline (quality filtering, dedup, decontamination, FIM) and writes:
- `tokens.bin` — flat uint16 memmap of all token IDs
- `index.npy` — per-sample `(offset, length)` index for shuffling
- `metadata.json` — config snapshot and filtering stats

### Pretraining

```bash
# From pre-tokenized data (fast, deterministic)
codeforge-train --config configs/pretrain_150m.yaml \
  --pretokenized data/pretokenized/150m

# From HuggingFace streaming (no pre-tokenization needed)
codeforge-train --config configs/pretrain_150m.yaml

# Smoke test (synthetic data, no data pipeline)
codeforge-train --smoke-test --preset 150m

# Override specific fields
codeforge-train --config configs/pretrain_150m.yaml \
  --batch-size 8 --lr 3e-4 --max-steps 50000
```

### Supervised fine-tuning

```bash
python scripts/train_sft.py \
  --checkpoint checkpoints/pretrain_150m/latest.pt \
  --tokenizer tokenizer/codeforge.json \
  --config configs/sft_150m.yaml
```

### DPO alignment

```bash
python scripts/train_dpo.py \
  --checkpoint checkpoints/pretrain_150m/latest.pt \
  --tokenizer tokenizer/codeforge.json
```

### Generation

```bash
# Interactive
codeforge-generate -c checkpoints/pretrain_150m/latest.pt \
  -t tokenizer/codeforge.json

# Single prompt
codeforge-generate -c checkpoints/pretrain_150m/latest.pt \
  -t tokenizer/codeforge.json \
  -p "def quicksort(arr):" --max-tokens 256

# Fill-in-the-Middle
codeforge-generate -c checkpoints/pretrain_150m/latest.pt \
  -t tokenizer/codeforge.json \
  --fim -p "def func():|||    return result"
```

### Evaluation

```bash
python scripts/evaluate.py \
  --checkpoint checkpoints/pretrain_150m/latest.pt \
  --tokenizer tokenizer/codeforge.json \
  --benchmark humaneval   # or mbpp
```

### Export

```bash
# GGUF (llama.cpp / Ollama)
python scripts/export.py \
  --checkpoint checkpoints/pretrain_150m/latest.pt \
  --output model.gguf

# Post-training quantization
python scripts/export.py --quantize int8 \
  --checkpoint checkpoints/pretrain_150m/latest.pt
```

## Training configuration

Key fields from `configs/pretrain_150m.yaml`:

```yaml
model: configs/model_150m.yaml     # architecture (post-norm, QK-norm, etc.)

data:
  hf_dataset: "bigcode/starcoderdata"
  languages: [python, javascript, typescript, java, cpp, c, go, rust]
  fim_rate: 0.5
  quality_threshold: 0.35
  enable_dedup: true

training:
  batch_size: 8
  gradient_accumulation_steps: 32   # effective batch = 256
  learning_rate: 3.0e-4
  max_steps: 45000
  precision: bf16
  scheduler_type: wsd               # Warmup-Stable-Decay
  warmup_steps: 1500
  embed_lr_ratio: 0.1               # separate embedding LR
  use_tcfp: true
  tcfp_warmup_steps: 0              # born-quantized from step 0
  tcfp_delayed_scaling: true         # EMA-smoothed amax for stability
  tcfp_abd: false                    # conservative: both GEMMs in backward
  tcfp_srr: false                    # quality-first: EF + delayed_scaling
```

## Tests

```bash
pytest tests/                  # full suite
pytest tests/test_tcfp/        # quantization tests
pytest tests/test_tokenizer/   # tokenizer tests
```

Linting and type checking:

```bash
ruff check .
basedpyright
```

## License

Apache 2.0
