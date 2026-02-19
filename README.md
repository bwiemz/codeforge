# CodeForge

**A code-generation language model trained from scratch in FP8.**

CodeForge is a decoder-only transformer designed for code synthesis and understanding, built on a custom quantization format (TCFP) that enables efficient training on consumer GPUs. The architecture is *born-quantized*: it trains natively in FP8 from step 0, eliminating the precision loss typically introduced by post-training quantization.

---

## Key ideas

| Concept | What it does |
|---|---|
| **TCFP-12** | FP8 E4M3 + 4-bit residual correction. ~12.5 bits/value, ~5-6 effective mantissa bits. 19% faster than BF16 on FP8-capable hardware (Hopper, Ada Lovelace, Blackwell). |
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

## TCFP-12 format

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
