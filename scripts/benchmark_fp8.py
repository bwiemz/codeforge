"""Benchmark FP8 vs bf16 on actual CodeForge 150M model."""
import os
import sys
import time

os.environ["TORCHINDUCTOR_CACHE_DIR"] = r"C:\tc"
os.environ["TRITON_CACHE_DIR"] = r"C:\tc\triton"

sys.path.insert(0, r"c:\Users\bwiem\ai testing\codeforge")

import torch
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True

from torchao.float8 import Float8LinearConfig, convert_to_float8_training
from codeforge.model.config import get_preset
from codeforge.model.transformer import CodeForgeModel

device = torch.device("cuda")
config = get_preset("150m")
config.use_gradient_checkpointing = True

batch_size = 8
seq_len = 2048
grad_accum = 32

input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
targets = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)


def benchmark_step(model, optimizer, label, num_microsteps=32):
    """Run one full optimizer step and time it."""
    model.train()
    accum_loss = torch.tensor(0.0, device=device)

    # Warmup (3 micro-steps)
    for _ in range(3):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _, loss, _ = model(input_ids, targets)
            loss = loss / grad_accum
        loss.backward()
        optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

    # Benchmark one full optimizer step
    t0 = time.time()
    for i in range(num_microsteps):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _, loss, _ = model(input_ids, targets)
            loss = loss / grad_accum
        loss.backward()
        accum_loss += loss.detach()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    elapsed = time.time() - t0

    loss_val = accum_loss.item()
    peak_mem = torch.cuda.max_memory_allocated() / (1024**3)

    print(f"{label}:")
    print(f"  Step time: {elapsed:.2f}s ({elapsed/num_microsteps*1000:.0f}ms/microstep)")
    print(f"  Loss: {loss_val:.4f}")
    print(f"  Peak VRAM: {peak_mem:.2f} GB")
    print(f"  Tokens/sec: {batch_size * seq_len * num_microsteps / elapsed:.0f}")
    return elapsed, peak_mem


# ---- BASELINE: bf16 + torch.compile (current ZSFTP) ----
print("=" * 60)
print("BASELINE: bf16 + torch.compile + fused AdamW")
print("=" * 60)
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

model_bf16 = CodeForgeModel(config).to(device)
model_bf16_compiled = torch.compile(model_bf16, backend="inductor")
opt_bf16 = torch.optim.AdamW(model_bf16.parameters(), lr=3e-4, betas=(0.9, 0.95), fused=True)

bf16_time, bf16_mem = benchmark_step(model_bf16_compiled, opt_bf16, "bf16 + compile")

del model_bf16, model_bf16_compiled, opt_bf16
torch.cuda.empty_cache()

# ---- FP8: bf16 + FP8 linear + torch.compile ----
print()
print("=" * 60)
print("FP8: bf16 autocast + FP8 Linear layers + torch.compile + fused AdamW")
print("=" * 60)
torch.cuda.reset_peak_memory_stats()

model_fp8 = CodeForgeModel(config).to(device)

# Convert all nn.Linear layers to Float8Linear for FP8 matmuls
fp8_config = Float8LinearConfig()
convert_to_float8_training(model_fp8, config=fp8_config)

# Count FP8 layers
fp8_count = sum(1 for m in model_fp8.modules() if "Float8" in type(m).__name__)
linear_count = sum(1 for m in model_fp8.modules() if isinstance(m, torch.nn.Linear))
print(f"  Converted {fp8_count} layers to FP8 ({linear_count} remaining as standard Linear)")

model_fp8_compiled = torch.compile(model_fp8, backend="inductor")
opt_fp8 = torch.optim.AdamW(model_fp8.parameters(), lr=3e-4, betas=(0.9, 0.95), fused=True)

fp8_time, fp8_mem = benchmark_step(model_fp8_compiled, opt_fp8, "FP8 + compile")

# ---- Summary ----
print()
print("=" * 60)
print("COMPARISON")
print("=" * 60)
speedup = bf16_time / fp8_time
mem_savings = (bf16_mem - fp8_mem) / bf16_mem * 100
remaining_steps = 86000
bf16_hours = bf16_time * remaining_steps / 3600
fp8_hours = fp8_time * remaining_steps / 3600

print(f"  Speed: {speedup:.2f}x {'faster' if speedup > 1 else 'slower'}")
print(f"  Memory: {mem_savings:.1f}% {'saved' if mem_savings > 0 else 'more'}")
print(f"  bf16 ETA: {bf16_hours:.0f}h ({bf16_hours/24:.1f} days)")
print(f"  FP8 ETA:  {fp8_hours:.0f}h ({fp8_hours/24:.1f} days)")
print(f"  Saved:    {(bf16_hours - fp8_hours):.0f}h ({(bf16_hours - fp8_hours)/24:.1f} days)")
