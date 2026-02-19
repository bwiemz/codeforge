"""Training loop for CodeForge.

Zero-Sync Fused Training Pipeline (ZSFTP):
  - torch.compile with inductor backend for kernel fusion
  - Fused AdamW optimizer (entire update on GPU, no CPU-GPU sync)
  - TF32 matmul precision for faster float32 ops on Blackwell/Ampere+
  - FP8 linear layers via torchao (doubles matmul throughput on Blackwell)
  - Non-blocking CUDA transfers with prefetch overlap
  - GPU-resident loss accumulation (no .item() per micro-step)
  - Real-valued RoPE for full torch.compile compatibility
  - Chunked cross-entropy to reduce peak memory by ~75%
  - TCFP (Tensor Core Floating Point): custom FP8 format with 19% speedup vs BF16
"""

import json
import logging
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

from ..model.transformer import CodeForgeModel
from .scheduler import get_cosine_schedule_with_warmup


@dataclass
class TrainingConfig:
    # Optimization
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    min_lr_ratio: float = 0.1

    # Schedule
    warmup_steps: int = 1000
    max_steps: int = 100000

    # Precision
    precision: str = "bf16"  # "bf16", "fp16", or "fp32"

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    checkpoint_every: int = 5000
    keep_last_n_checkpoints: int = 5
    resume_from: str | None = None

    # Logging
    log_every: int = 10
    eval_every: int = 1000
    use_wandb: bool = False
    wandb_project: str = "codeforge"

    # DataLoader
    num_workers: int = 0  # 0 for Windows compatibility

    # ZSFTP optimizations
    use_compile: bool = True
    use_fused_optimizer: bool = True
    use_fp8: bool = False  # FP8 adds overhead at 150M scale (5.9x slower than BF16)

    # TCFP — Tensor Core Floating Point (custom FP8 format, 19% faster than BF16)
    use_tcfp: bool = False
    tcfp_warmup_steps: int = 0          # Full-precision warmup before quantizing
    tcfp_use_tensor_cores: bool = True  # Real FP8 tensor-core GEMMs
    tcfp_delayed_scaling: bool = True   # EMA-smoothed W_hi amax (stability)
    tcfp_use_fused_kernel: bool = True  # Triton fused dual-GEMM
    tcfp_use_cuda_ext: bool = True      # cuBLASLt C++ extension when available
    tcfp_abd: bool = False              # Asymmetric Backward Decomposition
    tcfp_srr: bool = False              # Stochastic Residual Rounding
    tcfp_monitor_interval: int = 100    # Steps between EF buffer health checks
    tcfp_disable_compile: bool = True   # Disable torch.compile (TCFP Triton already JIT)

    # Reclaimer Protocol
    embed_lr_ratio: float = 0.1  # Embedding LR = learning_rate * this ratio

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps


def _setup_torch_optimizations() -> None:
    """Configure global PyTorch settings for maximum GPU throughput."""
    # TF32: Use TensorFloat-32 for float32 matmuls (3x faster on Ampere+/Blackwell).
    # This affects the float32 ops in RMSNorm and anywhere autocast doesn't reach.
    # Precision: 10-bit mantissa (vs 23-bit fp32), which is more than sufficient
    # for training since we're already using bf16 (7-bit mantissa) for most ops.
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Inductor cache: use short path to avoid Windows MAX_PATH (260 char) limit.
    # Triton generates kernel filenames that can exceed this on Windows.
    cache_dir = r"C:\tc"
    os.makedirs(cache_dir, exist_ok=True)
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", cache_dir)
    os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(cache_dir, "triton"))


class Trainer:
    """Handles the training loop with ZSFTP optimizations.

    Zero-Sync Fused Training Pipeline eliminates CPU-GPU synchronization from
    the hot path by:
    1. Accumulating loss on GPU (torch.Tensor, not float)
    2. Using fused AdamW (entire optimizer step on GPU)
    3. Non-blocking .to(device) transfers
    4. Only calling .item() at logging boundaries
    """

    def __init__(
        self,
        model: CodeForgeModel,
        train_dataset: IterableDataset,
        config: TrainingConfig,
        eval_dataset: IterableDataset | None = None,
    ):
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.config = config

        # Debug log — appends to checkpoints/training_debug.log
        log_path = Path(config.checkpoint_dir).parent / "training_debug.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = logging.getLogger("codeforge.trainer")
        self._log.setLevel(logging.DEBUG)
        if not self._log.handlers:
            fh = logging.FileHandler(str(log_path), encoding="utf-8")
            fh.setFormatter(logging.Formatter(
                "%(asctime)s | %(levelname)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            ))
            self._log.addHandler(fh)
        self._log.info("=== Trainer init started ===")

        # Apply global optimizations before anything touches CUDA
        _setup_torch_optimizations()

        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        # Pre-compile raw module reference — typed as nn.Module to avoid the union type
        # that torch.compile introduces on self.model. Both point to the same object.
        self._raw_model_ref: nn.Module = model  # type: ignore[assignment]

        # TCFP handles — initialized here so basedpyright sees them as declared instance
        # attributes. _setup_tcfp() overwrites these with real values when use_tcfp=True.
        # IMPORTANT: must be set BEFORE _setup_tcfp() is called below.
        self._tcfp_monitor: Any = None
        self._tcfp_bench_log: IO[str] | None = None

        # TCFP: Must load model weights BEFORE conversion so TCFPLinear copies
        # the trained BF16 weights (not random init) when replacing nn.Linear.
        if config.use_tcfp and config.resume_from and self.device.type == "cuda":
            self._load_model_weights_only(config.resume_from)

        # TCFP: Convert linear layers to FP8 dual-GEMM format.
        # Applied before torch.compile so compile captures the TCFP forward/backward.
        # Skips output/tok_embeddings (weight-tied pair).
        if config.use_tcfp and self.device.type == "cuda":
            self._setup_tcfp()
        elif config.use_fp8 and self.device.type == "cuda":
            # Existing torchao FP8 path (mutually exclusive with TCFP)
            try:
                from torchao.float8 import Float8LinearConfig, convert_to_float8_training
                fp8_config = Float8LinearConfig()
                convert_to_float8_training(self.model, config=fp8_config)
                fp8_count = sum(1 for m in self.model.modules() if "Float8" in type(m).__name__)
                print(f"  FP8: Converted {fp8_count} linear layers to Float8Linear")
            except ImportError:
                print("  FP8: torchao not installed, skipping (pip install torchao)")
            except Exception as e:
                print(f"  FP8: Conversion failed ({e}), continuing with bf16")

        # Compile model for kernel fusion (disabled for TCFP — Triton kernels already JIT)
        compile_enabled = (
            config.use_compile
            and self.device.type == "cuda"
            and not (config.use_tcfp and config.tcfp_disable_compile)
        )
        if compile_enabled:
            print("Compiling model with torch.compile (inductor)...")
            self.model = torch.compile(self.model, backend="inductor")
            print("  Model compiled (kernels will be generated on first forward pass)")
        elif config.use_tcfp and config.tcfp_disable_compile:
            print("  torch.compile disabled (TCFP Triton kernels are already JIT-compiled)")

        # Optimizer: Fused AdamW runs entirely on GPU — no CPU-GPU sync per step
        # 3 param groups: (1) decay, (2) no-decay, (3) embedding with separate LR
        embed_param = model.tok_embeddings.weight  # Same tensor as output.weight (tied)
        embed_ptr = embed_param.data_ptr()

        no_decay = {"bias", "norm", "RMSNorm"}
        decay_params = []
        no_decay_params = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if p.data_ptr() == embed_ptr:
                continue  # Embedding goes in its own group
            if any(nd in name for nd in no_decay):
                no_decay_params.append(p)
            else:
                decay_params.append(p)

        embed_lr = config.learning_rate * config.embed_lr_ratio

        param_groups = [
            {"params": decay_params, "weight_decay": config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
            {"params": [embed_param], "lr": embed_lr, "weight_decay": 0.0},
        ]
        use_fused = config.use_fused_optimizer and self.device.type == "cuda"
        self.optimizer = torch.optim.AdamW(
            param_groups,
            lr=config.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            fused=use_fused,
        )
        if use_fused:
            print("  Using fused AdamW (GPU-resident optimizer)")

        # LR scheduler
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            config.warmup_steps,
            config.max_steps,
            config.min_lr_ratio,
        )

        # Mixed precision — keep enabled even for TCFP (autocast benefits norms/embeddings)
        self.use_amp = config.precision != "fp32" and self.device.type == "cuda"
        self.amp_dtype = torch.bfloat16 if config.precision == "bf16" else torch.float16
        self.scaler = torch.amp.GradScaler("cuda", enabled=(config.precision == "fp16"))

        # Tracking
        self.global_step = 0
        self.tokens_seen = 0
        self.best_eval_loss = float("inf")

        # WandB
        self.wandb_run = None
        if config.use_wandb:
            try:
                import wandb  # pyright: ignore[reportMissingImports]
                self.wandb_run = wandb.init(
                    project=config.wandb_project,
                    config={
                        "model": model.config.__dict__,
                        "training": config.__dict__,
                    },
                )
            except ImportError:
                print("Warning: wandb not installed, skipping experiment tracking")

        # Resume from checkpoint — loads optimizer/scheduler/step/EF state.
        # Model weights were already loaded in _load_model_weights_only() above
        # (if use_tcfp), so this call only restores training state.
        if config.resume_from:
            self._load_checkpoint(config.resume_from)

        self._log.info(
            "Trainer init done: step=%d, tcfp=%s, monitor=%s, device=%s",
            self.global_step,
            config.use_tcfp,
            self._tcfp_monitor is not None,
            self.device,
        )

    def __del__(self) -> None:
        """Ensure benchmark log is flushed/closed even on unexpected exit."""
        bench_log = getattr(self, "_tcfp_bench_log", None)
        if bench_log is not None and not bench_log.closed:
            bench_log.close()

    # ─── TCFP Setup ──────────────────────────────────────────────────────────

    def _setup_tcfp(self) -> None:
        """Convert model linear layers to TCFP-12 FP8 format and set up monitoring."""
        try:
            from ..tcfp.core import TCFPMode
            from ..tcfp.nn import convert_to_tcfp
            from ..tcfp.training.monitoring import TCFPMonitor
        except ImportError as e:
            print(f"  TCFP: Import failed ({e})")
            return

        raw_model = self._raw_model_ref

        convert_to_tcfp(
            raw_model,
            mode=TCFPMode.TCFP12,
            use_tensor_cores=self.config.tcfp_use_tensor_cores,
            error_feedback=True,
            delayed_scaling=self.config.tcfp_delayed_scaling,
            use_fused_kernel=self.config.tcfp_use_fused_kernel,
            use_cuda_ext=self.config.tcfp_use_cuda_ext,
            abd=self.config.tcfp_abd,
            srr=self.config.tcfp_srr,
            warmup_steps=self.config.tcfp_warmup_steps,
            skip_patterns=("output", "tok_embeddings"),
        )

        converted = sum(
            1 for m in raw_model.modules() if type(m).__name__ == "TCFPLinear"
        )
        print(f"  TCFP-12: Converted {converted} linear layers "
              f"(output + tok_embeddings kept FP32, weight-tied)")

        self._tcfp_monitor = TCFPMonitor(
            grad_norm_threshold=self.config.max_grad_norm * 10.0,
            ef_buffer_threshold=1.0,
            l2_interval=10,
            l3_interval=self.config.tcfp_monitor_interval,
        )

        # Benchmark JSONL log — one line per log_every steps
        bench_path = Path(self.config.checkpoint_dir).parent / "tcfp_benchmark.jsonl"
        bench_path.parent.mkdir(parents=True, exist_ok=True)
        self._tcfp_bench_log = open(bench_path, "a", buffering=1, encoding="utf-8")  # noqa: SIM115
        print(f"  TCFP benchmark log: {bench_path}")

    def _load_model_weights_only(self, path: str) -> None:
        """Load only model weights from a checkpoint (before TCFP conversion).

        This populates the nn.Linear weights from the BF16 checkpoint so that
        convert_to_tcfp() copies trained weights (not random init) into TCFPLinear.
        Optimizer/scheduler/step state is loaded later by _load_checkpoint().
        """
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self._raw_model_ref.load_state_dict(ckpt["model_state_dict"])
        print(f"  TCFP pre-load: model weights loaded from step {ckpt.get('global_step', '?')}")

    # ─── TCFP EF State Serialization ─────────────────────────────────────────

    def _collect_tcfp_ef_state(self) -> dict[str, Any]:
        """Extract error feedback state from all TCFPLinear modules."""
        raw_model = self._raw_model_ref

        ef_state: dict[str, Any] = {}
        for name, module in raw_model.named_modules():
            if type(module).__name__ != "TCFPLinear":
                continue
            ef = getattr(module, "_error_state", None)
            if ef is None:
                continue
            # Serialize ErrorFeedbackState to CPU tensors
            ef_state[name] = {
                "_buffers": {k: v.cpu() for k, v in ef._buffers.items()},
                "_amax_ema": {k: v.cpu() for k, v in ef._amax_ema.items()},
                "_delayed_amax": {k: v.cpu() for k, v in ef._delayed_amax.items()},
                "_residual_ratio": {k: v.cpu() for k, v in ef._residual_ratio.items()},
                "_step_count": dict(ef._step_count),
                "_ema_decay": ef._ema_decay,
            }
        return ef_state

    def _restore_tcfp_ef_state(self, ef_state: dict[str, Any]) -> None:
        """Restore error feedback state into TCFPLinear modules from checkpoint."""
        raw_model = self._raw_model_ref

        restored = 0
        for name, module in raw_model.named_modules():
            if type(module).__name__ != "TCFPLinear":
                continue
            if name not in ef_state:
                continue
            ef = getattr(module, "_error_state", None)
            if ef is None:
                continue
            saved = ef_state[name]
            ef._buffers = {k: v.to(self.device) for k, v in saved["_buffers"].items()}
            ef._amax_ema = {k: v.to(self.device) for k, v in saved["_amax_ema"].items()}
            ef._delayed_amax = {k: v.to(self.device) for k, v in saved["_delayed_amax"].items()}
            ef._residual_ratio = {k: v.to(self.device) for k, v in saved["_residual_ratio"].items()}
            ef._step_count = dict(saved["_step_count"])
            ef._ema_decay = saved["_ema_decay"]
            restored += 1
        print(f"  Restored TCFP EF state for {restored}/{len(ef_state)} layers")

    def _collect_ef_stats(self, raw_model: nn.Module) -> dict[str, Any]:
        """Compute EF buffer mean |x| statistics across all TCFPLinear layers."""
        per_layer: dict[str, float] = {}
        for name, module in raw_model.named_modules():
            if type(module).__name__ != "TCFPLinear":
                continue
            ef = getattr(module, "_error_state", None)
            if ef is None:
                continue
            buf = ef._buffers.get(getattr(module, "_param_name", "weight"))
            if buf is None:
                continue
            per_layer[name] = float(buf.float().abs().mean().item())

        if not per_layer:
            return {"max": 0.0, "avg": 0.0, "per_layer": {}}
        vals = list(per_layer.values())
        return {
            "max": max(vals),
            "avg": sum(vals) / len(vals),
            "per_layer": per_layer,
        }

    # ─── Checkpointing ───────────────────────────────────────────────────────

    def _save_checkpoint(self, path: str | None = None) -> None:
        """Save model, optimizer, scheduler, RNG state, and TCFP EF state."""
        ckpt_dir = Path(path or self.config.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        raw_model = self._raw_model_ref

        ckpt_path = ckpt_dir / f"step_{self.global_step}.pt"
        save_dict: dict[str, Any] = {
            "global_step": self.global_step,
            "tokens_seen": self.tokens_seen,
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "model_config": raw_model.config.__dict__,
            "best_eval_loss": self.best_eval_loss,
            "torch_rng_state": torch.random.get_rng_state(),
            "precision": self.config.precision,
        }
        if torch.cuda.is_available():
            save_dict["cuda_rng_state"] = torch.cuda.get_rng_state()

        # Save TCFP error feedback state for clean resume
        if self.config.use_tcfp and self._tcfp_monitor is not None:
            save_dict["tcfp_ef_state"] = self._collect_tcfp_ef_state()

        torch.save(save_dict, ckpt_path)
        print(f"Checkpoint saved: {ckpt_path}")

        # Copy to latest.pt for easy resume
        latest_path = ckpt_dir / "latest.pt"
        shutil.copy2(ckpt_path, latest_path)

        # Cleanup old checkpoints (keep last N)
        if self.config.keep_last_n_checkpoints > 0 and path is None:
            all_ckpts = sorted(
                ckpt_dir.glob("step_*.pt"),
                key=lambda p: int(p.stem.split("_")[1]),
            )
            while len(all_ckpts) > self.config.keep_last_n_checkpoints:
                old = all_ckpts.pop(0)
                old.unlink()
                print(f"  Removed old checkpoint: {old.name}")

    def _load_checkpoint(self, path: str) -> None:
        """Load optimizer, scheduler, training state, and TCFP EF state from checkpoint.

        When use_tcfp=True, the model weights were already loaded by
        _load_model_weights_only() before TCFP conversion. This call only
        restores the optimizer/scheduler/step/EF state.
        """
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        raw_model = self._raw_model_ref

        if self.config.use_tcfp:
            # Model weights already loaded (with strict=True) by _load_model_weights_only()
            # before TCFP conversion. Skip the second load — it would attempt to overwrite
            # TCFPLinear's internal buffers with the BF16 checkpoint which doesn't have them,
            # and strict=False would silently ignore genuinely missing weight keys.
            pass
        else:
            raw_model.load_state_dict(ckpt["model_state_dict"])

        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        self.global_step = ckpt["global_step"]
        self.tokens_seen = ckpt["tokens_seen"]
        self.best_eval_loss = ckpt.get("best_eval_loss", float("inf"))

        # Restore RNG state for reproducibility
        if "torch_rng_state" in ckpt:
            rng_state = ckpt["torch_rng_state"].cpu().byte()
            torch.random.set_rng_state(rng_state)
        if ckpt.get("cuda_rng_state") is not None and torch.cuda.is_available():
            cuda_state = ckpt["cuda_rng_state"].cpu().byte()
            torch.cuda.set_rng_state(cuda_state)

        # Restore TCFP EF state (from TCFP checkpoints; absent in BF16 checkpoints)
        if self.config.use_tcfp and "tcfp_ef_state" in ckpt and self._tcfp_monitor is not None:
            self._restore_tcfp_ef_state(ckpt["tcfp_ef_state"])

        print(f"Resumed from step {self.global_step} ({self.tokens_seen:,} tokens)")

    # ─── Evaluation ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(self, max_batches: int = 50) -> float:
        """Run evaluation and return average loss."""
        if self.eval_dataset is None:
            return float("inf")

        self.model.eval()  # type: ignore[union-attr]
        total_loss = 0.0
        count = 0

        loader = DataLoader(self.eval_dataset, batch_size=self.config.batch_size)
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            targets = batch["targets"].to(self.device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                _, loss, _ = self.model(input_ids, targets)

            total_loss += loss.item()
            count += 1

        self.model.train()  # type: ignore[union-attr]
        return total_loss / max(count, 1)

    # ─── Training Loop ───────────────────────────────────────────────────────

    def train(self) -> None:
        """Main training loop with Zero-Sync Fused Training Pipeline.

        Key optimization: loss is accumulated as a GPU tensor, not a Python float.
        The old code called loss.item() on every micro-step (32x per optimizer step),
        each call forcing a full CUDA synchronization (~50-200us of GPU idle time).
        Over 85K remaining steps * 32 micro-steps * ~100us = ~4.5 minutes of pure
        sync overhead eliminated.
        """
        try:
            self._train_inner()
        except Exception:
            self._log.exception(
                "TRAINING CRASHED at step %d (tokens=%d)",
                self.global_step, self.tokens_seen,
            )
            raise

    def _train_inner(self) -> None:
        """Inner training loop (wrapped by train() for exception logging)."""
        self.model.train()  # type: ignore[union-attr]
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=True,
            prefetch_factor=2 if self.config.num_workers > 0 else None,
        )

        # Raw model reference for TCFP monitoring (compile wrapper-safe)
        _raw_model = self._raw_model_ref

        # GPU-resident loss accumulator — no .item() until logging time
        accumulation_loss = torch.tensor(0.0, device=self.device)
        step_start_time = time.time()

        # Step wall-clock timer for benchmark recording
        _step_wall_start = time.perf_counter()
        _step_wall_ms: float = 0.0

        pbar = tqdm(total=self.config.max_steps, initial=self.global_step, desc="Training")
        self._log.info(
            "Training loop started at step %d (tokens_seen=%d)",
            self.global_step, self.tokens_seen,
        )
        _data_wait_start = time.perf_counter()

        for micro_step, batch in enumerate(loader):
            # Log data loading stalls (>60s waiting for a batch)
            _data_wait_s = time.perf_counter() - _data_wait_start
            if _data_wait_s > 60:
                self._log.warning(
                    "Data stall: waited %.1fs for batch (micro_step=%d, global_step=%d)",
                    _data_wait_s, micro_step, self.global_step,
                )

            if self.global_step >= self.config.max_steps:
                break

            # Non-blocking transfer: CPU->GPU copy overlaps with previous backward
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            targets = batch["targets"].to(self.device, non_blocking=True)

            # Forward pass with mixed precision
            try:
                with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                    _, loss, _ = self.model(input_ids, targets)
                    loss = loss / self.config.gradient_accumulation_steps
            except Exception:
                self._log.exception(
                    "Forward pass FAILED at micro_step=%d, global_step=%d",
                    micro_step, self.global_step,
                )
                raise

            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()

            # Accumulate loss on GPU — NO .item() here (avoids CUDA sync)
            accumulation_loss += loss.detach()

            # Track tokens
            self.tokens_seen += input_ids.numel()

            # Optimizer step after accumulation
            if (micro_step + 1) % self.config.gradient_accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self._raw_model_ref.parameters(), self.config.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()
                self.global_step += 1

                # Capture step wall time (covers one full optimizer step's worth of microsteps)
                _step_wall_ms = (time.perf_counter() - _step_wall_start) * 1000
                _step_wall_start = time.perf_counter()

                # Logging — only sync here (once per log_every steps)
                if self.global_step % self.config.log_every == 0:
                    # Single .item() call for this step's loss (accumulator resets every step)
                    loss_val = accumulation_loss.item()
                    elapsed = time.time() - step_start_time
                    tokens_per_sec = (
                        self.config.effective_batch_size
                        * self._raw_model_ref.config.max_seq_len  # type: ignore[attr-defined]
                        * self.config.log_every
                        / elapsed
                    )
                    lr = self.scheduler.get_last_lr()[0]
                    gn = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
                    # step_ms averaged over the log window — reuse elapsed (wall time for window)
                    step_ms_avg = elapsed * 1000.0 / self.config.log_every

                    log_data: dict[str, Any] = {
                        "loss": loss_val,
                        "lr": lr,
                        "grad_norm": gn,
                        "tokens_per_sec": tokens_per_sec,
                        "tokens_seen": self.tokens_seen,
                        "step_ms": step_ms_avg,
                    }

                    # TCFP health monitoring and benchmark recording
                    if self.config.use_tcfp and self._tcfp_monitor is not None:
                        # Enable kurtosis capture at L3 intervals
                        if self.global_step % self._tcfp_monitor.l3_interval == 0:
                            for layer in _raw_model.layers:
                                layer._capture_kurtosis = True

                        alerts = self._tcfp_monitor.check(
                            model=_raw_model,
                            loss=loss_val,
                            step=self.global_step,
                        )
                        ef_stats = self._collect_ef_stats(_raw_model)
                        log_data["tcfp_ef_max"] = ef_stats["max"]
                        log_data["tcfp_ef_avg"] = ef_stats["avg"]
                        log_data["tcfp_alerts"] = len(alerts)

                        # Write benchmark record (line-buffered, flushes immediately)
                        bench_record = {
                            "step": self.global_step,
                            "loss": loss_val,
                            "step_ms": step_ms_avg,
                            "tokens_per_sec": tokens_per_sec,
                            "grad_norm": gn,
                            "ef_mean_abs_max": ef_stats["max"],
                            "ef_mean_abs_avg": ef_stats["avg"],
                            "ef_layer_stats": ef_stats["per_layer"],
                            "alerts": [
                                {"level": a.level, "layer": a.layer, "reason": a.reason}
                                for a in alerts
                            ],
                        }
                        if self._tcfp_bench_log is not None:
                            self._tcfp_bench_log.write(json.dumps(bench_record) + "\n")
                            self._tcfp_bench_log.flush()

                        # Surface critical alerts immediately
                        for alert in alerts:
                            if alert.level == "CRITICAL":
                                print(
                                    f"\n[TCFP CRITICAL] Step {self.global_step} "
                                    f"| {alert.layer}: {alert.reason}"
                                )

                    pbar.set_postfix(
                        loss=f"{loss_val:.4f}",
                        lr=f"{lr:.2e}",
                        gn=f"{gn:.2f}",
                        tps=f"{tokens_per_sec:.0f}",
                        ms=f"{step_ms_avg:.1f}",
                    )
                    pbar.update(self.config.log_every)

                    if self.wandb_run:
                        import wandb  # pyright: ignore[reportMissingImports]
                        wandb.log(log_data, step=self.global_step)

                    step_start_time = time.time()

                # Reset GPU loss accumulator
                accumulation_loss.zero_()

                # Periodic debug heartbeat (every 50 steps)
                if self.global_step % 50 == 0:
                    self._log.info(
                        "Heartbeat step=%d tokens=%d",
                        self.global_step, self.tokens_seen,
                    )

                # Evaluation
                if self.global_step % self.config.eval_every == 0:
                    self._log.info("Eval starting at step %d", self.global_step)
                    eval_loss = self.evaluate()
                    self._log.info("Eval done: loss=%.4f", eval_loss)
                    print(f"\nStep {self.global_step} | Eval loss: {eval_loss:.4f}")
                    if eval_loss < self.best_eval_loss:
                        self.best_eval_loss = eval_loss
                        self._save_checkpoint(
                            str(Path(self.config.checkpoint_dir) / "best")
                        )
                    if self.wandb_run:
                        import wandb  # pyright: ignore[reportMissingImports]
                        wandb.log({"eval_loss": eval_loss}, step=self.global_step)

                # Checkpointing
                if self.global_step % self.config.checkpoint_every == 0:
                    self._log.info("Checkpoint saving at step %d", self.global_step)
                    self._save_checkpoint()
                    self._log.info("Checkpoint saved")

            # Reset data wait timer for next batch
            _data_wait_start = time.perf_counter()

        pbar.close()
        self._log.info("Training loop exited normally at step %d", self.global_step)
        self._save_checkpoint()
        if self._tcfp_bench_log is not None and not self._tcfp_bench_log.closed:
            self._tcfp_bench_log.close()
        print(
            f"\nTraining complete. {self.global_step} steps, "
            f"{self.tokens_seen:,} tokens processed."
        )
