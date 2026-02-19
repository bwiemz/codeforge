"""Training entry point for CodeForge.

Supports two modes:
  --smoke-test: Quick test with synthetic data (no tokenizer/data needed)
  --config configs/pretrain_150m.yaml: Full training with data pipeline
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from codeforge.data.dataset import (
    EvalCodeDataset,
    MixedCodeDataset,
    SyntheticCodeDataset,
)
from codeforge.data.pretokenized import PreTokenizedDataset
from codeforge.model.config import ModelConfig, get_preset
from codeforge.model.transformer import CodeForgeModel
from codeforge.tokenizer.tokenizer import CodeForgeTokenizer
from codeforge.training.trainer import Trainer, TrainingConfig


def main():
    parser = argparse.ArgumentParser(description="Train CodeForge model")
    parser.add_argument(
        "--config", "-c", type=str, default=None,
        help="Path to training config YAML (e.g. configs/pretrain_150m.yaml)",
    )
    parser.add_argument(
        "--preset", type=str, default="150m",
        choices=["150m", "350m", "1b", "3b"],
        help="Model size preset (default: 150m)",
    )
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="Path to trained tokenizer JSON")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Quick test with synthetic data")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--hf-token", type=str, default=None,
                        help="HuggingFace token for gated datasets")
    parser.add_argument("--pretokenized", type=str, default=None,
                        help="Path to pre-tokenized data dir (overrides data.pretokenized_path)")
    args = parser.parse_args()

    # HuggingFace login if token provided
    if args.hf_token:
        from huggingface_hub import login
        login(token=args.hf_token)
        print("Logged in to HuggingFace Hub")

    # ---- Load config ----
    raw_config = {}
    if args.config:
        with open(args.config) as f:
            raw_config = yaml.safe_load(f)

    # Model config
    model_config_path = raw_config.get("model")
    if model_config_path:
        model_config = ModelConfig.from_yaml(model_config_path)
    else:
        model_config = get_preset(args.preset)

    # Training config
    train_kwargs = raw_config.get("training", {})
    train_config = TrainingConfig(**train_kwargs)

    # CLI overrides
    if args.batch_size:
        train_config.batch_size = args.batch_size
    if args.lr:
        train_config.learning_rate = args.lr
    if args.max_steps:
        train_config.max_steps = args.max_steps
    if args.resume:
        train_config.resume_from = args.resume

    # Auto-resume: if no explicit --resume and latest.pt exists in checkpoint_dir, use it.
    # This makes restarts seamless — re-run the same command after any crash/stop.
    if not train_config.resume_from:
        auto_resume = Path(train_config.checkpoint_dir) / "latest.pt"
        if auto_resume.exists():
            train_config.resume_from = str(auto_resume)
            print(f"  Auto-resuming from {auto_resume}")

    # ---- Data config ----
    data_config = raw_config.get("data", {})

    # Enable gradient checkpointing for real training
    if not args.smoke_test:
        model_config.use_gradient_checkpointing = data_config.get(
            "gradient_checkpointing", True
        )

    # ---- Create model ----
    model = CodeForgeModel(model_config)

    # ---- Create dataset ----
    dataset = None

    eval_dataset = None

    if args.smoke_test:
        print("\nMode: SMOKE TEST (synthetic data)")
        # Set smoke test defaults, but respect CLI overrides
        if not args.max_steps:
            train_config.max_steps = 20
        train_config.batch_size = 2
        train_config.gradient_accumulation_steps = 2
        train_config.log_every = 5
        train_config.eval_every = 10
        train_config.checkpoint_every = 10
        # WSD schedule: 5 warmup + 10 stable + 5 decay = 20 steps
        train_config.scheduler_type = "wsd"
        train_config.warmup_steps = 5
        train_config.stable_steps = 10
        train_config.decay_steps = 5
        dataset = SyntheticCodeDataset(
            vocab_size=model_config.vocab_size,
            seq_len=model_config.max_seq_len,
            num_samples=500,
        )
        eval_dataset = SyntheticCodeDataset(
            vocab_size=model_config.vocab_size,
            seq_len=model_config.max_seq_len,
            num_samples=50,
        )
    else:
        # Real training with full data pipeline
        tokenizer_path = args.tokenizer or raw_config.get("tokenizer_path")
        if not tokenizer_path:
            print("\nError: Provide --tokenizer or set tokenizer_path in config YAML")
            sys.exit(1)

        tokenizer_path = Path(tokenizer_path)
        if not tokenizer_path.exists():
            print(f"\nError: Tokenizer not found at {tokenizer_path}")
            print("Train one first: "
                  "python scripts/train_tokenizer.py --from-hf bigcode/starcoderdata ...")
            sys.exit(1)

        tokenizer = CodeForgeTokenizer(str(tokenizer_path))

        # Sync vocab size from tokenizer
        if model_config.vocab_size != tokenizer.vocab_size:
            print(f"\n  Updating vocab_size: {model_config.vocab_size} -> {tokenizer.vocab_size}")
            model_config.vocab_size = tokenizer.vocab_size
            model = CodeForgeModel(model_config)

        hf_dataset = data_config.get("hf_dataset", "bigcode/starcoderdata")
        languages = data_config.get("languages", [
            "python", "javascript", "typescript", "java", "cpp", "go", "rust",
        ])
        language_weights = data_config.get("language_weights")

        # Check for pre-tokenized data (CLI flag or config)
        pretokenized_path = args.pretokenized or data_config.get("pretokenized_path")

        if pretokenized_path:
            pt_dir = Path(pretokenized_path)
            if not pt_dir.exists():
                print(f"\nError: Pre-tokenized data dir not found: {pretokenized_path}")
                print("Run scripts/pretokenize.py first to generate it.")
                sys.exit(1)
            dataset = PreTokenizedDataset(
                data_dir=pt_dir,
                max_seq_len=model_config.max_seq_len,
                eos_id=tokenizer.EOS_ID,
                shuffle=True,
                seed=data_config.get("seed", 42),
            )
            print("\nData pipeline: PRE-TOKENIZED")
            print(f"  Source: {pretokenized_path}")
            print(f"  Samples: {dataset.num_samples:,}")
            print(f"  Tokens:  {dataset.total_tokens:,}")
        else:
            dataset = MixedCodeDataset(
                tokenizer=tokenizer,
                max_seq_len=model_config.max_seq_len,
                hf_dataset=hf_dataset,
                languages=languages,
                language_weights=language_weights,
                quality_threshold=data_config.get("quality_threshold", 0.3),
                enable_dedup=data_config.get("enable_dedup", True),
                dedup_threshold=data_config.get("dedup_threshold", 0.8),
                enable_decontamination=data_config.get("enable_decontamination", True),
                fim_rate=data_config.get("fim_rate", 0.5),
                seed=data_config.get("seed", 42),
            )
            print("\nData pipeline:")
            print(f"  Dataset: {hf_dataset}")
            print(f"  Languages: {', '.join(languages)}")
            print(f"  Quality threshold: {data_config.get('quality_threshold', 0.3)}")
            dedup_thresh = data_config.get('dedup_threshold', 0.8)
            print(f"  Dedup: {data_config.get('enable_dedup', True)} "
                  f"(threshold: {dedup_thresh})")
            print(f"  Decontamination: "
                  f"{data_config.get('enable_decontamination', True)}")
            print(f"  FIM rate: {data_config.get('fim_rate', 0.5)}")

        # Create eval dataset (held-out samples from the same source)
        eval_dataset = EvalCodeDataset(
            tokenizer=tokenizer,
            max_seq_len=model_config.max_seq_len,
            hf_dataset=hf_dataset,
            languages=languages,
            quality_threshold=data_config.get('quality_threshold', 0.3),
            fim_rate=0.0,  # No FIM for eval — measure raw language modeling
            num_eval_samples=200,
            skip_samples=50_000,  # Skip first 50k samples per language to avoid train overlap
            seed=99999,
        )
        print(f"  Eval dataset: {200} held-out samples (skip first 50k per lang)")

    # ---- Print summary ----
    param_count = model.param_count()
    print(f"\n{'='*60}")
    print("CodeForge Training")
    print(f"{'='*60}")
    print(f"Model: {args.preset} ({param_count:,} parameters)")
    print(f"  dim={model_config.dim}, layers={model_config.n_layers}, "
          f"heads={model_config.n_heads}, kv_heads={model_config.n_kv_heads}")
    print(f"  grad_checkpointing={model_config.use_gradient_checkpointing}")
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        vram = getattr(props, 'total_memory', getattr(props, 'total_mem', 0)) / (1024**3)
        print(f"GPU: {torch.cuda.get_device_name(0)} ({vram:.1f} GB)")

    eff_batch = train_config.effective_batch_size
    tokens_per_step = eff_batch * model_config.max_seq_len
    total_tokens = tokens_per_step * train_config.max_steps
    print("\nTraining:")
    bs, ga = train_config.batch_size, train_config.gradient_accumulation_steps
    print(f"  Batch: {bs} x {ga} accum = {eff_batch} effective")
    print(
        f"  Tokens/step: {tokens_per_step:,} | "
        f"Total: ~{total_tokens/1e9:.1f}B tokens over {train_config.max_steps:,} steps"
    )
    print(
        f"  LR: {train_config.learning_rate:.1e}, "
        f"warmup: {train_config.warmup_steps}, precision: {train_config.precision}"
    )
    print(f"{'='*60}\n")

    # ---- Train ----
    trainer = Trainer(model, dataset, train_config, eval_dataset=eval_dataset)
    trainer.train()

    # Smoke test: checkpoint round-trip validation
    if args.smoke_test:
        print(f"\n{'='*60}")
        print("Smoke test: checkpoint round-trip validation")
        print(f"{'='*60}")
        ckpt_path = Path(train_config.checkpoint_dir) / "latest.pt"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            # Verify required keys
            required = [
                "global_step", "tokens_seen", "model_state_dict",
                "optimizer_state_dict", "scheduler_state_dict", "model_config",
            ]
            missing = [k for k in required if k not in ckpt]
            if missing:
                print(f"  FAIL: Missing checkpoint keys: {missing}")
            else:
                print(f"  Checkpoint keys: PASS ({len(ckpt)} keys)")

            # Verify model loads cleanly
            model2 = CodeForgeModel(model_config)
            model2.load_state_dict(ckpt["model_state_dict"])
            print("  Model reload: PASS")

            # Verify Reclaimer config preserved
            saved = ckpt["model_config"]
            checks = {
                "post_norm": saved.get("use_post_norm"),
                "qk_norm": saved.get("use_qk_norm"),
                "z_loss": saved.get("z_loss_alpha", 0) > 0,
            }
            print(f"  Reclaimer config: {checks}")

            # Verify training_config saved
            if "training_config" in ckpt:
                tc = ckpt["training_config"]
                print(
                    f"  Training config: scheduler={tc.get('scheduler_type')}, "
                    f"embed_lr_ratio={tc.get('embed_lr_ratio')}"
                )
            else:
                print("  Training config: NOT SAVED (old format)")

            # Verify optimizer has 3 param groups
            n_groups = len(trainer.optimizer.param_groups)
            print(f"  Optimizer groups: {n_groups} (expected 3)")

            # Final LR from scheduler
            final_lr = trainer.scheduler.get_last_lr()[0]
            print(f"  Final LR: {final_lr:.6f}")
            print(f"  Step: {ckpt['global_step']}, Tokens: {ckpt['tokens_seen']:,}")
        else:
            print(f"  SKIP: No checkpoint at {ckpt_path}")
        print(f"{'='*60}\n")

    # Print data pipeline stats
    if hasattr(dataset, 'print_stats'):
        dataset.print_stats()


if __name__ == "__main__":
    main()
