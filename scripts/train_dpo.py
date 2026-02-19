"""DPO training entry point."""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from codeforge.tokenizer.tokenizer import CodeForgeTokenizer

from codeforge.model.config import ModelConfig
from codeforge.model.transformer import CodeForgeModel
from codeforge.training.dpo_trainer import DPODataset, DPOTrainer


def main():
    parser = argparse.ArgumentParser(description="DPO training for CodeForge")
    parser.add_argument("--checkpoint", "-c", required=True, help="SFT model checkpoint")
    parser.add_argument("--tokenizer", "-t", required=True, help="Tokenizer path")
    parser.add_argument("--config", type=str, default=None, help="YAML config file")
    parser.add_argument("--data", "-d", nargs="+", default=None, help="DPO data JSONL files")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta parameter")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--precision", default=None, choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--embed-lr-ratio", type=float, default=None)
    parser.add_argument("--scheduler-type", default=None, choices=["cosine", "wsd"])
    parser.add_argument("--stable-steps", type=int, default=None)
    parser.add_argument("--decay-steps", type=int, default=None)
    parser.add_argument("--decay-type", default=None, choices=["cosine", "linear"])
    args = parser.parse_args()

    # Load YAML config defaults if provided
    train_kwargs: dict = {}
    if args.config:
        import yaml

        with open(args.config) as f:
            raw = yaml.safe_load(f)
        train_section = raw.get("training", {})
        yaml_map = {
            "batch_size": "batch_size", "learning_rate": "learning_rate",
            "weight_decay": "weight_decay", "max_steps": "max_steps",
            "warmup_steps": "warmup_steps", "max_grad_norm": "max_grad_norm",
            "gradient_accumulation_steps": "gradient_accumulation_steps",
            "checkpoint_dir": "checkpoint_dir", "checkpoint_every": "checkpoint_every",
            "log_every": "log_every", "precision": "precision",
            "embed_lr_ratio": "embed_lr_ratio", "scheduler_type": "scheduler_type",
            "stable_steps": "stable_steps", "decay_steps": "decay_steps",
            "min_lr_ratio": "min_lr_ratio", "decay_type": "decay_type",
        }
        for yaml_key, kwarg_name in yaml_map.items():
            if yaml_key in train_section:
                train_kwargs[kwarg_name] = train_section[yaml_key]

    # CLI overrides
    cli_overrides = {
        "batch_size": args.batch_size, "learning_rate": args.lr,
        "weight_decay": args.weight_decay, "max_steps": args.max_steps,
        "gradient_accumulation_steps": args.grad_accum,
        "checkpoint_dir": args.checkpoint_dir, "precision": args.precision,
        "embed_lr_ratio": args.embed_lr_ratio,
        "scheduler_type": args.scheduler_type, "stable_steps": args.stable_steps,
        "decay_steps": args.decay_steps, "decay_type": args.decay_type,
    }
    for k, v in cli_overrides.items():
        if v is not None:
            train_kwargs[k] = v

    # Apply defaults for anything still unset
    defaults = {
        "batch_size": 2, "learning_rate": 5e-7, "max_steps": 2000,
        "gradient_accumulation_steps": 4, "checkpoint_dir": "checkpoints/dpo",
        "precision": "bf16",
    }
    for k, v in defaults.items():
        train_kwargs.setdefault(k, v)

    # Load SFT model
    print("Loading SFT model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ModelConfig(**ckpt["model_config"])

    model = CodeForgeModel(config)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Policy model loaded: {model.param_count():,} params")

    # Create reference model (frozen copy of SFT model)
    print("Creating reference model (frozen copy)...")
    ref_model = CodeForgeModel(config)
    ref_model.load_state_dict(ckpt["model_state_dict"])

    if not args.data:
        print("Error: Provide --data (DPO preference JSONL files)")
        sys.exit(1)

    # Load tokenizer and dataset
    tokenizer = CodeForgeTokenizer(args.tokenizer)
    dataset = DPODataset(args.data, tokenizer, max_seq_len=config.max_seq_len)

    # Train
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        dataset=dataset,
        beta=args.beta,
        **train_kwargs,
    )
    trainer.train()


if __name__ == "__main__":
    main()
