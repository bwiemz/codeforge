"""SFT training entry point."""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from codeforge.model.config import ModelConfig, get_preset
from codeforge.model.transformer import CodeForgeModel
from codeforge.tokenizer.tokenizer import CodeForgeTokenizer
from codeforge.training.sft_trainer import SFTTrainer, SFTDataset


def main():
    parser = argparse.ArgumentParser(description="SFT training for CodeForge")
    parser.add_argument("--checkpoint", "-c", required=True, help="Base model checkpoint")
    parser.add_argument("--tokenizer", "-t", required=True, help="Tokenizer path")
    parser.add_argument("--config", type=str, default=None, help="YAML config file")
    parser.add_argument("--data", "-d", nargs="+", default=None, help="SFT data JSONL files")
    parser.add_argument("--hf-dataset", type=str, default=None,
                        help="HuggingFace SFT dataset (e.g. ise-uiuc/Magicoder-OSS-Instruct-75K)")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--precision", default=None, choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--embed-lr-ratio", type=float, default=None)
    parser.add_argument("--scheduler-type", default=None, choices=["cosine", "wsd"])
    parser.add_argument("--stable-steps", type=int, default=None)
    parser.add_argument("--decay-steps", type=int, default=None)
    parser.add_argument("--decay-type", default=None, choices=["cosine", "linear"])
    parser.add_argument("--hf-token", type=str, default=None,
                        help="HuggingFace token for gated datasets")
    args = parser.parse_args()

    # Load YAML config defaults if provided
    train_kwargs: dict = {}
    if args.config:
        import yaml

        with open(args.config) as f:
            raw = yaml.safe_load(f)
        train_section = raw.get("training", {})
        # Map YAML keys to SFTTrainer kwargs
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

        # HF dataset from config
        if not args.hf_dataset and "hf_dataset" in raw.get("data", {}):
            args.hf_dataset = raw["data"]["hf_dataset"]

    # CLI overrides
    cli_overrides = {
        "batch_size": args.batch_size, "learning_rate": args.lr,
        "max_steps": args.max_steps, "warmup_steps": args.warmup_steps,
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
        "batch_size": 4, "learning_rate": 2e-5, "max_steps": 5000,
        "warmup_steps": 100, "gradient_accumulation_steps": 4,
        "checkpoint_dir": "checkpoints/sft", "precision": "bf16",
    }
    for k, v in defaults.items():
        train_kwargs.setdefault(k, v)

    # HuggingFace login if token provided
    if args.hf_token:
        from huggingface_hub import login
        login(token=args.hf_token)
        print("Logged in to HuggingFace Hub")

    if not args.data and not args.hf_dataset:
        print("Error: Provide --data (JSONL files) or --hf-dataset (HuggingFace dataset)")
        sys.exit(1)

    # Load base model
    print("Loading base model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ModelConfig(**ckpt["model_config"])
    model = CodeForgeModel(config)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Model loaded: {model.param_count():,} params")

    # Load tokenizer and dataset
    tokenizer = CodeForgeTokenizer(args.tokenizer)
    dataset = SFTDataset(
        data_paths=args.data or [],
        tokenizer=tokenizer,
        max_seq_len=config.max_seq_len,
        hf_dataset=args.hf_dataset,
    )

    # Train
    trainer = SFTTrainer(
        model=model,
        dataset=dataset,
        **train_kwargs,
    )
    trainer.train()


if __name__ == "__main__":
    main()
