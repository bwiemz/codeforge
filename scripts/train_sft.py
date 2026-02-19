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
    parser.add_argument("--data", "-d", nargs="+", default=None, help="SFT data JSONL files")
    parser.add_argument("--hf-dataset", type=str, default=None,
                        help="HuggingFace SFT dataset (e.g. ise-uiuc/Magicoder-OSS-Instruct-75K)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--checkpoint-dir", default="checkpoints/sft")
    parser.add_argument("--precision", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--hf-token", type=str, default=None, help="HuggingFace token for gated datasets")
    args = parser.parse_args()

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
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        gradient_accumulation_steps=args.grad_accum,
        checkpoint_dir=args.checkpoint_dir,
        precision=args.precision,
    )
    trainer.train()


if __name__ == "__main__":
    main()
