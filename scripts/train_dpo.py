"""DPO training entry point."""

import argparse
import copy
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from codeforge.model.config import ModelConfig
from codeforge.model.transformer import CodeForgeModel
from codeforge.tokenizer.tokenizer import CodeForgeTokenizer
from codeforge.training.dpo_trainer import DPOTrainer, DPODataset


def main():
    parser = argparse.ArgumentParser(description="DPO training for CodeForge")
    parser.add_argument("--checkpoint", "-c", required=True, help="SFT model checkpoint")
    parser.add_argument("--tokenizer", "-t", required=True, help="Tokenizer path")
    parser.add_argument("--data", "-d", nargs="+", required=True, help="DPO data JSONL files")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta parameter")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-7)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--checkpoint-dir", default="checkpoints/dpo")
    parser.add_argument("--precision", default="bf16", choices=["bf16", "fp16", "fp32"])
    args = parser.parse_args()

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

    # Load tokenizer and dataset
    tokenizer = CodeForgeTokenizer(args.tokenizer)
    dataset = DPODataset(args.data, tokenizer, max_seq_len=config.max_seq_len)

    # Train
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        dataset=dataset,
        beta=args.beta,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_steps=args.max_steps,
        gradient_accumulation_steps=args.grad_accum,
        checkpoint_dir=args.checkpoint_dir,
        precision=args.precision,
    )
    trainer.train()


if __name__ == "__main__":
    main()
