"""
TCFP-12 Fine-Tuning with Hugging Face Transformers
===================================================

Shows how to apply TCFP-12 to a Hugging Face model for full-parameter
fine-tuning on FP8 tensor cores.

Usage::

    pip install transformers datasets accelerate
    python llama_finetune.py --model meta-llama/Llama-3.2-1B --dataset wikitext

Requires:
    - CUDA GPU with FP8 tensor cores (SM89+)
    - Triton >= 3.0 (recommended, for fused kernels)
"""

from __future__ import annotations

import argparse

import torch

from tcfp import convert_to_tcfp, diagnose, export
from tcfp.training import TCFPMonitor, TrainingPreset


def main() -> None:
    parser = argparse.ArgumentParser(description="TCFP-12 fine-tuning example")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--dataset", default="wikitext")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=128, choices=[32, 64, 128])
    parser.add_argument("--abd", action="store_true", help="Asymmetric backward decomposition")
    parser.add_argument("--srr", action="store_true", help="Stochastic residual rounding")
    parser.add_argument("--no-error-feedback", action="store_true")
    parser.add_argument("--save-path", default="./tcfp_finetuned")
    args = parser.parse_args()

    # ---- Load model --------------------------------------------------------
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("This example requires: pip install transformers")
        return

    print(f"Loading {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32,
    ).cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # ---- Diagnose before conversion ----------------------------------------
    print("\n=== Pre-conversion diagnostics ===")
    report = diagnose(model, scale_block_size=args.block_size)
    print(f"  Total linear layers:    {report.total_linear_layers}")
    print(f"  TC-eligible:            {report.tc_eligible_layers}")
    print(f"  Block-scale eligible:   {report.block_scale_eligible_layers}")
    print(f"  Estimated VRAM savings: {report.estimated_savings_pct:.0f}%")

    if report.fallback_layers > 0:
        print(f"\n  WARNING: {report.fallback_layers} layers will use fallback path:")
        for layer in report.layer_details:
            if layer.fallback_reason:
                print(f"    {layer.name}: {layer.fallback_reason}")

    # ---- Convert to TCFP-12 ------------------------------------------------
    print("\nConverting to TCFP-12...")
    convert_to_tcfp(
        model,
        use_tensor_cores=True,
        scale_block_size=args.block_size,
        error_feedback=not args.no_error_feedback,
        abd=args.abd,
        srr=args.srr,
        skip_patterns=("lm_head",),
    )

    # ---- Set up training ---------------------------------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    monitor = TCFPMonitor(model)

    print(f"\nTraining for {args.steps} steps...")
    print("(This is a skeleton — plug in your real data pipeline)\n")

    for step in range(args.steps):
        # Placeholder: use your real tokenized batches here
        input_ids = torch.randint(0, tokenizer.vocab_size, (args.batch_size, 128)).cuda()
        labels = input_ids.clone()

        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % 10 == 0:
            snapshot = monitor.snapshot()
            print(
                f"  step {step:4d}  loss={loss.item():.4f}"
                f"  grad_norm={snapshot.grad_norm:.4f}"
            )

    # ---- Export for inference -----------------------------------------------
    print(f"\nExporting to {args.save_path}...")
    vanilla = export(model)
    vanilla.save_pretrained(args.save_path)
    tokenizer.save_pretrained(args.save_path)
    print("Done.")


if __name__ == "__main__":
    main()
