"""Export entry point for CodeForge models."""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from codeforge.model.config import ModelConfig
from codeforge.model.transformer import CodeForgeModel
from codeforge.export.quantize import quantize_model, save_quantized
from codeforge.export.gguf import export_to_gguf


def main():
    parser = argparse.ArgumentParser(description="Export CodeForge model")
    parser.add_argument("--checkpoint", "-c", required=True, help="Model checkpoint path")
    parser.add_argument(
        "--format", "-f", required=True, choices=["gguf", "quantized"],
        help="Export format",
    )
    parser.add_argument("--output", "-o", required=True, help="Output path")
    parser.add_argument(
        "--dtype", default="f16", choices=["f16", "f32"],
        help="Weight dtype for GGUF export",
    )
    parser.add_argument(
        "--bits", type=int, default=8, choices=[4, 8],
        help="Quantization bits (for quantized format)",
    )
    args = parser.parse_args()

    # Load model
    print("Loading model...")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ModelConfig(**ckpt["model_config"])
    model = CodeForgeModel(config)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Model loaded: {model.param_count():,} params")

    if args.format == "gguf":
        export_to_gguf(model, args.output, dtype=args.dtype)
    elif args.format == "quantized":
        model = quantize_model(model, bits=args.bits)
        save_quantized(model, args.output)

    print("Done!")


if __name__ == "__main__":
    main()
