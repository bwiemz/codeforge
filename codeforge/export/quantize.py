"""Post-training quantization for CodeForge models.

Supports weight-only quantization (INT8, INT4) for reduced memory usage
during inference on consumer hardware.
"""

import torch
import torch.nn as nn
from typing import Optional
from pathlib import Path

from ..model.config import ModelConfig
from ..model.transformer import CodeForgeModel


def _quantize_tensor_absmax(weight: torch.Tensor, bits: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a weight tensor using absmax per-channel quantization.

    Args:
        weight: Float weight tensor
        bits: Quantization bits (4 or 8)

    Returns:
        (quantized_weight, scale) where weight ≈ quantized_weight * scale
    """
    qmax = 2 ** (bits - 1) - 1
    # Per-channel (per output dim) scale
    scale = weight.abs().amax(dim=-1, keepdim=True) / qmax
    scale = scale.clamp(min=1e-8)

    quantized = torch.clamp(torch.round(weight / scale), -qmax, qmax).to(
        torch.int8 if bits == 8 else torch.int8  # Store int4 in int8 container
    )

    return quantized, scale.squeeze(-1)


def _dequantize_tensor(quantized: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize a tensor."""
    return quantized.float() * scale.unsqueeze(-1)


class QuantizedLinear(nn.Module):
    """Drop-in replacement for nn.Linear with quantized weights."""

    def __init__(self, quantized_weight: torch.Tensor, scale: torch.Tensor, bias: Optional[torch.Tensor] = None):
        super().__init__()
        self.register_buffer("quantized_weight", quantized_weight)
        self.register_buffer("scale", scale)
        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dequantize on-the-fly during forward pass
        weight = _dequantize_tensor(self.quantized_weight, self.scale)
        return nn.functional.linear(x, weight, self.bias)


def quantize_model(
    model: CodeForgeModel,
    bits: int = 8,
    exclude_layers: Optional[list[str]] = None,
) -> CodeForgeModel:
    """Quantize model weights in-place.

    Args:
        model: The model to quantize
        bits: Quantization bits (4 or 8)
        exclude_layers: Layer name patterns to skip (e.g., ["tok_embeddings", "norm"])

    Returns:
        The quantized model (same object, modified in-place)
    """
    if exclude_layers is None:
        exclude_layers = ["tok_embeddings", "norm", "output"]

    total_params = 0
    quantized_params = 0

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            total_params += module.weight.numel()

            # Skip excluded layers
            if any(excl in name for excl in exclude_layers):
                continue

            # Quantize
            q_weight, scale = _quantize_tensor_absmax(module.weight.data, bits)
            bias = module.bias.data if module.bias is not None else None

            # Replace with quantized version
            quantized_linear = QuantizedLinear(q_weight, scale, bias)

            # Find parent module and replace
            parts = name.rsplit(".", 1)
            if len(parts) == 2:
                parent = dict(model.named_modules())[parts[0]]
                setattr(parent, parts[1], quantized_linear)
            else:
                setattr(model, name, quantized_linear)

            quantized_params += module.weight.numel()

    print(f"Quantized {quantized_params:,} / {total_params:,} parameters to INT{bits}")
    print(f"Estimated memory reduction: {(1 - bits/32)*100:.0f}%")

    return model


def save_quantized(model: CodeForgeModel, path: str | Path) -> None:
    """Save a quantized model checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": model.config.__dict__,
        "quantized": True,
    }, path)
    print(f"Quantized model saved to {path}")


def load_quantized(path: str | Path, device: Optional[torch.device] = None) -> CodeForgeModel:
    """Load a quantized model checkpoint."""
    device = device or torch.device("cpu")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = ModelConfig(**ckpt["model_config"])

    # Create model shell and load quantized state
    model = CodeForgeModel(config)

    # Need to quantize the model structure first to match state dict keys
    model = quantize_model(model, bits=8)
    model.load_state_dict(ckpt["model_state_dict"])

    return model.to(device)
