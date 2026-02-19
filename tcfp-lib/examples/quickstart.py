"""
TCFP Quickstart
===============

Minimal example: convert a standard transformer to TCFP-12 and train.

Usage::

    python quickstart.py

Requires a CUDA GPU with FP8 tensor cores (Ada Lovelace / Hopper / Blackwell).
Falls back to fake-quantize + F.linear on CPU or older GPUs.
"""

import torch
import torch.nn as nn

from tcfp import TCFPMode, convert_to_tcfp, diagnose


# -- 1. Define a small transformer (or use your own) -----------------------

class TinyMLP(nn.Module):
    """Minimal MLP to demonstrate TCFP conversion."""

    def __init__(self, dim: int = 256, hidden: int = 512):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.up = nn.Linear(dim, hidden)
        self.down = nn.Linear(hidden, dim)
        self.head = nn.Linear(dim, 1)  # will be skipped by default

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln(x)
        h = torch.relu(self.up(h))
        h = self.down(h)
        return self.head(x + h)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyMLP().to(device)

    # -- 2. Run diagnostics before conversion --------------------------------
    print("=== Pre-conversion diagnostics ===")
    report = diagnose(model, scale_block_size=128)
    print(f"  Linear layers:     {report.total_linear_layers}")
    print(f"  TC-eligible:       {report.tc_eligible_layers}")
    print(f"  Block-scale OK:    {report.block_scale_eligible_layers}")
    print(f"  VRAM savings est.: {report.estimated_savings_pct:.0f}%")
    print()

    # -- 3. Convert to TCFP-12 -----------------------------------------------
    use_tc = device == "cuda"
    convert_to_tcfp(
        model,
        mode=TCFPMode.TCFP12,
        use_tensor_cores=use_tc,
        scale_block_size=128 if use_tc else None,
        error_feedback=True,
        skip_patterns=("head",),  # keep head in FP32
    )
    print("=== Model after conversion ===")
    print(model)
    print()

    # -- 4. Train normally ---------------------------------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for step in range(5):
        x = torch.randn(8, 256, device=device)
        target = torch.randn(8, 1, device=device)

        pred = model(x)
        loss = nn.functional.mse_loss(pred, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(f"  step {step}: loss={loss.item():.4f}")

    print("\nDone. TCFP-12 training works identically to standard PyTorch.")


if __name__ == "__main__":
    main()
