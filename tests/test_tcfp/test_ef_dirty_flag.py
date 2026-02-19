"""Verify EF buffers update correctly with the dirty-flag mechanism.

This specifically tests the scenario that caused frozen EF buffers:
fused AdamW doesn't increment weight._version, so the old version-based
check always evaluated to False. The new dirty-flag approach requires
explicit mark_weights_updated() calls from the trainer.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from codeforge.tcfp.nn import TCFPLinear, mark_tcfp_weights_updated

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_ef_updates_with_dirty_flag() -> None:
    """EF buffers change after mark_weights_updated + forward."""
    layer = TCFPLinear(
        128, 64, use_tensor_cores=True, error_feedback=True,
    ).to(DEVICE)
    layer._param_name = "test"

    # Prime EF on first forward (dirty=True initially)
    x1 = torch.randn(4, 128, device=DEVICE, requires_grad=True)
    layer(x1).sum().backward()

    ef_after_prime = layer._error_state._buffers["test"].clone()
    assert ef_after_prime.abs().sum() > 0, "EF should be non-zero after priming"
    assert not layer._weights_dirty, "Dirty flag should be False after forward"

    # Second forward WITHOUT mark_weights_updated — EF should NOT change
    x2 = torch.randn(4, 128, device=DEVICE, requires_grad=True)
    layer(x2).sum().backward()
    ef_no_update = layer._error_state._buffers["test"]
    assert torch.equal(ef_after_prime, ef_no_update), (
        "EF changed without mark_weights_updated()"
    )

    # Simulate optimizer step (use large LR to ensure weight actually changes)
    opt = torch.optim.AdamW(layer.parameters(), lr=0.1)
    opt.step()
    opt.zero_grad()

    # Mark weights dirty (what the trainer does after optimizer.step())
    layer.mark_weights_updated()
    assert layer._weights_dirty, "mark_weights_updated should set flag"

    # Forward after mark — EF should update to new values
    x3 = torch.randn(4, 128, device=DEVICE, requires_grad=True)
    layer(x3).sum().backward()

    ef_after_step = layer._error_state._buffers["test"]
    assert not torch.equal(ef_after_prime, ef_after_step), (
        "EF should differ after optimizer step + mark_weights_updated"
    )
    assert not layer._weights_dirty, "Dirty flag should be consumed"
    print("PASS: EF updates correctly with dirty-flag mechanism")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused AdamW requires CUDA")
def test_ef_updates_with_fused_adamw() -> None:
    """Specifically test fused AdamW — the scenario that was broken."""

    layer = TCFPLinear(
        128, 64, use_tensor_cores=True, error_feedback=True,
    ).to("cuda")
    layer._param_name = "test"

    # Use fused AdamW — the exact optimizer that broke _version tracking
    opt = torch.optim.AdamW(layer.parameters(), lr=0.1, fused=True)

    # Prime
    x = torch.randn(4, 128, device="cuda", requires_grad=True)
    layer(x).sum().backward()
    ef_step0 = layer._error_state._buffers["test"].clone()

    # Step 1
    opt.step()
    opt.zero_grad()
    layer.mark_weights_updated()

    x = torch.randn(4, 128, device="cuda", requires_grad=True)
    layer(x).sum().backward()
    ef_step1 = layer._error_state._buffers["test"].clone()

    assert not torch.equal(ef_step0, ef_step1), (
        "EF frozen after fused AdamW step — dirty flag not working!"
    )

    # Step 2
    opt.step()
    opt.zero_grad()
    layer.mark_weights_updated()

    x = torch.randn(4, 128, device="cuda", requires_grad=True)
    layer(x).sum().backward()
    ef_step2 = layer._error_state._buffers["test"]

    assert not torch.equal(ef_step1, ef_step2), (
        "EF frozen after second fused AdamW step"
    )
    print("PASS: EF updates correctly with fused AdamW")


def test_mark_tcfp_weights_updated_utility() -> None:
    """The model-level utility function marks all layers."""
    model = torch.nn.Sequential(
        TCFPLinear(64, 32, use_tensor_cores=True, error_feedback=True),
        TCFPLinear(32, 16, use_tensor_cores=True, error_feedback=True),
    ).to(DEVICE)

    # Forward to consume initial dirty flags
    for layer in model:
        if isinstance(layer, TCFPLinear):
            layer._param_name = f"test_{id(layer)}"
    x = torch.randn(4, 64, device=DEVICE, requires_grad=True)
    model(x).sum().backward()

    # All should be clean
    for layer in model:
        if isinstance(layer, TCFPLinear):
            assert not layer._weights_dirty

    # Mark all dirty via utility
    mark_tcfp_weights_updated(model)

    # All should be dirty
    for layer in model:
        if isinstance(layer, TCFPLinear):
            assert layer._weights_dirty, "mark_tcfp_weights_updated didn't set flag"

    print("PASS: mark_tcfp_weights_updated sets all layers dirty")


if __name__ == "__main__":
    test_ef_updates_with_dirty_flag()
    test_ef_updates_with_fused_adamw()
    test_mark_tcfp_weights_updated_utility()
    print("\nAll EF dirty-flag tests passed!")
