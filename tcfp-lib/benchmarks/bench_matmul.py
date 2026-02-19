"""
TCFP-12 vs BF16 vs FP8 Matmul Benchmark
========================================

Measures wall-clock throughput and numerical accuracy for:
  1. BF16 matmul (baseline)
  2. Single FP8 GEMM (torch._scaled_mm)
  3. TCFP-12 dual FP8 GEMM (2x _scaled_mm + add)
  4. TCFP-12 fused Triton kernel (if available)
  5. TCFP-12 cuBLASLt C++ ext (if available)

Usage::

    python bench_matmul.py
    python bench_matmul.py --m 4096 --n 4096 --k 4096 --warmup 50 --iters 200

Requires CUDA GPU with FP8 tensor cores (SM89+).
"""

from __future__ import annotations

import argparse
import time

import torch

from tcfp.core import to_fp8_e4m3


def _sync() -> None:
    torch.cuda.synchronize()


def bench(fn, warmup: int = 20, iters: int = 100) -> float:
    """Return median time in ms."""
    for _ in range(warmup):
        fn()
    _sync()

    times = []
    for _ in range(iters):
        _sync()
        t0 = time.perf_counter()
        fn()
        _sync()
        times.append((time.perf_counter() - t0) * 1000)

    times.sort()
    return times[len(times) // 2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=2048)
    parser.add_argument("--n", type=int, default=2048)
    parser.add_argument("--k", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    M, N, K = args.m, args.n, args.k
    print(f"Benchmarking matmul: M={M}, N={N}, K={K}")
    print(f"  warmup={args.warmup}, iters={args.iters}")
    print()

    device = "cuda"
    a_bf16 = torch.randn(M, K, device=device, dtype=torch.bfloat16)
    b_bf16 = torch.randn(K, N, device=device, dtype=torch.bfloat16)

    # Ground truth
    ref = (a_bf16.float() @ b_bf16.float())

    # ---- 1. BF16 baseline --------------------------------------------------
    ms_bf16 = bench(lambda: a_bf16 @ b_bf16, args.warmup, args.iters)
    out_bf16 = a_bf16.float() @ b_bf16.float()
    err_bf16 = (ref - out_bf16).norm() / ref.norm()
    print(f"  BF16:             {ms_bf16:7.3f} ms  rel_err={err_bf16:.2e}")

    # ---- 2. Single FP8 GEMM ------------------------------------------------
    a_fp8, a_inv = to_fp8_e4m3(a_bf16.float())
    # For W, we quantize the transposed weight (N, K) then use .t()
    w_fp8, w_inv = to_fp8_e4m3(b_bf16.t().float())

    def single_fp8():
        return torch._scaled_mm(
            a_fp8, w_fp8.t(), scale_a=a_inv, scale_b=w_inv,
            out_dtype=torch.float32, use_fast_accum=False,
        )

    ms_fp8 = bench(single_fp8, args.warmup, args.iters)
    out_fp8 = single_fp8()
    err_fp8 = (ref - out_fp8).norm() / ref.norm()
    print(f"  FP8 (single):     {ms_fp8:7.3f} ms  rel_err={err_fp8:.2e}")

    # ---- 3. TCFP-12: two _scaled_mm + add ----------------------------------
    w_f32 = b_bf16.t().float()
    w_hi_fp8, w_hi_inv = to_fp8_e4m3(w_f32)
    residual = w_f32 - w_hi_fp8.float() * w_hi_inv
    w_lo_fp8, w_lo_inv = to_fp8_e4m3(residual)

    def tcfp12_fallback():
        out_hi = torch._scaled_mm(
            a_fp8, w_hi_fp8.t(), scale_a=a_inv, scale_b=w_hi_inv,
            out_dtype=torch.float32, use_fast_accum=False,
        )
        out_lo = torch._scaled_mm(
            a_fp8, w_lo_fp8.t(), scale_a=a_inv, scale_b=w_lo_inv,
            out_dtype=torch.float32, use_fast_accum=False,
        )
        return out_hi + out_lo

    ms_tcfp = bench(tcfp12_fallback, args.warmup, args.iters)
    out_tcfp = tcfp12_fallback()
    err_tcfp = (ref - out_tcfp).norm() / ref.norm()
    print(f"  TCFP-12 (2x mm):  {ms_tcfp:7.3f} ms  rel_err={err_tcfp:.2e}")

    # ---- 4. TCFP-12: fused Triton kernel ------------------------------------
    try:
        from tcfp.kernels import fused_dual_gemm_forward, is_triton_available

        if is_triton_available():
            def tcfp12_triton():
                return fused_dual_gemm_forward(
                    a_fp8, w_hi_fp8, w_lo_fp8,
                    a_inv, w_hi_inv, w_lo_inv,
                )

            ms_triton = bench(tcfp12_triton, args.warmup, args.iters)
            out_triton = tcfp12_triton()
            err_triton = (ref - out_triton).norm() / ref.norm()
            print(f"  TCFP-12 (Triton): {ms_triton:7.3f} ms  rel_err={err_triton:.2e}")
        else:
            print("  TCFP-12 (Triton): [skipped — Triton not available]")
    except ImportError:
        print("  TCFP-12 (Triton): [skipped — import error]")

    # ---- 5. TCFP-12: cuBLASLt C++ ext --------------------------------------
    try:
        from tcfp.cuda_ext import cuda_ext_dual_gemm_forward, is_cuda_ext_available

        if is_cuda_ext_available():
            def tcfp12_cublas():
                return cuda_ext_dual_gemm_forward(
                    a_fp8, w_hi_fp8, w_lo_fp8,
                    a_inv, w_hi_inv, w_lo_inv,
                )

            ms_cublas = bench(tcfp12_cublas, args.warmup, args.iters)
            out_cublas = tcfp12_cublas()
            err_cublas = (ref - out_cublas).norm() / ref.norm()
            print(f"  TCFP-12 (cuBLAS): {ms_cublas:7.3f} ms  rel_err={err_cublas:.2e}")
        else:
            print("  TCFP-12 (cuBLAS): [skipped — CUDA ext not available]")
    except ImportError:
        print("  TCFP-12 (cuBLAS): [skipped — import error]")

    # ---- Summary -----------------------------------------------------------
    print()
    speedup_vs_bf16 = ms_bf16 / ms_tcfp if ms_tcfp > 0 else float("inf")
    precision_gain = err_fp8 / err_tcfp if err_tcfp > 0 else float("inf")
    print(f"  TCFP-12 vs BF16: {speedup_vs_bf16:.2f}x throughput")
    print(f"  TCFP-12 vs FP8:  {precision_gain:.1f}x lower error")


if __name__ == "__main__":
    main()
