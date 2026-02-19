import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Compute capability: {torch.cuda.get_device_capability(0)}")
print(f"FP8 e4m3fn: {hasattr(torch, 'float8_e4m3fn')}")
print(f"FP8 e5m2: {hasattr(torch, 'float8_e5m2')}")

# Test if FP8 matmul works on this GPU
try:
    a = torch.randn(64, 64, device="cuda").to(torch.float8_e4m3fn)
    b = torch.randn(64, 64, device="cuda").to(torch.float8_e4m3fn)
    # FP8 matmul requires scale tensors
    scale_a = torch.tensor(1.0, device="cuda")
    scale_b = torch.tensor(1.0, device="cuda")
    c = torch._scaled_mm(a, b.t(), scale_a=scale_a, scale_b=scale_b, out_dtype=torch.bfloat16)
    print(f"FP8 _scaled_mm: WORKS (output shape {c.shape}, dtype {c.dtype})")
except Exception as e:
    print(f"FP8 _scaled_mm: FAILED - {e}")

# Check torchao
try:
    import torchao
    print(f"torchao: {torchao.__version__}")
    try:
        from torchao.float8 import Float8LinearConfig
        print("torchao.float8.Float8LinearConfig: available")
    except ImportError as e:
        print(f"torchao.float8: {e}")
    try:
        from torchao.float8 import convert_to_float8_training
        print("torchao.float8.convert_to_float8_training: available")
    except ImportError as e:
        print(f"torchao.float8 convert: {e}")
except ImportError:
    print("torchao: NOT INSTALLED")

# Check transformer_engine
try:
    import transformer_engine as te
    print(f"transformer_engine: {te.__version__}")
except ImportError:
    print("transformer_engine: NOT INSTALLED")
