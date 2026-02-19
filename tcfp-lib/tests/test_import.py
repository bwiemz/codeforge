"""Smoke test: verify the standalone package imports correctly."""

import importlib


def test_top_level_imports():
    """All public symbols should be importable from the top-level package."""
    import tcfp

    assert hasattr(tcfp, "__version__")
    assert tcfp.__version__ == "0.1.0"

    # High-level API
    assert callable(tcfp.convert_to_tcfp)
    assert callable(tcfp.diagnose)
    assert callable(tcfp.export)
    assert tcfp.TCFPLinear is not None
    assert tcfp.TCFPLayerNorm is not None
    assert tcfp.TCFPEmbedding is not None

    # Core primitives
    assert callable(tcfp.to_fp8_e4m3)
    assert callable(tcfp.to_fp8_e5m2)
    assert tcfp.TCFPMode is not None


def test_submodule_imports():
    """Submodules should be importable independently."""
    core = importlib.import_module("tcfp.core")
    assert hasattr(core, "TCFPMode")

    nn = importlib.import_module("tcfp.nn")
    assert hasattr(nn, "TCFPLinear")

    tcfp12 = importlib.import_module("tcfp.tcfp12")
    assert hasattr(tcfp12, "fake_quantize_tcfp12")

    training = importlib.import_module("tcfp.training")
    assert hasattr(training, "TCFPMonitor")


def test_optional_kernels_no_crash():
    """Importing kernels should not crash even if Triton is missing."""
    try:
        from tcfp.kernels import is_triton_available
        # Should return True or False, never crash
        result = is_triton_available()
        assert isinstance(result, bool)
    except ImportError:
        # kernels module itself might not import if triton is truly broken
        pass


def test_optional_cuda_ext_no_crash():
    """Importing cuda_ext should not crash even without CUDA Toolkit."""
    from tcfp.cuda_ext import is_cuda_ext_available
    result = is_cuda_ext_available()
    assert isinstance(result, bool)
