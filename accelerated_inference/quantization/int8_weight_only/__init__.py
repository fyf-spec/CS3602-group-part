# INT8 Weight-Only CUDA Extension
# Provides Int8Linear module for INT8 weight-only quantization

try:
    from .int8_linear import Int8Linear
    from . import w8a16_gemm
    __all__ = ['Int8Linear', 'w8a16_gemm']
except ImportError:
    # CUDA extension not compiled yet
    Int8Linear = None
    w8a16_gemm = None
    __all__ = []
