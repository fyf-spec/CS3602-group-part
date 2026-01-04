"""
Accelerated Inference: Efficient KV Cache Strategies for LLM Inference

This package provides implementations of various KV cache compression strategies:
- H2O: Heavy-Hitter Oracle
- Lazy H2O: Periodic H2O updates
- StreamingLLM: Sink + Recent window
- SepLLM: Separator-aware eviction
- UnifiedKVCache: Combined strategies
- LazyUnifiedKVCache: Periodic update version of UnifiedKVCache
- INT8 Quantization: Weight-only INT8 quantization for Linear layers
"""

__version__ = "0.2.0"

# Import main utilities
from .utils import (
    enable_gpt_neox_pos_shift_attention,
    H2OKVCache,
    LazyH2OKVCache,
)

# Import KV cache presses
try:
    from .kvpress.presses.benchmark_presses import StartRecentKVCache, SepLLMKVCache
    from .kvpress.presses.unified_press import UnifiedKVCache, LazyUnifiedKVCache
except ImportError:
    # Fallback if kvpress module structure is different
    pass

# Import quantization module (optional, requires CUDA extension)
try:
    from .quantization import Int8Linear
    from .quantization.load_int8_model import load_int8_model
except ImportError:
    # CUDA extension not compiled yet
    Int8Linear = None
    load_int8_model = None

__all__ = [
    "__version__",
    "enable_gpt_neox_pos_shift_attention",
    "H2OKVCache",
    "LazyH2OKVCache",
    "StartRecentKVCache",
    "SepLLMKVCache",
    "UnifiedKVCache",
    "LazyUnifiedKVCache",
    "Int8Linear",
    "load_int8_model",
]
