# Quantization module for accelerated_inference
# INT8 weight-only quantization support

try:
    from .int8_weight_only import Int8Linear
except ImportError:
    # CUDA extension not compiled yet
    Int8Linear = None

__all__ = ['Int8Linear']
