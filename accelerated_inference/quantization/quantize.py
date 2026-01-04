"""
INT8 Weight-Only Quantization Script for Pythia Models

This script quantizes Linear layer weights to INT8 (per-channel scaling).
Adapted from CS3602-02 for transformers 4.33.0 compatibility.

Usage:
    python -m accelerated_inference.quantization.quantize \
        --model_path pythia-2.8b-local \
        --out_dir checkpoints/pythia-2.8b-int8

Output files:
    - weights_int8.pt: INT8 quantized weights (dict)
    - scales_fp32.pt: Per-output-channel scales (dict)
    - bias_fp16.pt: Original biases in FP16 (dict)
"""

import os
import argparse
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def quantize_linear_weight(W):
    """
    Quantize weight matrix to INT8 with per-output-channel scaling.
    
    Args:
        W: Weight tensor [out_features, in_features]
    
    Returns:
        q: INT8 quantized weights
        s: FP32 scales (per output channel)
    """
    Wf = W.float()
    # Per-output-channel max absolute value / 127
    s = torch.clamp(Wf.abs().amax(dim=1) / 127.0, min=1e-8)
    # Quantize: W_int8 = round(W / scale)
    q = torch.clamp((Wf / s[:, None]).round(), -128, 127).to(torch.int8)
    return q, s


def quantize_model(model_path, out_dir, dtype="float16"):
    """
    Quantize all Linear layers in a model to INT8.
    
    Args:
        model_path: Path to HuggingFace model
        out_dir: Output directory for quantized weights
        dtype: Model loading dtype ("float16" or "bfloat16")
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # Convert dtype string to torch dtype
    torch_dtype = torch.float16 if dtype == "float16" else torch.bfloat16
    
    print(f"Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    
    weights_q = {}
    scales = {}
    biases = {}
    
    modules = list(model.named_modules())
    
    print(f"Quantizing {len(modules)} modules...")
    for name, m in tqdm(modules, desc="Quantizing", total=len(modules)):
        if isinstance(m, torch.nn.Linear):
            W = m.weight.detach().to("cpu")
            q, s = quantize_linear_weight(W)
            weights_q[name] = q
            scales[name] = s.to(torch.float32)
            if m.bias is not None:
                biases[name] = m.bias.detach().to("cpu").half()
    
    # Save quantized weights
    torch.save(weights_q, os.path.join(out_dir, "weights_int8.pt"))
    torch.save(scales, os.path.join(out_dir, "scales_fp32.pt"))
    torch.save(biases, os.path.join(out_dir, "bias_fp16.pt"))
    
    print("=" * 60)
    print(f"Quantization complete!")
    print(f"Linear layers quantized: {len(weights_q)}")
    print(f"Output directory: {out_dir}")
    
    # Print scale statistics
    if len(scales) > 0:
        all_s = torch.cat([v.flatten() for v in scales.values()])
        print(f"Scale stats: min={float(all_s.min()):.6f}, "
              f"mean={float(all_s.mean()):.6f}, max={float(all_s.max()):.6f}")
    
    # Estimate memory savings
    fp16_bytes = sum(v.numel() * 2 for v in weights_q.values())  # 2 bytes per FP16
    int8_bytes = sum(v.numel() for v in weights_q.values())  # 1 byte per INT8
    scale_bytes = sum(v.numel() * 4 for v in scales.values())  # 4 bytes per FP32 scale
    
    print(f"Memory savings: {(fp16_bytes - int8_bytes - scale_bytes) / 1024 / 1024:.1f} MB")
    print(f"Compression ratio: {fp16_bytes / (int8_bytes + scale_bytes):.2f}x")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="INT8 weight-only quantization")
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="pythia-2.8b-local",
        help="Path to HuggingFace model"
    )
    parser.add_argument(
        "--out_dir", 
        type=str, 
        default="checkpoints/pythia-2.8b-int8",
        help="Output directory for quantized weights"
    )
    parser.add_argument(
        "--dtype", 
        type=str, 
        default="float16",
        choices=["float16", "bfloat16"],
        help="Model loading dtype"
    )
    args = parser.parse_args()
    
    quantize_model(args.model_path, args.out_dir, args.dtype)


if __name__ == "__main__":
    main()
