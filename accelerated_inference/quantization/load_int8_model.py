"""
INT8 Model Loader for accelerated_inference

This module provides functions to load an INT8 quantized model
and replace Linear layers with Int8Linear modules.

Designed for transformers 4.33.0 compatibility.

Usage:
    from accelerated_inference.quantization.load_int8_model import load_int8_model
    
    model, tokenizer = load_int8_model(
        model_path="pythia-2.8b-local",
        ckpt_dir="checkpoints/pythia-2.8b-int8",
        dtype=torch.bfloat16,  # INT8 kernel requires bfloat16
        device="cuda"
    )
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import Int8Linear from the CUDA extension
try:
    from .int8_weight_only import Int8Linear
except ImportError:
    # Fallback if extension not compiled yet
    Int8Linear = None


def load_quantized_checkpoint(ckpt_dir):
    """
    Load INT8 quantized checkpoint files.
    
    Args:
        ckpt_dir: Directory containing the checkpoint files
    
    Returns:
        wq: Dict of INT8 weights (name -> tensor)
        sc: Dict of scales (name -> tensor)
        b: Dict of biases (name -> tensor)
    """
    wq = torch.load(os.path.join(ckpt_dir, "weights_int8.pt"), map_location="cpu")
    sc = torch.load(os.path.join(ckpt_dir, "scales_fp32.pt"), map_location="cpu")
    b = torch.load(os.path.join(ckpt_dir, "bias_fp16.pt"), map_location="cpu")
    return wq, sc, b


def replace_linear_with_int8(model, wq, sc, b, dtype, device):
    """
    Replace Linear layers with Int8Linear modules.
    
    Args:
        model: HuggingFace model
        wq: Dict of INT8 weights
        sc: Dict of scales
        b: Dict of biases
        dtype: Target dtype (torch.float16 or torch.bfloat16)
        device: Target device
    """
    if Int8Linear is None:
        raise ImportError(
            "Int8Linear not available. Please compile the CUDA extension:\n"
            "  cd accelerated_inference/accelerated_inference/quantization/int8_weight_only\n"
            "  pip install . --no-build-isolation"
        )
    
    replaced_count = 0
    
    for name, m in list(model.named_modules()):
        if isinstance(m, torch.nn.Linear) and name in wq:
            # Get parent module
            parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
            child_name = name.rsplit('.', 1)[-1]
            parent = model.get_submodule(parent_name) if parent_name != '' else model
            
            # Create Int8Linear replacement
            new_m = Int8Linear(
                in_features=m.in_features,
                out_features=m.out_features,
                bias=(m.bias is not None),
                device=device,
                dtype=dtype
            )
            
            # Load quantized weights
            new_m.Wq = wq[name].to(device)
            new_m.scale = sc[name].to(device)
            if m.bias is not None and name in b:
                new_m.bias = b[name].to(device).to(dtype)
            
            # Replace module
            setattr(parent, child_name, new_m)
            replaced_count += 1
    
    return replaced_count


def load_int8_model(model_path, ckpt_dir, dtype=torch.bfloat16, device="cuda"):
    """
    Load an INT8 quantized model.
    
    This function:
    1. Loads the original FP16/BF16 model
    2. Loads the INT8 checkpoint
    3. Replaces Linear layers with Int8Linear modules
    
    Args:
        model_path: Path to HuggingFace model (or model name)
        ckpt_dir: Directory containing INT8 checkpoint files
        dtype: Activation dtype (torch.bfloat16 recommended for INT8 kernel)
        device: Target device ("cuda" or "cpu")
    
    Returns:
        model: Model with INT8 Linear layers
        tokenizer: Tokenizer for the model
    """
    print(f"Loading INT8 model from {model_path}")
    print(f"Checkpoint directory: {ckpt_dir}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.pad_token_id = 0
    
    # Load model (will be replaced with INT8 layers)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )
    
    print(f"Model Before Quantization: {model}")
    
    # Load INT8 checkpoint
    wq, sc, b = load_quantized_checkpoint(ckpt_dir)
    print(f"Loaded {len(wq)} quantized layers from checkpoint")        
    for name, m in list(model.named_modules()):
        if "mlp" in name and isinstance(m, torch.nn.Module):
            # Check if it has the standard GPT-NeoX MLP structure
            if hasattr(m, "dense_h_to_4h") and hasattr(m, "dense_4h_to_h") and hasattr(m, "act"):
                # Fuse Activation into dense_h_to_4h
                # 1. Replace dense_h_to_4h with Int8Linear(act_type="gelu")
                linear_name = name + ".dense_h_to_4h"
                if linear_name in wq:
                    old_linear = m.dense_h_to_4h
                    new_linear = Int8Linear(
                        in_features=old_linear.in_features,
                        out_features=old_linear.out_features,
                        bias=(old_linear.bias is not None),
                        device=device,
                        dtype=dtype,
                        act_type="gelu"
                    )
                    new_linear.Wq = wq[linear_name].to(device)
                    new_linear.scale = sc[linear_name].to(device)
                    if old_linear.bias is not None and linear_name in b:
                        new_linear.bias = b[linear_name].to(device).to(dtype)
                        
                    m.dense_h_to_4h = new_linear
                    # 2. Replace act with Identity
                    m.act = torch.nn.Identity()
                        
                    # 3. Replace dense_4h_to_h with standard Int8Linear
                    linear_name_2 = name + ".dense_4h_to_h"
                    if linear_name_2 in wq:
                        old_linear_2 = m.dense_4h_to_h
                        new_linear_2 = Int8Linear(
                            in_features=old_linear_2.in_features,
                            out_features=old_linear_2.out_features,
                            bias=(old_linear_2.bias is not None),
                            device=device,
                            dtype=dtype,
                            act_type="none"
                        )
                        new_linear_2.Wq = wq[linear_name_2].to(device)
                        new_linear_2.scale = sc[linear_name_2].to(device)
                        if old_linear_2.bias is not None and linear_name_2 in b:
                            new_linear_2.bias = b[linear_name_2].to(device).to(dtype)
                        m.dense_4h_to_h = new_linear_2
    
    # Replace Linear with Int8Linear
    replaced_count = replace_linear_with_int8(model, wq, sc, b, dtype, device)
    print(f"Replaced {replaced_count} Linear layers with Int8Linear")
    
    model.eval()
    
    print(f"Model After Quantization: {model}")
    
    return model, tokenizer


def theoretical_param_bytes_fp16(model):
    """Calculate theoretical parameter bytes for FP16 model."""
    total = 0
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            total += m.weight.numel() * 2  # 2 bytes per FP16
            if m.bias is not None:
                total += m.bias.numel() * 2
    return total


def theoretical_param_bytes_int8(wq, sc, b, dtype=torch.float16):
    """Calculate theoretical parameter bytes for INT8 model."""
    bytes_per_scale = 2 if dtype == torch.float16 else 4
    bytes_per_bias = 2  # FP16 bias
    
    w_bytes = sum(t.numel() for t in wq.values())  # 1 byte per INT8
    s_bytes = sum(t.numel() for t in sc.values()) * bytes_per_scale
    b_bytes = sum(t.numel() for t in b.values()) * bytes_per_bias if b else 0
    
    return w_bytes + s_bytes + b_bytes
