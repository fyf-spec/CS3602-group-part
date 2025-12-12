import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils.logging import disable_progress_bar
import transformers
import time
import os
import sys
import argparse
import warnings
# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from kvpress import KnormPress
except ImportError:
    # kvpress might not be in path if not installed
    pass 
try:
    from attention.gqa import convert_gptneox_to_gqa
except ImportError:
    pass

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()

parser = argparse.ArgumentParser(description="Speed Benchmark for Attention Mechanisms")
parser.add_argument("--gqa", action="store_true", help="Enable GQA conversion")
parser.add_argument("--kv_heads", type=int, default=4, help="Number of KV heads for GQA")
parser.add_argument("--model_path", type=str, default=r"e:\github\accelerated_inference\pythia-2.8b-local", help="Path to model")
args = parser.parse_args()

MODEL_PATH = args.model_path
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading model: {MODEL_PATH}...")
try:
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype="auto", attn_implementation=None).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    raise e

if args.gqa:
    print(f"Converting model to GQA with {args.kv_heads} KV heads...")
    model = convert_gptneox_to_gqa(model, kv_heads=args.kv_heads)
    print("Conversion complete.")

# Define context lengths to test
context_lengths = [256, 512, 1024, 1536, 2048]

results = {
    "context_lengths": context_lengths,
    "prefill_memory": [],
    "prefill_time": [],
    "gen_memory": [],
    "gen_time": []
}

def get_prefilling_stats(model, n_tokens, device):
    torch.cuda.empty_cache()
    torch.cuda.reset_peek_memory_stats()
    
    input_ids = torch.randint(0, model.config.vocab_size, (1, n_tokens)).to(device)
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    with torch.no_grad():
        model(input_ids)
    end_event.record()
    
    torch.cuda.synchronize()
    
    peak_memory = torch.cuda.max_memory_allocated() / 1024**3 # GB
    elapsed_time = start_event.elapsed_time(end_event) / 1000 # seconds
    
    return {"Peak memory usage": peak_memory, "Prefilling time": elapsed_time}

def get_generation_stats(model, n_tokens, max_new_tokens, device):
    torch.cuda.empty_cache()
    torch.cuda.reset_peek_memory_stats()
    
    input_ids = torch.randint(0, model.config.vocab_size, (1, n_tokens)).to(device)
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    with torch.no_grad():
        model.generate(input_ids, max_new_tokens=max_new_tokens, min_new_tokens=max_new_tokens, do_sample=False)
    end_event.record()
    
    torch.cuda.synchronize()
    
    peak_memory = torch.cuda.max_memory_allocated() / 1024**3
    elapsed_time = start_event.elapsed_time(end_event) / 1000
    
    return {"Peak memory usage": peak_memory, "Total time": elapsed_time}

print("Running benchmarks...")
for seq_len in context_lengths:
    print(f"Testing context length: {seq_len}")
    
    try:
        # Prefill
        prefill = get_prefilling_stats(model, n_tokens=seq_len, device=DEVICE)
        results["prefill_memory"].append(prefill["Peak memory usage"])
        results["prefill_time"].append(prefill["Prefilling time"])
        
        # Generation
        gen = get_generation_stats(model, n_tokens=seq_len, max_new_tokens=50, device=DEVICE)
        results["gen_memory"].append(gen["Peak memory usage"])
        results["gen_time"].append(gen["Total time"])
    except Exception as e:
        print(f"Error at context length {seq_len}: {e}")
        results["prefill_memory"].append(0)
        results["prefill_time"].append(0)
        results["gen_memory"].append(0)
        results["gen_time"].append(0)

# Plotting
def plot_metric(x, y, ylabel, title, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker='o', linestyle='-', linewidth=2, label='Model')
    plt.xlabel("Context Length")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(filename)
    # print(f"Saved plot to {filename}") # Reduce clutter
    plt.close()

plots_dir = "plots"
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

suffix = "_gqa" if args.gqa else "_raw"
plot_metric(context_lengths, results["prefill_memory"], "Peak Memory (GB)", "Prefill Peak Memory", os.path.join(plots_dir, f"prefill_memory{suffix}.png"))
plot_metric(context_lengths, results["prefill_time"], "Time (s)", "Prefill Time", os.path.join(plots_dir, f"prefill_time{suffix}.png"))
plot_metric(context_lengths, results["gen_memory"], "Peak Memory (GB)", "Generation Peak Memory", os.path.join(plots_dir, f"gen_memory{suffix}.png"))
plot_metric(context_lengths, results["gen_time"], "Time (s)", "Generation Time", os.path.join(plots_dir, f"gen_time{suffix}.png"))

print("Benchmarks completed.")