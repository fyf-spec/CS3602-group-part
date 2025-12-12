import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
import os
import sys
import argparse
import warnings
import matplotlib.pyplot as plt
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from attention.gqa import convert_gptneox_to_gqa

warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()

parser = argparse.ArgumentParser(description="Speed Benchmark for Attention Variants")
parser.add_argument("--model_path", type=str, default=r"e:\github\accelerated_inference\pythia-2.8b-local", help="Path to model")
args = parser.parse_args()

MODEL_PATH = args.model_path
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading model: {MODEL_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# Define variants
# Pythia 2.8B has 32 heads
# MHA: 32 kv_heads
# MQA: 1 kv_head
# GQA-8: 4 kv_heads (32/8 = 4 groups? No, usually GQA-8 means 8 groups or 8 KV heads. If Pythia has 32 heads, 8 groups means 4 heads per group?
# Reference: GQA paper uses 'number of groups' usually? Or 'number of KV heads'?
# `MultiheadGQA` args: `kv_heads`.
# Common settings:
# MHA: kv_heads = 32
# GQA-8: kv_heads = 8
# GQA-2: kv_heads = 2 (or 4? Let's pick 8 and 4)
# MQA: kv_heads = 1
VARIANTS = {
    "MHA": 32,
    "GQA-8": 8,
    "GQA-4": 4, 
    "MQA": 1
}

# Context lengths to test
context_lengths = [256, 512, 1024, 2048]

results = {
    "context_lengths": context_lengths,
    "prefill_speed": {k: [] for k in VARIANTS},
    "gen_speed": {k: [] for k in VARIANTS},
    "prefill_mem": {k: [] for k in VARIANTS},
    "gen_mem": {k: [] for k in VARIANTS},
}

def get_stats(model, seq_len, max_new_tokens):
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    
    input_ids = torch.randint(0, model.config.vocab_size, (1, seq_len)).to(DEVICE)
    
    # Prefill
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    with torch.no_grad():
        model(input_ids)
    end.record()
    torch.cuda.synchronize()
    prefill_time = start.elapsed_time(end) / 1000
    prefill_mem = torch.cuda.max_memory_allocated() / 1024**3
    
    # Generation
    torch.cuda.empty_cache() # Clear cache from prefill
    
    start.record()
    with torch.no_grad():
        model.generate(input_ids, max_new_tokens=max_new_tokens, min_new_tokens=max_new_tokens, do_sample=False)
    end.record()
    torch.cuda.synchronize()
    gen_time = start.elapsed_time(end) / 1000
    gen_mem = torch.cuda.max_memory_allocated() / 1024**3
    
    return prefill_time, prefill_mem, gen_time, gen_mem

for name, kv_heads in VARIANTS.items():
    print(f"Benchmarking {name} (kv_heads={kv_heads})...")
    
    # Reload model to reset simple conversion
    if 'model' in locals():
        del model
    torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype="auto", attn_implementation=None).to(DEVICE)
    
    if kv_heads < 32:
        model = convert_gptneox_to_gqa(model, kv_heads=kv_heads)
    
    for seq_len in context_lengths:
        try:
            pt, pm, gt, gm = get_stats(model, seq_len, 50)
            results["prefill_speed"][name].append(pt)
            results["prefill_mem"][name].append(pm)
            results["gen_speed"][name].append(gt)
            results["gen_mem"][name].append(gm)
            print(f"  Seq {seq_len}: {gt:.2f}s, {gm:.2f}GB")
        except Exception as e:
            print(f"  Error at {seq_len}: {e}")
            results["prefill_speed"][name].append(0)
            results["prefill_mem"][name].append(0)
            results["gen_speed"][name].append(0)
            results["gen_mem"][name].append(0)
    
    # Simple Plotting for this variant (or aggregated later)
    # We will aggregate at the end

# Plotting
plots_dir = "plots_variants"
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

def plot_all(metric_dict, ylabel, title, filename):
    plt.figure(figsize=(10, 6))
    for name in VARIANTS:
        plt.plot(context_lengths, metric_dict[name], marker='o', label=name)
    plt.xlabel("Context Length")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

plot_all(results["prefill_speed"], "Time (s)", "Prefill Time", os.path.join(plots_dir, "prefill_time.png"))
plot_all(results["prefill_mem"], "Memory (GB)", "Prefill Memory", os.path.join(plots_dir, "prefill_mem.png"))
plot_all(results["gen_speed"], "Time (s)", "Generation Time", os.path.join(plots_dir, "gen_time.png"))
plot_all(results["gen_mem"], "Memory (GB)", "Generation Memory", os.path.join(plots_dir, "gen_mem.png"))

print("Benchmarks completed.")
