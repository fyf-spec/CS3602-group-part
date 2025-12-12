import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
import os
import sys
import argparse
import warnings
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from kvpress.presses.benchmark_presses import StreamLLMPress, SnapKVPress

warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()

parser = argparse.ArgumentParser(description="Speed Benchmark for KVPress")
parser.add_argument("--model_path", type=str, default=r"e:\github\accelerated_inference\pythia-2.8b-local", help="Path to model")
args = parser.parse_args()

MODEL_PATH = args.model_path
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading model: {MODEL_PATH}...")
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype="auto", attn_implementation=None).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

context_lengths = [256, 512, 1024, 2048]
presses = {
    "StreamLLM": StreamLLMPress,
    "SnapKV": SnapKVPress
}
results = {
    "context_lengths": context_lengths,
    "gen_time": {k: [] for k in presses},
    "gen_mem": {k: [] for k in presses}
}

def get_stats(model, seq_len, max_new_tokens, press_cls):
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    
    input_ids = torch.randint(0, model.config.vocab_size, (1, seq_len)).to(DEVICE)
    press = press_cls(compression_ratio=0.5)
    
    # Generation Only (Prefill usually similar or pressed? depends on impl)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    with torch.no_grad(), press(model):
        model.generate(input_ids, max_new_tokens=max_new_tokens, min_new_tokens=max_new_tokens, do_sample=False)
    end.record()
    torch.cuda.synchronize()
    gen_time = start.elapsed_time(end) / 1000
    gen_mem = torch.cuda.max_memory_allocated() / 1024**3
    
    return gen_time, gen_mem

print("Running benchmarks...")
for name, press_cls in presses.items():
    print(f"Testing {name}...")
    for seq_len in context_lengths:
        try:
            gt, gm = get_stats(model, seq_len, 50, press_cls)
            results["gen_time"][name].append(gt)
            results["gen_mem"][name].append(gm)
            print(f"  Seq {seq_len}: {gt:.2f}s, {gm:.2f}GB")
        except Exception as e:
            print(f"  Error at {seq_len}: {e}")
            results["gen_time"][name].append(0)
            results["gen_mem"][name].append(0)

# Plotting
plots_dir = "plots_kvpress"
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

def plot_all(metric_dict, ylabel, title, filename):
    plt.figure(figsize=(10, 6))
    for name in presses:
        plt.plot(context_lengths, metric_dict[name], marker='o', label=name)
    plt.xlabel("Context Length")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

plot_all(results["gen_time"], "Time (s)", "Generation Time", os.path.join(plots_dir, "gen_time_kvpress.png"))
plot_all(results["gen_mem"], "Memory (GB)", "Generation Memory", os.path.join(plots_dir, "gen_mem_kvpress.png"))
print("Benchmarks completed.")
