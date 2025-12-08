from transformers.utils.logging import disable_progress_bar
import transformers
from kvpress import KnormPress
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()
print(f"Loading model: {MODEL_NAME}...")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype="auto", attn_implementation=None).to(DEVICE)

# Define context lengths to test
# Pythia-70M has max position embeddings of 2048
context_lengths = [256, 512, 1024, 1536, 2048]

results = {
    "context_lengths": context_lengths,
    "prefill_memory": [],
    "prefill_time": [],
    "gen_memory": [],
    "gen_time": []
}

# Define a null press (Raw)
from contextlib import contextmanager
@contextmanager
def null_press(model):
    yield
print("Running benchmarks for Raw model...")
for seq_len in context_lengths:
    print(f"Testing context length: {seq_len}")
    
    try:
        # Prefill
        prefill = get_prefilling_stats(model, null_press, n_tokens=seq_len, device=DEVICE)
        results["prefill_memory"].append(prefill["Peak memory usage"])
        results["prefill_time"].append(prefill["Prefilling time"])
        
        # Generation
        # Generate a small number of tokens to measure speed/memory
        gen = get_generation_stats(model, null_press, n_tokens=seq_len, max_new_tokens=50, device=DEVICE)
        results["gen_memory"].append(gen["Peak memory usage"])
        results["gen_time"].append(gen["Total time"])
    except Exception as e:
        print(f"Error at context length {seq_len}: {e}")
        # Append None or previous value to keep lists aligned, or break
        results["prefill_memory"].append(0)
        results["prefill_time"].append(0)
        results["gen_memory"].append(0)
        results["gen_time"].append(0)
# Plotting
def plot_metric(x, y, ylabel, title, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker='o', linestyle='-', linewidth=2, label='Raw')
    plt.xlabel("Context Length")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved plot to {filename}")
    plt.close()
# Create plots directory if not exists
plots_dir = "plots"
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)
plot_metric(context_lengths, results["prefill_memory"], "Peak Memory (GB)", "Prefill Peak Memory", os.path.join(plots_dir, "prefill_memory.png"))
plot_metric(context_lengths, results["prefill_time"], "Time (s)", "Prefill Time", os.path.join(plots_dir, "prefill_time.png"))
plot_metric(context_lengths, results["gen_memory"], "Peak Memory (GB)", "Generation Peak Memory", os.path.join(plots_dir, "gen_memory.png"))
plot_metric(context_lengths, results["gen_time"], "Time (s)", "Generation Time", os.path.join(plots_dir, "gen_time.png"))
print("Benchmarks completed.")