"""
Ablation Study for LUCID-Q: INT8 + Lazy Unified KV Cache

This script runs parameter ablation experiments:
1. Update interval ablation: m = 5, 10, 20, 50
2. H2O retention ratio ablation: ratio = 0.1, 0.2, 0.3, 0.5

Experiment Setup:
- Mode: int8_lazy_unified (LUCID-Q)
- Prefill length: 128 tokens (fixed)
- Decode length: 2048 tokens (uses max decode_length)

Output:
- LaTeX three-line tables for NeurIPS paper
- Publication-quality matplotlib visualizations
- JSON result files

Usage:
    python evaluate/ablation_study.py --dry_run
    python evaluate/ablation_study.py
    python evaluate/ablation_study.py --skip_ratio  # Only interval ablation
"""

import sys
import os
import subprocess
import json
import argparse
from datetime import datetime
from tqdm import tqdm

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

EVAL_SCRIPT = os.path.join(SCRIPT_DIR, "eval_speed_benchmark.py")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "ablation_study")

# Default parameters
DEFAULT_PARAMS = {
    "model_name_or_path": "pythia-2.8b-local",
    "ckpt_dir": "checkpoints/pythia-2.8b-int8",
    "prefill_len": 128,
    "start_size": 4,
    "recent_size": 252,
    "separator_size": 64,
    "heavy_size": 128,
    "local_size": 256,
    "update_interval": 10,
}

# Decode lengths to test
DECODE_LENGTHS = [256, 512, 1024, 2048]

# =============================================================================
# Experiment Configurations
# =============================================================================

# Main ablation: 3 methods
MAIN_ABLATION = [
    {
        "mode": "baseline",
        "label": "FP16 Baseline",
        "int8": False,
        "extra_args": [],
    },
    {
        "mode": "int8_baseline",
        "label": "INT8 Baseline", 
        "int8": True,
        "extra_args": [],
    },
    {
        "mode": "int8_lazy_unified",
        "label": "LUCID-Q (Ours)",
        "int8": True,
        "extra_args": [
            "--separator_size", str(DEFAULT_PARAMS["separator_size"]),
            "--heavy_size", str(DEFAULT_PARAMS["heavy_size"]),
            "--local_size", str(DEFAULT_PARAMS["local_size"]),
            "--update_interval", str(DEFAULT_PARAMS["update_interval"]),
        ],
    },
]

# Update interval ablation
UPDATE_INTERVALS = [5, 10, 20, 50]

# H2O retention ratio ablation
HEAVY_RATIOS = [0.1, 0.2, 0.3, 0.5]


def parse_args():
    parser = argparse.ArgumentParser(description="LUCID-Q Ablation Study")
    parser.add_argument("--model_name_or_path", type=str, default=DEFAULT_PARAMS["model_name_or_path"])
    parser.add_argument("--ckpt_dir", type=str, default=DEFAULT_PARAMS["ckpt_dir"])
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--prefill_len", type=int, default=DEFAULT_PARAMS["prefill_len"],
                        help="Prefill length (fixed at 128)")
    parser.add_argument("--decode_lengths", type=int, nargs="+", default=DECODE_LENGTHS,
                        help="Decode lengths to benchmark: 256, 512, 1024, 2048")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without running")
    parser.add_argument("--skip_interval", action="store_true", help="Skip update interval ablation")
    parser.add_argument("--skip_ratio", action="store_true", help="Skip H2O ratio ablation")
    return parser.parse_args()


def run_single_benchmark(mode, extra_args, args, output_subdir, decode_lengths=None):
    """Run a single benchmark configuration across multiple decode lengths."""
    if decode_lengths is None:
        decode_lengths = args.decode_lengths
    
    method_output_dir = os.path.join(args.output_dir, output_subdir)
    os.makedirs(method_output_dir, exist_ok=True)
    
    # Build decode_lengths argument
    seq_lengths_str = " ".join(str(d) for d in decode_lengths)
    
    cmd = [
        sys.executable, EVAL_SCRIPT,
        "--mode", mode,
        "--model_name_or_path", args.model_name_or_path,
        "--output_dir", method_output_dir,
        "--prefill_len", str(args.prefill_len),
        "--decode_lengths", *[str(d) for d in decode_lengths],
        "--start_size", str(DEFAULT_PARAMS["start_size"]),
        "--recent_size", str(DEFAULT_PARAMS["recent_size"]),
    ]
    
    # Add INT8 checkpoint if needed
    if mode.startswith("int8"):
        cmd.extend(["--ckpt_dir", args.ckpt_dir])
    
    # Add extra arguments
    cmd.extend(extra_args)
    
    print(f"  Command: {' '.join(cmd)}")
    
    if args.dry_run:
        return {"status": "dry_run"}
    
    try:
        # Run subprocess with real-time output (don't capture)
        print(f"  [Running benchmark...]")
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, timeout=7200)
        
        if result.returncode != 0:
            print(f"  [ERROR] Benchmark failed with code {result.returncode}")
        
        # Load results from JSON file
        result_file = os.path.join(method_output_dir, f"speed_results_{mode}.json")
        if os.path.exists(result_file):
            with open(result_file, "r") as f:
                data = json.load(f)
            return {"status": "success", "results": data.get("results", [])}
        else:
            return {"status": "error", "error": "Result file not found"}
    except subprocess.TimeoutExpired:
        return {"status": "error", "error": "Timeout after 2 hours"}
    except Exception as e:
        return {"status": "error", "error": str(e)}



def run_interval_ablation(args):
    """Run update interval ablation (using largest decode length)."""
    print("\n" + "="*70)
    print("INTERVAL ABLATION: m = 5, 10, 20, 50")
    print(f"Using decode length: {max(args.decode_lengths)}")
    print("="*70)
    
    results = []
    for interval in tqdm(UPDATE_INTERVALS, desc="Interval Ablation"):
        tqdm.write(f"  → update_interval = {interval}")
        extra_args = [
            "--separator_size", str(DEFAULT_PARAMS["separator_size"]),
            "--heavy_size", str(DEFAULT_PARAMS["heavy_size"]),
            "--local_size", str(DEFAULT_PARAMS["local_size"]),
            "--update_interval", str(interval),
        ]
        result = run_single_benchmark(
            "int8_lazy_unified",
            extra_args,
            args,
            f"interval/m_{interval}",
            decode_lengths=[max(args.decode_lengths)]
        )
        result["interval"] = interval
        results.append(result)
    
    return results


def run_ratio_ablation(args):
    """Run H2O retention ratio ablation (using largest decode length)."""
    print("\n" + "="*70)
    print("RATIO ABLATION: heavy_ratio = 0.1, 0.2, 0.3, 0.5")
    print(f"Using decode length: {max(args.decode_lengths)}")
    print("="*70)
    
    results = []
    for ratio in tqdm(HEAVY_RATIOS, desc="Ratio Ablation"):
        tqdm.write(f"  → heavy_ratio = {ratio}")
        extra_args = [
            "--separator_size", str(DEFAULT_PARAMS["separator_size"]),
            "--heavy_ratio", str(ratio),
            "--local_size", str(DEFAULT_PARAMS["local_size"]),
            "--update_interval", str(DEFAULT_PARAMS["update_interval"]),
        ]
        result = run_single_benchmark(
            "int8_lazy_unified",
            extra_args,
            args,
            f"ratio/r_{ratio}",
            decode_lengths=[max(args.decode_lengths)]
        )
        result["ratio"] = ratio
        results.append(result)
    
    return results


# =============================================================================
# LaTeX Table Generation (Three-Line Tables)
# =============================================================================

def generate_main_table(results, decode_lengths):
    """
    Generate LaTeX three-line table for main ablation.
    Shows: Latency (ms), Throughput (tok/s), Peak GPU Memory (MB) for each decode length.
    """
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Ablation study on Pythia-2.8B. Prefill=128 tokens. Latency (ms) / Throughput (tok/s) / Peak Memory (MB).}",
        r"\label{tab:ablation_main}",
        r"\small",
    ]
    
    # Build column spec: Method + one column per decode length
    col_spec = "l" + "c" * len(decode_lengths)
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")
    
    # Header row with decode lengths
    header = "Method"
    for dl in decode_lengths:
        header += f" & Decode={dl}"
    header += r" \\"
    lines.append(header)
    lines.append(r"\midrule")
    
    # Data rows: each cell shows "latency / throughput / memory"
    for r in results:
        if r["status"] != "success":
            continue
        label = r["config"]["label"]
        row = label
        
        for dl in decode_lengths:
            # Find result for this decode length
            data = None
            for res in r["results"]:
                if res.get("seq_length") == dl:
                    data = res
                    break
            
            if data:
                latency = data.get("avg_decode_latency_ms", 0)
                throughput = data.get("tokens_per_sec", 0)
                memory = data.get("peak_memory_mb", 0)
                
                if "LUCID-Q" in label:
                    row += f" & \\textbf{{{latency:.1f} / {throughput:.0f} / {memory:.0f}}}"
                else:
                    row += f" & {latency:.1f} / {throughput:.0f} / {memory:.0f}"
            else:
                row += " & --"
        
        row += r" \\"
        lines.append(row)
    
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    
    return "\n".join(lines)


def generate_interval_table(results):
    """Generate LaTeX table for update interval ablation."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Effect of update interval $m$ on LUCID-Q performance.}",
        r"\label{tab:ablation_interval}",
        r"\begin{tabular}{cccc}",
        r"\toprule",
        r"Update Interval ($m$) & Latency (ms) & Throughput (tok/s) & Peak Memory (MB) \\",
        r"\midrule",
    ]
    
    for r in results:
        if r["status"] != "success":
            continue
        interval = r["interval"]
        data = r["results"][0] if r["results"] else {}
        latency = data.get("avg_decode_latency_ms", 0)
        throughput = data.get("tokens_per_sec", 0)
        memory = data.get("peak_memory_mb", 0)
        lines.append(f"{interval} & {latency:.2f} & {throughput:.1f} & {memory:.0f} \\\\")
    
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    
    return "\n".join(lines)


def generate_ratio_table(results):
    """Generate LaTeX table for H2O ratio ablation."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Effect of heavy hitter retention ratio on LUCID-Q.}",
        r"\label{tab:ablation_ratio}",
        r"\begin{tabular}{ccccc}",
        r"\toprule",
        r"Retention Ratio & Cache Size & Latency (ms) & Throughput (tok/s) & Memory (MB) \\",
        r"\midrule",
    ]
    
    for r in results:
        if r["status"] != "success":
            continue
        ratio = r["ratio"]
        data = r["results"][0] if r["results"] else {}
        latency = data.get("avg_decode_latency_ms", 0)
        throughput = data.get("tokens_per_sec", 0)
        memory = data.get("peak_memory_mb", 0)
        cache_size = data.get("kv_cache_size", 0)
        lines.append(f"{ratio:.1f} & {cache_size} & {latency:.2f} & {throughput:.1f} & {memory:.0f} \\\\")
    
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    
    return "\n".join(lines)


# =============================================================================
# Visualization
# =============================================================================

def generate_plots(main_results, interval_results, ratio_results, decode_lengths, output_dir):
    """Generate matplotlib visualizations."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not available, skipping plot generation")
        return
    
    # Set style for publication
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'serif',
        'axes.labelsize': 13,
        'axes.titlesize': 13,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 10,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
    })
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # Plot 1: Main ablation - decode length vs latency (line plot)
    ax1 = axes[0]
    colors = {"FP16 Baseline": "#808080", "INT8 Baseline": "#4CAF50", "LUCID-Q (Ours)": "#2196F3"}
    markers = {"FP16 Baseline": "o", "INT8 Baseline": "s", "LUCID-Q (Ours)": "^"}
    
    if main_results:
        for r in main_results:
            if r["status"] != "success":
                continue
            label = r["config"]["label"]
            x_vals = []
            y_vals = []
            for res in r["results"]:
                x_vals.append(res["seq_length"])
                y_vals.append(res["avg_decode_latency_ms"])
            
            ax1.plot(x_vals, y_vals, marker=markers.get(label, "o"), 
                    color=colors.get(label, "#000000"), linewidth=2, markersize=8, label=label)
        
        ax1.set_xlabel("Decode Length (tokens)")
        ax1.set_ylabel("Latency (ms)")
        ax1.set_title("(a) Main Ablation")
        ax1.legend(loc="upper left")
        ax1.set_xticks(decode_lengths)
    
    # Plot 2: Update interval effect
    ax2 = axes[1]
    if interval_results:
        intervals = [r["interval"] for r in interval_results if r["status"] == "success"]
        latencies = [r["results"][0]["avg_decode_latency_ms"] for r in interval_results if r["status"] == "success"]
        throughputs = [r["results"][0]["tokens_per_sec"] for r in interval_results if r["status"] == "success"]
        
        ax2.plot(intervals, latencies, 'o-', color="#2196F3", linewidth=2, markersize=8)
        ax2.set_xlabel("Update Interval (m)")
        ax2.set_ylabel("Latency (ms)", color="#2196F3")
        ax2.tick_params(axis='y', labelcolor="#2196F3")
        ax2.set_title("(b) Update Interval Effect")
        ax2.set_xticks(intervals)
        
        ax2_twin = ax2.twinx()
        ax2_twin.plot(intervals, throughputs, 's--', color="#FF5722", linewidth=2, markersize=8)
        ax2_twin.set_ylabel("Throughput (tok/s)", color="#FF5722")
        ax2_twin.tick_params(axis='y', labelcolor="#FF5722")
    
    # Plot 3: Ratio ablation
    ax3 = axes[2]
    if ratio_results:
        ratios = [r["ratio"] for r in ratio_results if r["status"] == "success"]
        latencies = [r["results"][0]["avg_decode_latency_ms"] for r in ratio_results if r["status"] == "success"]
        cache_sizes = [r["results"][0]["kv_cache_size"] for r in ratio_results if r["status"] == "success"]
        
        ax3.plot(ratios, latencies, 'o-', color="#9C27B0", linewidth=2, markersize=8)
        ax3.set_xlabel("H2O Retention Ratio")
        ax3.set_ylabel("Latency (ms)")
        ax3.set_title("(c) Retention Ratio Effect")
        
        # Add cache size annotations
        for ratio, lat, cache in zip(ratios, latencies, cache_sizes):
            ax3.annotate(f'cache={cache}', xy=(ratio, lat), xytext=(5, 5), 
                        textcoords="offset points", fontsize=8, color="gray")
    
    plt.tight_layout()
    
    # Save plots
    plot_path = os.path.join(output_dir, "ablation_plots.pdf")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.savefig(plot_path.replace(".pdf", ".png"), dpi=150, bbox_inches='tight')
    print(f"\nPlots saved to {plot_path}")
    plt.close()


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*70)
    print("LUCID-Q Ablation Study")
    print("="*70)
    print(f"Model: {args.model_name_or_path}")
    print(f"INT8 Checkpoint: {args.ckpt_dir}")
    print(f"Prefill length: {args.prefill_len} (fixed)")
    print(f"Decode lengths: {args.decode_lengths}")
    print(f"Output directory: {args.output_dir}")
    print(f"Dry run: {args.dry_run}")
    print("="*70)
    
    all_results = {}
    
    # Run experiments (only interval and ratio ablations by default)
    if not args.skip_ratio:
        all_results["ratio"] = run_ratio_ablation(args)

    if not args.skip_interval:
        all_results["interval"] = run_interval_ablation(args)
    
    
    
    if args.dry_run:
        print("\n[DRY RUN] No experiments executed.")
        return
    
    # Generate LaTeX tables
    print("\n" + "="*70)
    print("GENERATING LATEX TABLES")
    print("="*70)
    
    tables_dir = os.path.join(args.output_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)
    
    if "main" in all_results:
        table = generate_main_table(all_results["main"], args.decode_lengths)
        table_path = os.path.join(tables_dir, "ablation_main.tex")
        with open(table_path, "w") as f:
            f.write(table)
        print(f"Main table saved to {table_path}")
        print("\n" + table + "\n")
    
    if "interval" in all_results:
        table = generate_interval_table(all_results["interval"])
        table_path = os.path.join(tables_dir, "ablation_interval.tex")
        with open(table_path, "w") as f:
            f.write(table)
        print(f"Interval table saved to {table_path}")
        print("\n" + table + "\n")
    
    if "ratio" in all_results:
        table = generate_ratio_table(all_results["ratio"])
        table_path = os.path.join(tables_dir, "ablation_ratio.tex")
        with open(table_path, "w") as f:
            f.write(table)
        print(f"Ratio table saved to {table_path}")
        print("\n" + table + "\n")
    
    # Generate plots
    generate_plots(
        all_results.get("main", []),
        all_results.get("interval", []),
        all_results.get("ratio", []),
        args.decode_lengths,
        args.output_dir
    )
    
    # Save all results to JSON
    summary = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "model": args.model_name_or_path,
            "prefill_len": args.prefill_len,
            "decode_lengths": args.decode_lengths,
        },
        "results": all_results,
    }
    
    summary_path = os.path.join(args.output_dir, "ablation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")
    
    print("\n" + "="*70)
    print("ABLATION STUDY COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
