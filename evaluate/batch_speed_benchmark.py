"""
Batch Speed Benchmark: Compare Baseline vs INT8 vs INT8+LazyUnified

This script runs speed benchmarks for the three main modes and generates
a comparison table and visualization.

Usage:
    python evaluate/batch_speed_benchmark.py
    python evaluate/batch_speed_benchmark.py --seq_lengths 512 1024 2048
"""

import subprocess
import sys
import os
import json
import time
from datetime import datetime

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
EVAL_SCRIPT = os.path.join(SCRIPT_DIR, "eval_speed_benchmark.py")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "speed_int8_comparison")

# Default parameters
DEFAULT_PARAMS = {
    "model_name_or_path": "pythia-2.8b-local",
    "ckpt_dir": "checkpoints/pythia-2.8b-int8",
    "start_size": 4,
    "recent_size": 252,
    "separator_size": 64,
    "heavy_size": 128,
    "local_size": 256,
    "update_interval": 10,
    "num_decode_tokens": 128,
}

# Methods to compare
METHODS = [
    {
        "name": "baseline",
        "description": "FP16 Full KV Cache",
        "extra_args": [],
    },
    {
        "name": "int8_baseline", 
        "description": "INT8 Full KV Cache",
        "extra_args": [
            "--ckpt_dir", DEFAULT_PARAMS["ckpt_dir"],
        ],
    },
    {
        "name": "int8_lazy_unified",
        "description": "INT8 + LazyUnified KV Cache",
        "extra_args": [
            "--ckpt_dir", DEFAULT_PARAMS["ckpt_dir"],
            "--separator_size", str(DEFAULT_PARAMS["separator_size"]),
            "--heavy_size", str(DEFAULT_PARAMS["heavy_size"]),
            "--local_size", str(DEFAULT_PARAMS["local_size"]),
            "--update_interval", str(DEFAULT_PARAMS["update_interval"]),
        ],
    },
]


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Batch speed benchmark comparison")
    parser.add_argument("--model_name_or_path", type=str, default=DEFAULT_PARAMS["model_name_or_path"])
    parser.add_argument("--ckpt_dir", type=str, default=DEFAULT_PARAMS["ckpt_dir"])
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--seq_lengths", type=int, nargs="+", default=[512, 1024, 2048],
                        help="Sequence lengths to benchmark")
    parser.add_argument("--num_decode_tokens", type=int, default=DEFAULT_PARAMS["num_decode_tokens"])
    parser.add_argument("--dry_run", action="store_true", help="Print commands without running")
    return parser.parse_args()


def run_benchmark(method_config, args):
    """Run benchmark for a single method."""
    mode = method_config["name"]
    print(f"\n{'='*70}")
    print(f"Testing: {mode}")
    print(f"Description: {method_config['description']}")
    print(f"{'='*70}")
    
    # Build output path for this method
    method_output_dir = os.path.join(args.output_dir, mode)
    
    # Build command
    cmd = [
        sys.executable, EVAL_SCRIPT,
        "--mode", mode,
        "--model_name_or_path", args.model_name_or_path,
        "--output_dir", method_output_dir,
        "--seq_lengths", *[str(s) for s in args.seq_lengths],
        "--num_decode_tokens", str(args.num_decode_tokens),
        "--start_size", str(DEFAULT_PARAMS["start_size"]),
        "--recent_size", str(DEFAULT_PARAMS["recent_size"]),
    ]
    
    # Add method-specific arguments
    cmd.extend(method_config["extra_args"])
    
    print(f"Command: {' '.join(cmd)}\n")
    
    if args.dry_run:
        return {"mode": mode, "status": "dry_run", "results": []}
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
        )
        elapsed_time = time.time() - start_time
        
        if result.stdout:
            print(result.stdout[-2000:])  # Print last 2000 chars
        if result.returncode != 0 and result.stderr:
            print("STDERR:", result.stderr[-1000:])
        
        # Load results
        result_file = os.path.join(method_output_dir, f"speed_results_{mode}.json")
        if os.path.exists(result_file):
            with open(result_file, "r") as f:
                data = json.load(f)
            return {
                "mode": mode,
                "description": method_config["description"],
                "status": "success",
                "results": data.get("results", []),
                "time_seconds": elapsed_time,
            }
        else:
            return {
                "mode": mode,
                "status": "error",
                "error": "Result file not found",
                "time_seconds": elapsed_time,
            }
            
    except subprocess.TimeoutExpired:
        return {"mode": mode, "status": "timeout", "time_seconds": time.time() - start_time}
    except Exception as e:
        return {"mode": mode, "status": "exception", "error": str(e)}


def generate_comparison_table(all_results, args):
    """Generate comparison table for all modes."""
    lines = []
    lines.append("=" * 100)
    lines.append("SPEED COMPARISON: Baseline vs INT8 vs INT8+LazyUnified")
    lines.append("=" * 100)
    lines.append("")
    
    # Table header
    header = f"{'Seq Length':>12} | "
    for r in all_results:
        if r["status"] == "success":
            header += f"{r['description'][:20]:>22} | "
    lines.append(header)
    lines.append("-" * 100)
    
    # Build table rows
    for seq_len in args.seq_lengths:
        row = f"{seq_len:>12} | "
        for method_result in all_results:
            if method_result["status"] != "success":
                row += f"{'ERROR':>22} | "
                continue
            
            # Find result for this seq_len
            result_for_seq = None
            for r in method_result["results"]:
                if r["seq_length"] == seq_len:
                    result_for_seq = r
                    break
            
            if result_for_seq:
                latency = result_for_seq["avg_decode_latency_ms"]
                tps = result_for_seq["tokens_per_sec"]
                row += f"{latency:>8.2f}ms ({tps:>7.1f}t/s) | "
            else:
                row += f"{'N/A':>22} | "
        
        lines.append(row)
    
    lines.append("-" * 100)
    lines.append("")
    
    return "\n".join(lines)


def save_summary(all_results, args):
    """Save summary to JSON file."""
    summary = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "model": args.model_name_or_path,
            "ckpt_dir": args.ckpt_dir,
            "seq_lengths": args.seq_lengths,
            "num_decode_tokens": args.num_decode_tokens,
        },
        "results": all_results,
    }
    
    summary_path = os.path.join(args.output_dir, "comparison_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    return summary_path


def main():
    args = parse_args()
    
    # Check if INT8 checkpoint exists
    if not os.path.exists(args.ckpt_dir):
        print(f"WARNING: INT8 checkpoint not found at {args.ckpt_dir}")
        print("Generate it with: python -m accelerated_inference.quantization.quantize")
        print("")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 70)
    print("Batch Speed Benchmark: Baseline vs INT8 vs INT8+LazyUnified")
    print("=" * 70)
    print(f"Model: {args.model_name_or_path}")
    print(f"INT8 Checkpoint: {args.ckpt_dir}")
    print(f"Sequence lengths: {args.seq_lengths}")
    print(f"Decode tokens: {args.num_decode_tokens}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 70)
    
    all_results = []
    
    for method in METHODS:
        result = run_benchmark(method, args)
        all_results.append(result)
    
    # Generate comparison
    if not args.dry_run:
        table = generate_comparison_table(all_results, args)
        print("\n" + table)
        
        # Save table to file
        table_path = os.path.join(args.output_dir, "comparison_table.txt")
        with open(table_path, "w") as f:
            f.write(table)
        print(f"\nTable saved to: {table_path}")
        
        # Save summary JSON
        summary_path = save_summary(all_results, args)
        print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
