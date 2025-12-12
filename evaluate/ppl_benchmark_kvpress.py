#!/usr/bin/env python
# coding: utf-8

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk
from tqdm import tqdm
import os
import sys
import argparse
# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import KVPress modules (assuming available in path)
try:
    from kvpress import KnormPress, SnapKVPress # Example presses
except ImportError:
    print("Error: kvpress not found.")

parser = argparse.ArgumentParser(description="PPL Benchmark for KVPress")
parser.add_argument("--press", type=str, default="knorm", choices=["knorm", "snapkv"], help="Press method to use")
parser.add_argument("--model_path", type=str, default=r"e:\github\accelerated_inference\pythia-2.8b-local", help="Path to model")
args = parser.parse_args()

MODEL_PATH = args.model_path
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_DIR = "../dataset"

print(f"Loading model from: {MODEL_PATH} on {DEVICE}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(DEVICE)
model.eval()

def calculate_ppl_with_press(model, tokenizer, text, press_class, seq_len=2048, stride=512, device="cuda"):
    encodings = tokenizer(text, return_tensors="pt")
    input_ids_all = encodings.input_ids
    if input_ids_all.size(1) > 100000:
        input_ids_all = input_ids_all[:, :100000]

    nlls = []
    prev_end_loc = 0
    
    # Initialize press
    # Note: Press initialization might need specific args depending on the implementation
    # This assumes a context manager style usage as seen in reference snippets
    
    press_kwargs = {} # Add specific args here if needed (e.g. compression ratio)
    press = press_class(compression_ratio=0.5) # Example default
    
    for begin_loc in tqdm(range(0, input_ids_all.size(1), stride), desc="Calculating PPL"):
        end_loc = min(begin_loc + seq_len, input_ids_all.size(1))
        trg_len = end_loc - prev_end_loc
        
        input_ids = input_ids_all[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        try:
            with torch.no_grad():
                with press(model): # Apply press context
                    outputs = model(input_ids, labels=target_ids)
                    nlls.append(outputs.loss)
        except Exception as e:
            print(f"Error during inference with press: {e}")
            break

        prev_end_loc = end_loc
        if end_loc == input_ids_all.size(1):
            break

    if not nlls:
        return float('nan')

    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl.item()

# Evaluate on Wikitext-2
wikitext_path = os.path.join(DATASET_DIR, "wikitext-2-raw-v1")
if os.path.exists(wikitext_path):
    data = load_from_disk(wikitext_path)
    if "test" in data: data = data["test"]
    text = "\n\n".join(data["text"])
    
    if args.press == "knorm":
        press_cls = KnormPress
    elif args.press == "snapkv":
        press_cls = SnapKVPress # Verify imports first
    else:
        press_cls = KnormPress
        
    print(f"Wikitext PPL with {args.press}: {calculate_ppl_with_press(model, tokenizer, text, press_cls, device=DEVICE):.2f}")
