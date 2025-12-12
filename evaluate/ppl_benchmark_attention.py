#!/usr/bin/env python
# coding: utf-8

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk
from tqdm import tqdm
import os
import sys
import argparse
import math

# Add parent directory to path to import attention
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from attention.gqa import convert_gptneox_to_gqa
except ImportError:
    print("Warning: Could not import convert_gptneox_to_gqa. GQA features disabled.")

# Argument Parsing
parser = argparse.ArgumentParser(description="PPL Benchmark for Attention Mechanisms")
parser.add_argument("--gqa", action="store_true", help="Enable GQA conversion")
parser.add_argument("--kv_heads", type=int, default=4, help="Number of KV heads for GQA")
parser.add_argument("--model_path", type=str, default=r"e:\github\accelerated_inference\pythia-2.8b-local", help="Path to model")
args = parser.parse_args()

MODEL_PATH = args.model_path
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_DIR = "../dataset"

print(f"Loading model from: {MODEL_PATH} on {DEVICE}")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(DEVICE)
except Exception as e:
    print(f"Failed to load local model: {e}")
    print("Falling back to EleutherAI/pythia-2.8b (if internet available) or checking path.")
    # Fallback or exit
    # model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-2.8b").to(DEVICE)
    raise e

if args.gqa:
    print(f"Converting model to GQA with {args.kv_heads} KV heads...")
    model = convert_gptneox_to_gqa(model, kv_heads=args.kv_heads)
    print("Conversion complete.")

model.eval()

def calculate_ppl(model, tokenizer, text, seq_len=2048, stride=512, device="cuda"):
    encodings = tokenizer(text, return_tensors="pt")
    max_length = model.config.max_position_embeddings
    seq_len = min(seq_len, max_length)
    
    nlls = []
    prev_end_loc = 0
    
    # Limit processing for speed in benchmark
    input_ids_all = encodings.input_ids
    if input_ids_all.size(1) > 100000:
        input_ids_all = input_ids_all[:, :100000]
    
    for begin_loc in tqdm(range(0, input_ids_all.size(1), stride), desc="Calculating PPL"):
        end_loc = min(begin_loc + seq_len, input_ids_all.size(1))
        trg_len = end_loc - prev_end_loc
        
        input_ids = input_ids_all[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            nlls.append(outputs.loss)

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
    print("Loading Wikitext-2 from local disk...")
    # Logic to load dataset might vary depending on how it was saved
    # Assuming standard 'datasets' save_to_disk format
    try:
        data = load_from_disk(wikitext_path)
        # If datasetDict
        if "test" in data:
            data = data["test"]
        
        text = "\n\n".join(data["text"])
        print(f"Total text length: {len(text)} chars")
        ppl = calculate_ppl(model, tokenizer, text, device=DEVICE)
        print(f"Wikitext PPL: {ppl:.2f}")
    except Exception as e:
        print(f"Error loading/processing Wikitext: {e}")
else:
    print(f"Wikitext dataset not found at {wikitext_path}")
