#!/usr/bin/env python
# coding: utf-8

# # Perplexity Evaluation for Pythia-70M
# 
# This notebook evaluates the perplexity of the Pythia-70M model on Wikitext-2 and PG19 datasets.

# In[ ]:


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk
from tqdm import tqdm
import os
import math


# In[ ]:


MODEL_NAME = "EleutherAI/pythia-70m"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_DIR = "../dataset"

print(f"Loading model: {MODEL_NAME} on {DEVICE}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()


# In[ ]:


def calculate_ppl(model, tokenizer, text, seq_len=2048, stride=512, device="cuda"):
    encodings = tokenizer(text, return_tensors="pt")
    max_length = model.config.max_position_embeddings
    seq_len = min(seq_len, max_length)
    
    nlls = []
    prev_end_loc = 0
    
    for begin_loc in tqdm(range(0, encodings.input_ids.size(1), stride), desc="Calculating PPL"):
        end_loc = min(begin_loc + seq_len, encodings.input_ids.size(1))
        trg_len = end_loc - prev_end_loc
        
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            nlls.append(outputs.loss)

        prev_end_loc = end_loc
        if end_loc == encodings.input_ids.size(1):
            break

    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl.item()


# ## Evaluate on Wikitext-2

# In[ ]:


wikitext_path = os.path.join(DATASET_DIR, "wikitext-2-raw-v1")
if os.path.exists(wikitext_path):
    print("Loading Wikitext-2 from local disk...")
    data = load_from_disk(wikitext_path)
    text = "\n\n".join(data["text"])
    print(f"Total text length: {len(text)} chars")
    ppl = calculate_ppl(model, tokenizer, text, device=DEVICE)
    print(f"Wikitext PPL: {ppl:.2f}")
else:
    print(f"Wikitext dataset not found at {wikitext_path}")


# ## Evaluate on PG19 (Subset)

# In[ ]:


import glob
pg19_path = os.path.join(DATASET_DIR, "pg19")
if os.path.exists(pg19_path):
    print("Loading PG19 from local disk...")
    txt_files = glob.glob(os.path.join(pg19_path, "*.txt"))
    if len(txt_files) > 0:
        data = []
        for f in txt_files:
            try:
                with open(f, 'r', encoding='utf-8') as file:
                    data.append({"text": file.read()})
            except Exception as e:
                print(f"Error reading file {f}: {e}")

        if len(data) > 0:
            total_ppl = 0
            count = 0
            # Evaluate on a subset of PG19 to save time, e.g., first 10 books
            num_samples = 10
            print(f"Evaluating on first {num_samples} samples of PG19...")
            
            for i, sample in enumerate(data):
                if i >= num_samples:
                    break
                print(f"Processing sample {i+1}/{num_samples} (Length: {len(sample['text'])} chars)...")
                try:
                    ppl = calculate_ppl(model, tokenizer, sample["text"], device=DEVICE)
                    print(f"Sample {i+1} PPL: {ppl:.2f}")
                    total_ppl += ppl
                    count += 1
                except Exception as e:
                    print(f"Error processing sample {i+1}: {e}")
            
            if count > 0:
                print(f"Average PG19 PPL (over {count} samples): {total_ppl/count:.2f}")
        else:
             print("No valid text files read from PG19.")
    else:
        print("PG19 dataset is empty (no .txt files found).")
else:
    print(f"PG19 dataset not found at {pg19_path}")

