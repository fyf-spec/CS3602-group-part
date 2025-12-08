import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))

from accelerated_inference.kvpress.presses.knorm_press import KnormPress

MODEL_NAME = "EleutherAI/pythia-70m"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading model: {MODEL_NAME} on {DEVICE}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

press = KnormPress(compression_ratio=0.2)

input_ids = tokenizer("Hello, this is a test.", return_tensors="pt").input_ids.to(DEVICE)

print("Running forward pass with KnormPress...")
try:
    with press(model):
        outputs = model(input_ids, use_cache=True)
    print("Forward pass successful!")
except Exception as e:
    print(f"Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
