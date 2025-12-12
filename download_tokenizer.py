import requests

url = "https://huggingface.co/EleutherAI/pythia-2.8b/raw/main/tokenizer.json"
output_path = "e:/github/accelerated_inference/pythia-2.8b-local/tokenizer.json"

try:
    response = requests.get(url)
    response.raise_for_status()
    with open(output_path, "wb") as f:
        f.write(response.content)
    print("Downloaded tokenizer.json successfully.")
except Exception as e:
    print(f"Failed to download tokenizer.json: {e}")
