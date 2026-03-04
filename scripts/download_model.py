"""Run once to download and save the model to a local directory."""
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_ID = "AdamLucek/Orpo-Llama-3.2-1B-15k"
# Save under repo root so path is portable
REPO_ROOT = Path(__file__).resolve().parent.parent
SAVE_DIR = REPO_ROOT / "models" / "Orpo-Llama-3.2-1B-15k"

def main():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    print("Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.save_pretrained(SAVE_DIR)
    print("Downloading model (fp16)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.save_pretrained(SAVE_DIR)
    print(f"Done. Model saved to {SAVE_DIR}")

if __name__ == "__main__":
    main()