import json
import math
from pathlib import Path
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_DIR = REPO_ROOT / "workspace"
sys.path.insert(0, str(WORKSPACE_DIR))

import quantize  # workspace/quantize.py, must define quantize_model(model)

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = str(REPO_ROOT / "models" / "Orpo-Llama-3.2-1B-15k")
PROMPT = "What is a language model?"


def load_base_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    return tokenizer, model


def model_num_bytes(model: torch.nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in model.parameters())


def perplexity(model, tokenizer, prompt: str) -> float:
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs, labels=inputs["input_ids"])
    loss = out.loss.item()
    return math.exp(loss)


def main():
    tokenizer, base_model = load_base_model()
    quantized_model = quantize.quantize_model(base_model)

    base_bytes = model_num_bytes(base_model)
    quant_bytes = model_num_bytes(quantized_model)

    ppl_base = perplexity(base_model, tokenizer, PROMPT)
    ppl_quant = perplexity(quantized_model, tokenizer, PROMPT)

    # any float16 params left in quantized model?
    has_fp16_params = any(
        p.dtype == torch.float16 for p in quantized_model.parameters()
    )

    metrics = {
        "base_model_size_bytes": int(base_bytes),
        "quantized_model_size_bytes": int(quant_bytes),
        "base_ppl": float(ppl_base),
        "quantized_ppl": float(ppl_quant),
        "has_fp16_params": bool(has_fp16_params),
    }
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()