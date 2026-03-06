import json
import math
from pathlib import Path
import sys
import torch
import copy
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
    )
    return tokenizer, model


def model_num_bytes(model):
    total = 0

    for p in model.parameters():
        total += p.numel() * p.element_size()

    for b in model.buffers():
        total += b.numel() * b.element_size()

    return total


def perplexity(model, tokenizer, prompt: str) -> float:
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs, labels=inputs["input_ids"])
    loss = out.loss.item()
    return math.exp(loss)

def linear_layers_have_fp16(model):
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            if module.weight.dtype == torch.float16:
                return True
    return False

def linear_layers_have_non_int8_weights(model):
    for module in model.modules():
        if hasattr(module, "weight_quant"):
            if module.weight_quant.dtype != torch.int8:
                return True
    return False



def main():
    tokenizer, base_model = load_base_model()
    base_bytes = model_num_bytes(base_model)
    ppl_base = perplexity(base_model, tokenizer, PROMPT)

    quantized_model = copy.deepcopy(base_model)
    quantized_model = quantize.quantize_model(quantized_model)
    quant_bytes = model_num_bytes(quantized_model)
    ppl_quant = perplexity(quantized_model, tokenizer, PROMPT)

    # any float16 params left in quantized model?
    has_non_int8_params = linear_layers_have_non_int8_weights(quantized_model)

    metrics = {
        "base_model_size_bytes": int(base_bytes),
        "quantized_model_size_bytes": int(quant_bytes),
        "base_ppl": float(ppl_base),
        "quantized_ppl": float(ppl_quant),
        "has_non_int8_params": bool(has_non_int8_params),
    }
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()