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

def has_sub_int8_params_weights(model):
    for p in model.parameters():
        if p.dtype in [torch.int4, torch.uint4]:
            return True
    return False

def has_quant_scales_buffers(model):
    for name, buf in model.named_buffers():
        if "scale" in name.lower():
            return True
    return False

def same_layer_count_modules(base_model, quant_model):
    base_layers = sum(1 for _ in base_model.modules())
    quant_layers = sum(1 for _ in quant_model.modules())
    return base_layers == quant_layers

def check_per_channel_scales(model):
    per_channel = 0
    per_tensor = 0

    for module in model.modules():

        if hasattr(module, "weight_scale"):

            scale = module.weight_scale

            if scale.numel() > 1:
                per_channel += 1
            else:
                per_tensor += 1

    return per_channel, per_tensor

def detect_activation_quantization(model):

    count = 0

    for module in model.modules():

        for attr in ["act_scale", "activation_scale", "input_scale"]:
            if hasattr(module, attr):
                count += 1

    return count

def check_activation_dtype(model, tokenizer, prompt):

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    int8_seen = False

    def hook(module, inp, out):
        nonlocal int8_seen

        if isinstance(inp, tuple):
            for x in inp:
                if isinstance(x, torch.Tensor) and x.dtype == torch.int8:
                    int8_seen = True

    handles = []

    for m in model.modules():
        handles.append(m.register_forward_hook(hook))

    model(**inputs)

    for h in handles:
        h.remove()

    return int8_seen



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
    has_sub_int8_params = has_sub_int8_params_weights(quantized_model)
    has_quant_scales = has_quant_scales_buffers(quantized_model)
    same_layer_count = same_layer_count_modules(base_model, quantized_model)

    # check if per-channel scales are used
    per_channel, per_tensor = check_per_channel_scales(quantized_model)

    # check if activation quantization modules exists
    activation_quantization = detect_activation_quantization(quantized_model)
    #Verify activations are actually quantized during forward
    int8_seen = check_activation_dtype(quantized_model, tokenizer, PROMPT)


    metrics = {
        "base_model_size_bytes": int(base_bytes),
        "quantized_model_size_bytes": int(quant_bytes),
        "base_ppl": float(ppl_base),
        "quantized_ppl": float(ppl_quant),
        #"has_non_int8_params": bool(has_non_int8_params),
        "has_sub_int8_params": bool(has_sub_int8_params),
        #"has_quant_scales": bool(has_quant_scales),
        "same_layer_count": bool(same_layer_count),
        #new added metrics
        "per_channel_scales": bool(
            (per_channel + per_tensor) > 0 and per_channel / (per_channel + per_tensor) > 0.5
        ),
        #"activation_quantization_modules_exists": bool(activation_quantization > 0),
        "activation_quantization_during_forward": bool(int8_seen),
    }
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()