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

def _is_activation_quant_module(module) -> bool:
    """
    Heuristic detector for activation-quant related modules.
    We avoid requiring int8 tensors to reach nn.Linear directly.
    """
    cls = module.__class__.__name__.lower()
    # Common naming patterns for activation quantizers/wrappers
    name_hits = any(k in cls for k in [
        "actquant", "activationquant", "inputquant", "fakequant", "quantizer"
    ])
    # Common attribute/method patterns
    attr_hits = any(
        hasattr(module, attr) for attr in [
            "act_scale",
            "activation_scale",
            "input_scale",
            "activation_quantizer",
            "act_quant",
            "input_quant",
            "quantize_activation",
            "quantize_input",
        ]
    )
    return name_hits or attr_hits
def detect_activation_quantization(model):
    """Count activation-quant modules by robust heuristic."""
    mods = [m for m in model.modules() if _is_activation_quant_module(m)]
    return len(mods)

def check_activation_quantization_during_forward(model, tokenizer, prompt):
    """
    Verify activation-quant modules are actually executed in forward pass.
    Returns: (fired_modules, total_detected_modules)
    """
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    actq_modules = [m for m in model.modules() if _is_activation_quant_module(m)]
    total = len(actq_modules)
    if total == 0:
        return 0, 0
    fired = set()
    handles = []
    def hook(module, inp, out):
        fired.add(id(module))
    for m in actq_modules:
        handles.append(m.register_forward_hook(hook))
    try:
        with torch.no_grad():
            model(**inputs)
    finally:
        for h in handles:
            h.remove()
    return len(fired), total

def detect_int8_tensor_during_forward(model, tokenizer, prompt):
    """
    Detect if any tensor with dtype int8 appears anywhere during forward.
    Returns: bool
    """
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    seen_int8 = False
    handles = []

    def check_tensor(obj):
        nonlocal seen_int8
        if isinstance(obj, torch.Tensor):
            if obj.dtype == torch.int8:
                seen_int8 = True

    def recursive_check(obj):
        if seen_int8:
            return

        if isinstance(obj, torch.Tensor):
            check_tensor(obj)

        elif isinstance(obj, (list, tuple)):
            for item in obj:
                recursive_check(item)

        elif isinstance(obj, dict):
            for v in obj.values():
                recursive_check(v)

    def hook(module, inp, out):
        nonlocal seen_int8
        if seen_int8:
            return

        recursive_check(inp)
        recursive_check(out)

    for m in model.modules():
        handles.append(m.register_forward_hook(hook))

    try:
        with torch.no_grad():
            model(**inputs)
    finally:
        for h in handles:
            h.remove()

    return seen_int8



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

    #Verify activations are actually quantized during forward
    #seen_int8 = detect_int8_tensor_during_forward(quantized_model, tokenizer, PROMPT)
    actq_fired, actq_total = check_activation_quantization_during_forward(
        quantized_model, tokenizer, PROMPT
    )


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
        #"detected_int8_tensor_during_forward": bool(seen_int8),
        #"activation_quantization_modules_exists": bool(activation_quantization > 0),
        "activation_quantization_during_forward": bool(
            actq_total > 0 and actq_fired / actq_total > 0.5
        ),
    }
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()