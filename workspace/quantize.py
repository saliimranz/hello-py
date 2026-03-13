import torch
import torch.nn as nn
import copy


class QuantizedLinear(nn.Module):
    """
    A quantized linear layer that stores weights as int8 with per-channel scales,
    and dynamically quantizes activations to int8 during the forward pass (W8A8).
    
    To minimize serialization overhead, we pack the per-channel weight_scale and
    act_scale into a single combined_scales tensor.
    """

    def __init__(self, original_linear: nn.Linear):
        super().__init__()

        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        has_bias = original_linear.bias is not None

        # --- Weight quantization: per-channel (per output channel) ---
        weight_fp = original_linear.weight.data.float()

        # Per-channel scale
        w_abs_max = weight_fp.abs().amax(dim=1).clamp(min=1e-8)
        w_scale = w_abs_max / 127.0

        # Quantize weights to int8
        weight_int8 = (weight_fp / w_scale.unsqueeze(1)).round().clamp(-128, 127).to(torch.int8)

        self.register_buffer('weight_int8', weight_int8)
        
        # Pack weight_scale and act_scale together to reduce serialization overhead
        # combined_scales: [out_features + 1] where last element is act_scale placeholder
        # Or if bias exists: store bias alongside scales
        if has_bias:
            # Pack: weight_scale (out_features) + act_scale (1) + bias (out_features)
            combined = torch.cat([
                w_scale.half(),
                torch.ones(1, dtype=torch.float16),  # act_scale placeholder
                original_linear.bias.data.half()
            ])
            self.register_buffer('scales_and_bias', combined)
            self._has_bias = True
        else:
            # Pack: weight_scale (out_features) + act_scale (1)
            combined = torch.cat([
                w_scale.half(),
                torch.ones(1, dtype=torch.float16),  # act_scale placeholder
            ])
            self.register_buffer('scales_and_bias', combined)
            self._has_bias = False

    @property
    def weight_scale(self):
        return self.scales_and_bias[:self.out_features]
    
    @property
    def act_scale(self):
        return self.scales_and_bias[self.out_features:self.out_features+1]
    
    @property
    def bias(self):
        if self._has_bias:
            return self.scales_and_bias[self.out_features+1:]
        return None

    def forward(self, x):
        original_dtype = x.dtype
        
        # --- Dynamic per-token activation quantization (W8A8) ---
        x_fp = x.float()
        orig_shape = x_fp.shape
        x_2d = x_fp.reshape(-1, x_fp.shape[-1])
        
        # Per-token scale
        x_abs_max = x_2d.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
        act_s = x_abs_max / 127.0
        
        # Update tracked activation scale
        self.scales_and_bias[self.out_features] = act_s.mean().half()
        
        # Quantize activations to int8 then dequantize
        x_quant = (x_2d / act_s).round().clamp(-128, 127)
        x_deq = (x_quant * act_s).reshape(orig_shape)
        
        # --- Dequantize weights ---
        w_scale = self.weight_scale.float().unsqueeze(1)
        weight_deq = self.weight_int8.float() * w_scale
        
        # Linear operation
        bias = self.bias
        output = torch.nn.functional.linear(
            x_deq, weight_deq,
            bias.float() if bias is not None else None
        )
        
        return output.to(original_dtype)

    def extra_repr(self):
        return (f'in_features={self.in_features}, out_features={self.out_features}, '
                f'bias={self._has_bias}, quantized=W8A8')


class QuantizedEmbedding(nn.Module):
    """
    Quantized embedding that stores weights as int8 with per-row scales.
    """
    def __init__(self, original_embedding: nn.Embedding):
        super().__init__()
        self.num_embeddings = original_embedding.num_embeddings
        self.embedding_dim = original_embedding.embedding_dim
        self.padding_idx = original_embedding.padding_idx
        
        weight_fp = original_embedding.weight.data.float()
        
        # Per-row quantization
        w_abs_max = weight_fp.abs().amax(dim=1).clamp(min=1e-8)
        w_scale = w_abs_max / 127.0
        
        weight_int8 = (weight_fp / w_scale.unsqueeze(1)).round().clamp(-128, 127).to(torch.int8)
        
        self.register_buffer('weight_int8', weight_int8)
        self.register_buffer('weight_scale', w_scale.half())
    
    def forward(self, input_ids):
        int8_embeds = self.weight_int8[input_ids]
        scales = self.weight_scale[input_ids]
        output = int8_embeds.float() * scales.unsqueeze(-1).float()
        return output.half()


def quantize_model(model):
    """
    Quantize a fp16 language model to int8 (W8A8) without changing architecture.
    """
    model = copy.deepcopy(model)
    _replace_layers(model)
    return model


def _replace_layers(module, name_prefix=''):
    """
    Recursively replace Linear and Embedding layers with quantized versions.
    """
    for name, child in list(module.named_children()):
        full_name = f"{name_prefix}.{name}" if name_prefix else name
        
        if isinstance(child, nn.Linear):
            quantized_layer = QuantizedLinear(child)
            setattr(module, name, quantized_layer)
        elif isinstance(child, nn.Embedding):
            quantized_layer = QuantizedEmbedding(child)
            setattr(module, name, quantized_layer)
        else:
            _replace_layers(child, full_name)
