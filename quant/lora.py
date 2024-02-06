from diffusers.models import unet_2d_condition as unet
from diffusers.models import embeddings
from diffusers.models.lora import LoRACompatibleLinear
from diffusers.utils import USE_PEFT_BACKEND
from diffusers.models.activations import get_activation
import torch
import torch.nn as nn
from typing import Optional, Union, Tuple, Dict
from torch.nn.modules import Linear

from torch_int.nn.linear import W8A8BFP32OFP32Linear, W8A8B8O8Linear, W8A8B8O8LinearReLU
from torch_int.nn.fused import LayerNormQ
from torch_int.nn.bmm import BMM_S8T_S8N_S8T, BMM_S8T_S8N_F32T


# Since all LoRACompatibleLinear.lora_layer in inference is None, we can treat it as Linear
class INTLoRALayer(torch.nn.Module):
    def __init__(self, in_features, out_features, alpha=1, beta=1):
        self.proxy_model =  W8A8B8O8Linear(in_features, out_features, alpha, beta)

    @staticmethod
    def from_float(model: LoRACompatibleLinear, input_scale, output_scale):
        proxy_model = W8A8B8O8Linear.from_float(model, input_scale, output_scale)
        intmodel = INTLoRALayer(model.in_features, model.out_features)
        intmodel.proxy_model = proxy_model
        return intmodel

    @torch.no_grad()
    def forward(self, x):
        self.proxy_model(x)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        self.bias = self.bias.to(*args, **kwargs)
        return self
