from __future__ import annotations
from diffusers.models.attention_processor import (
    AttnProcessor,
    AttnProcessor2_0, 
    Attention, 
    SlicedAttnAddedKVProcessor, 
    SlicedAttnProcessor, 
    AttnAddedKVProcessor,
    SpatialNorm
    )
from diffusers.models.lora import LoRACompatibleLinear, LoRALinearLayer
from diffusers.models.activations import GEGLU
from diffusers.models.attention import FeedForward, BasicTransformerBlock
import torch
from torch.nn import functional as F
from typing import Optional
from torch import nn
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.utils import logging

from diffusers.utils import USE_PEFT_BACKEND

from .activation import GEGLUQ
from torch_int.nn.linear import W8A8B8O8Linear
from torch_int.nn.fused import LayerNormQ
from torch_int.nn.attention import W8A8B8O8Attention

from typing import cast, Union, Dict, Any

logger = logging.get_logger(__name__) 

class INTFeedForward(nn.Module):
    r"""
    A feed-forward layer.

    Parameters:
        dim (`int`): The number of channels in the input.
        dim_out (`int`, *optional*): The number of channels in the output. If not given, defaults to `dim`.
        mult (`int`, *optional*, defaults to 4): The multiplier to use for the hidden dimension.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        final_dropout (`bool` *optional*, defaults to False): Apply a final dropout.
    """

    def __init__(
        self,
        # dim: int,
        # dim_out: Optional[int] = None,
        # mult: int = 4,
        # dropout: float = 0.0,
        # activation_fn: str = "geglu",
        # final_dropout: bool = False,
    ):
        super().__init__()
        # inner_dim = int(dim * mult)
        # dim_out = dim_out if dim_out is not None else dim
        # linear_cls = LoRACompatibleLinear if not USE_PEFT_BACKEND else nn.Linear

        # if activation_fn == "gelu":
        #     act_fn = GELU(dim, inner_dim)
        # if activation_fn == "gelu-approximate":
        #     act_fn = GELU(dim, inner_dim, approximate="tanh")
        # elif activation_fn == "geglu":
        #     act_fn = GEGLU(dim, inner_dim)
        # elif activation_fn == "geglu-approximate":
        #     act_fn = ApproximateGELU(dim, inner_dim)

        # assert activation_fn == "geglu", "Only Support GEGLU now"
        # act_fn = GEGLU(dim, inner_dim)

        # self.net = nn.ModuleList([])
        # # project in
        # self.net.append(act_fn)
        # # project dropout
        # self.net.append(nn.Dropout(dropout))
        # # project out
        # self.net.append(linear_cls(inner_dim, dim_out))
        # # FF as used in Vision Transformer, MLP-Mixer, etc. have a final dropout
        # if final_dropout:
        #     self.net.append(nn.Dropout(dropout))

    def forward(self, hidden_states: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        # compatible_cls = (GEGLU,) if USE_PEFT_BACKEND else (GEGLU, LoRACompatibleLinear)
        # for module in self.net:
        #     if isinstance(module, compatible_cls):
        #         hidden_states = module(hidden_states, scale)
        #     else:
        #         hidden_states = module(hidden_states)
        # return hidden_states
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states
    
    @staticmethod
    def from_float(module: FeedForward):
        int_ff = INTFeedForward()

        int_ff.net = nn.ModuleList([
            GEGLUQ.from_float(cast(GEGLU, module.net[0])),
            # omit dropout module.net[1]
            W8A8B8O8Linear.from_float(cast(Union[nn.Linear, LoRACompatibleLinear], module.net[2]), 1., 1.)
        ])

        return int_ff


@maybe_allow_in_graph
class INTBasicTransformerBlock(BasicTransformerBlock):
    @staticmethod
    def from_float(module: BasicTransformerBlock) -> INTBasicTransformerBlock:
        if hasattr(module, "norm1"):
            module.norm1 = LayerNormQ.from_float(cast(torch.nn.LayerNorm, module.norm1), 1.)
        if hasattr(module, "norm2"):
            module.norm2 = LayerNormQ.from_float(cast(torch.nn.LayerNorm, module.norm2), 1.)
        if hasattr(module, "norm3"):
            module.norm3 = LayerNormQ.from_float(cast(torch.nn.LayerNorm, module.norm3), 1.)
        module.attn1 = W8A8B8O8Attention.from_float(module.attn1)
        if module.attn2 is not None:
            module.attn2 = W8A8B8O8Attention.from_float(module.attn2)
        module.ff = INTFeedForward.from_float(module.ff)

        module.__class__ = INTBasicTransformerBlock
        module = cast(INTBasicTransformerBlock,  module)
        return module