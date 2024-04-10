from __future__ import annotations

from diffusers.models.activations import GEGLU
from torch import nn
import torch
from torch.nn import functional as F
from diffusers.models.lora import LoRACompatibleLinear
from diffusers.utils import USE_PEFT_BACKEND

from torch_int.nn.linear import W8A8B8O8Linear

class SiLUQ(nn.Module):
    r"""Applies the Sigmoid Linear Unit (SiLU) function, element-wise.
    The SiLU function is also known as the swish function.

    .. math::
        \text{silu}(x) = x * \sigma(x), \text{where } \sigma(x) \text{ is the logistic sigmoid.}

    .. note::
        See `Gaussian Error Linear Units (GELUs) <https://arxiv.org/abs/1606.08415>`_
        where the SiLU (Sigmoid Linear Unit) was originally coined, and see
        `Sigmoid-Weighted Linear Units for Neural Network Function Approximation
        in Reinforcement Learning <https://arxiv.org/abs/1702.03118>`_ and `Swish:
        a Self-Gated Activation Function <https://arxiv.org/abs/1710.05941v1>`_
        where the SiLU was experimented with later.

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/SiLU.png

    Examples::

        >>> m = nn.SiLU()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """
    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = input.to(dtype=torch.float16)
        return F.silu(input, inplace=self.inplace).to(dtype=torch.int8)

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str
    
    @staticmethod
    def from_float(module: torch.nn.SiLU):
        int_silu = SiLUQ(module.inplace)
        return int_silu



class GEGLUQ(nn.Module):
    r"""
    A [variant](https://arxiv.org/abs/2002.05202) of the gated linear unit activation function.
    """

    def __init__(self):
        super().__init__()
        # linear_cls = LoRACompatibleLinear if not USE_PEFT_BACKEND else nn.Linear
        # self.proj = linear_cls(dim_in, dim_out * 2)

    def gelu(self, gate: torch.Tensor) -> torch.Tensor:
        if gate.device.type != "mps":
            return F.gelu(gate)
        # mps: gelu is not implemented for float16
        return F.gelu(gate.to(dtype=torch.float16)).to(dtype=gate.dtype)

    def forward(self, hidden_states, scale: float = 1.0):
        args = () if USE_PEFT_BACKEND else (scale,)
        hidden_states = hidden_states.to(dtype=self.proj.weight.dtype)
        hidden_states, gate = self.proj(hidden_states, *args).chunk(2, dim=-1)
        gate = gate.to(dtype=torch.float16)
        hidden_states = hidden_states * self.gelu(gate)
        hidden_states = hidden_states.to(dtype=torch.int8)
        return hidden_states
    
    @staticmethod
    def from_float(module: GEGLU) -> GEGLUQ:
        int_geglu = GEGLUQ()
        int_geglu.proj = W8A8B8O8Linear.from_float(module.proj, 1. , 1.)
        return int_geglu

