from diffusers.models import unet_2d_condition as unet
from diffusers.models import embeddings
from diffusers.models.lora import LoRACompatibleLinear, LoRACompatibleConv
from diffusers.utils import USE_PEFT_BACKEND
from diffusers.models.activations import get_activation
import torch
import torch.nn as nn
from typing import Optional, Union, Tuple, Dict

from torch_int.nn.linear import W8A8BFP32OFP32Linear, W8A8B8O8Linear, W8A8B8O8LinearReLU
from torch_int.nn.conv import TestW8A8B8O8Conv2D16
from torch_int.nn.fused import LayerNormQ
from torch_int.nn.bmm import BMM_S8T_S8N_S8T, BMM_S8T_S8N_F32T



def replace_unet_conv(model):
    for name, module in model.named_children():
        # 如果子模块是要替换的类型
        if isinstance(module, LoRACompatibleConv):
            # 创建一个新的替换模块，假设构造函数参数相同
            # 注意：这里你可能需要根据实际情况调整参数
            new_module = TestW8A8B8O8Conv2D16.from_float(module, 1., 1.)
            setattr(model, name, new_module)
        else:
            # 否则，递归遍历当前模块的子模块
            replace_unet_conv(module)

class INT8TimeStepEmbedding(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        act_fn: str = "silu",
        out_dim: int = None,
        post_act_fn: Optional[str] = None,
        cond_proj_dim=None,
    ):
        super().__init__()
        linear_cls = nn.Linear if USE_PEFT_BACKEND else LoRACompatibleLinear

        self.linear_1 = linear_cls(in_channels, time_embed_dim)

        if cond_proj_dim is not None:
            self.cond_proj = nn.Linear(cond_proj_dim, in_channels, bias=False)
        else:
            self.cond_proj = None

        self.act = get_activation(act_fn)

        if out_dim is not None:
            time_embed_dim_out = out_dim
        else:
            time_embed_dim_out = time_embed_dim
        self.linear_2 = linear_cls(time_embed_dim, time_embed_dim_out)

        if post_act_fn is None:
            self.post_act = None
        else:
            self.post_act = get_activation(post_act_fn)

    def forward(self, sample, condition=None):
        if condition is not None:
            sample = sample + self.cond_proj(condition)
        sample = self.linear_1(sample)

        if self.act is not None:
            sample = self.act(sample)

        sample = self.linear_2(sample)

        if self.post_act is not None:
            sample = self.post_act(sample)
        return sample
    
    @staticmethod
    def from_float(model: embeddings.TimestepEmbedding, fc1_input_scale, fc2_input_scale):
        int8model = INT8TimeStepEmbedding(model.config)
        int8model.config = model.config

        int8model.fc1 = W8A8B8O8LinearReLU.from_float(model.fc1, fc1_input_scale, fc2_input_scale)
        int8model.fc2 = W8A8BFP32OFP32Linear.from_float(model.fc2, fc2_input_scale)
        return int8model