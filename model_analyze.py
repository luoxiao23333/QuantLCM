import torch
from typing import Dict
import inspect
# from submodules.torch_int.torch_int.nn.linear import W8A8B8O8Linear
import torch_int
from quant.utils import LatencyLogger
from diffusers.models.lora import LoRACompatibleConv

def analyze(model):
    f = open("text_encoder.txt", 'w')
    print(model, file=f)

def write_model_arch(models: Dict[str, torch.nn.Module], write_dir="model_arch/"):
    for model_name, model in models.items():
        model_def_path = inspect.getfile(model.__class__)
        f = open(write_dir+model_name+".txt", "w")
        print(model_def_path, file=f)
        print(model, file=f)


# unit: Bytes
def model_memory_usage(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    total_size = param_size + buffer_size
    return total_size


def print_tensor_dtypes_need_padding(model):
    for name, param in model.named_parameters():
        if len(param.shape) >= 2 and (param.shape[-1] % 16 !=0 or param.shape[-2] % 16 != 0):
            print(f"Parameter {name}: dtype={param.dtype}, {param.shape}, n={param.nelement()}, ds={param.element_size()}")
    for name, buffer in model.named_buffers():
        if len(buffer.shape) >= 2 and (buffer.shape[-1] % 16 !=0 or buffer.shape[-2] % 16 != 0):
            print(f"Buffer {name}: dtype={buffer.dtype}, {buffer.shape}, n={buffer.nelement()}, ds={buffer.element_size()}")


def print_tensor_dtypes(model):
    for name, param in model.named_parameters():
        print(f"Parameter {name}: dtype={param.dtype}, {param.shape}, n={param.nelement()}, ds={param.element_size()}")
    for name, buffer in model.named_buffers():
        print(f"Buffer {name}: dtype={buffer.dtype}, {buffer.shape}, n={buffer.nelement()}, ds={buffer.element_size()}")




def print_modules_with_lora_layer(model):
    # 遍历模型的所有子模块
    for name, module in model.named_children():
        # 检查是否包含名为 'lora_layer' 的子模块并且不为 None
        if isinstance(module, torch.nn.Module) and hasattr(module, 'lora_layer') and module.lora_layer is not None:
            print("Module with 'lora_layer' found in {}: {}, and lora_layer: {}".format(name, module, module.lora_layer))
        
        # 如果子模块还包含其他子模块，递归遍历它们
        if isinstance(module, torch.nn.Module):
            print_modules_with_lora_layer(module)


'''
For unet:
{"{'stride': (1, 1), 'padding': (1, 1), 'dilation': (1, 1), 'groups': 1, 'hasBias': True}", 
"{'stride': (1, 1), 'padding': (0, 0), 'dilation': (1, 1), 'groups': 1, 'hasBias': True}", 
"{'stride': (2, 2), 'padding': (1, 1), 'dilation': (1, 1), 'groups': 1, 'hasBias': True}"}
'''
def print_conv_settings(model):
    info_set = set()
    def __print_conv_settings(model):
        for name, module in model.named_children():
            if isinstance(module, LoRACompatibleConv):
                info_set.add(
                    repr({
                        "stride": module.stride,
                        "padding": module.padding,
                        "dilation": module.dilation,
                        "groups": module.groups,
                        "hasBias": module.bias is not None,
                    })
                )

            if isinstance(module, torch.nn.Module):
                __print_conv_settings(module)

    __print_conv_settings(model)
    return info_set
