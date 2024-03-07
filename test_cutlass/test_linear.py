'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2024-02-06 15:35:21
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2024-02-28 16:53:10
FilePath: /xluo/LCM/test_cutlass/test_linear.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE768
'''
import torch
import time
from torch_int._CUDA import linear_a8_w8_b8_o8
from torch_int.nn.linear import W8A8B8O8Linear

'''
ThreadBlock <128, 256, 64>
For weight 16*16
linear needs 0.02952180802822113 ms, with result shape torch.Size([1, 16, 16])
cutlass needs 0.02308916300535202 ms, with result shape torch.Size([1, 16, 16])
torch_int needs 0.02545006573200226 ms, with result shapetorch.Size([1, 16, 16])
For weight 160*160
linear needs 0.028998032212257385 ms, with result shape torch.Size([1, 160, 160])
cutlass needs 0.023794732987880707 ms, with result shape torch.Size([1, 160, 160])
torch_int needs 0.025547482073307037 ms, with result shapetorch.Size([1, 160, 160])
For weight 1600*1600
linear needs 0.2869335934519768 ms, with result shape torch.Size([1, 1600, 1600])
cutlass needs 0.16133226454257965 ms, with result shape torch.Size([1, 1600, 1600])
torch_int needs 0.16166716814041138 ms, with result shapetorch.Size([1, 1600, 1600])
-------------------------------------------------------------------------------
ThreadBlock <256, 128, 64>
For weight 16*16
linear needs 0.031057698652148247 ms, with result shape torch.Size([1, 16, 16])
cutlass needs 0.024655699962750077 ms, with result shape torch.Size([1, 16, 16])
torch_int needs 0.024805800057947636 ms, with result shapetorch.Size([1, 16, 16])
For weight 160*160
linear needs 0.02785070100799203 ms, with result shape torch.Size([1, 160, 160])
cutlass needs 0.022977800108492374 ms, with result shape torch.Size([1, 160, 160])
torch_int needs 0.0254607992246747 ms, with result shapetorch.Size([1, 160, 160])
For weight 768*768
linear needs 0.03165670204907656 ms, with result shape torch.Size([1, 768, 768])
cutlass needs 0.030318700009956956 ms, with result shape torch.Size([1, 768, 768])
torch_int needs 0.030185701325535774 ms, with result shapetorch.Size([1, 768, 768])
For weight 1280*1280
linear needs 0.08588619530200958 ms, with result shape torch.Size([1, 1280, 1280])
cutlass needs 0.043898599687963724 ms, with result shape torch.Size([1, 1280, 1280])
torch_int needs 0.04408460226841271 ms, with result shapetorch.Size([1, 1280, 1280])
For weight 1600*1600
linear needs 0.15714349574409425 ms, with result shape torch.Size([1, 1600, 1600])
cutlass needs 0.09391209459863603 ms, with result shape torch.Size([1, 1600, 1600])
torch_int needs 0.09470009827055037 ms, with result shapetorch.Size([1, 1600, 1600])
-------------------------------------------------------------------------------
ThreadBlock <256, 128, 256>
'''

@torch.no_grad
def test_linear(test, data_scale, verbose):
    run_time = 10

    if test == "linear":
        linear = torch.nn.Linear(in_features=data_scale, out_features=data_scale, bias=True)
        x = torch.ones((1,data_scale, data_scale), dtype=torch.float16)
        start_time = time.perf_counter()
        x = x.cuda()
        linear.weight = torch.nn.Parameter(torch.ones((data_scale, data_scale), dtype=torch.float16).cuda())
        linear.bias = torch.nn.Parameter(torch.ones((data_scale, data_scale), dtype=torch.float16).cuda())
        for i in range(run_time):
            result = linear.forward(x)
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        for _ in range(run_time):
            result = linear.forward(x)
        torch.cuda.synchronize()
        if verbose:
            print(f"{test} needs {1000*(time.perf_counter() - start_time)/run_time} ms, with result shape {result.shape}")
        del linear
        del x

    elif test == "cutlass":
        x = torch.ones((1,data_scale, data_scale), dtype=torch.float32).to(dtype=torch.int8)
        start_time = time.perf_counter()
        x = x.cuda()
        weight = torch.ones((data_scale, data_scale), dtype=torch.float32).cuda().to(dtype=torch.int8)
        bias = torch.ones((data_scale), dtype=torch.float32).cuda().to(dtype=torch.int8)
        for i in range(10):
            result = linear_a8_w8_b8_o8(x, weight, bias, 1., 1.)
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        for _ in range(10):
            result = linear_a8_w8_b8_o8(x, weight, bias, 1., 1.)
        torch.cuda.synchronize()
        if verbose:
            print(f"{test} needs {1000*(time.perf_counter() - start_time)/10} ms, with result shape {result.shape}")
        del x
        del weight
        del bias

    elif test == "torch_int":
        linear = torch.nn.Linear(in_features=data_scale, out_features=data_scale, bias=True)
        x = torch.ones((1, data_scale, data_scale), dtype=torch.float32).to(dtype=torch.int8)
        start_time = time.perf_counter()
        x = x.cuda()
        linear.weight = torch.nn.Parameter(torch.ones((data_scale, data_scale), dtype=torch.float32).cuda())
        linear.bias = torch.nn.Parameter(torch.ones((data_scale), dtype=torch.float32).cuda())
        linear = W8A8B8O8Linear.from_float(linear, 1., 1.)
        for i in range(10):
            result = linear.forward(x)
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        for _ in range(run_time):
            result = linear.forward(x)
        torch.cuda.synchronize()
        if verbose:
            print(f"{test} needs {1000*(time.perf_counter() - start_time)/run_time} ms, with result shape{result.shape}")
        del linear
        del x



# warm up GPU
for i in range(2):
    for test in ["linear", "cutlass", "torch_int"]:
        test_linear(test, 1600, False)
        torch.cuda.empty_cache()


for data_scale in [16, 160, 768, 1280, 1600]:
    print(f"For weight {data_scale}*{data_scale}")
    for test in ["linear", "cutlass", "torch_int"]:
        test_linear(test, data_scale, True)
        torch.cuda.empty_cache()