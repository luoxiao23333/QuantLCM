'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2024-02-06 15:35:21
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2024-02-12 14:53:25
FilePath: /xluo/LCM/test.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
import time
from torch_int._CUDA import linear_a8_w8_b8_o8
from torch_int.nn.linear import W8A8B8O8Linear

'''
For weight 16*16
linear need 0.00023228861391544342 secs for data movement
linear needs 2.9807910323143005e-05 secs, with result shape torch.Size([1, 16, 16])
cutlass need 0.00022237375378608704 secs for data movement
cutlass needs 2.0337663590908052e-05 secs, with result shape torch.Size([1, 16, 16])
torch_int need 0.00022565573453903198 secs for data movement
torch_int needs 2.2893957793712617e-05 secs, with result shapetorch.Size([1, 16, 16])
For weight 160*160
linear need 0.00023368559777736664 secs for data movement
linear needs 2.630222588777542e-05 secs, with result shape torch.Size([1, 160, 160])
cutlass need 0.00022712349891662598 secs for data movement
cutlass needs 2.0952522754669188e-05 secs, with result shape torch.Size([1, 160, 160])
torch_int need 0.000226631760597229 secs for data movement
torch_int needs 2.2670067846775054e-05 secs, with result shapetorch.Size([1, 160, 160])
For weight 1600*1600
linear need 0.0006774552166461945 secs for data movement
linear needs 2.477390855550766e-05 secs, with result shape torch.Size([1, 1600, 1600])
cutlass need 0.00030701979994773865 secs for data movement
cutlass needs 2.2097676992416383e-05 secs, with result shape torch.Size([1, 1600, 1600])
torch_int need 0.0003273412585258484 secs for data movement
torch_int needs 2.27123498916626e-05 secs, with result shapetorch.Size([1, 1600, 1600])
For weight 16000*16000
linear need 0.07419094629585743 secs for data movement
linear needs 2.7587078511714937e-05 secs, with result shape torch.Size([1, 16000, 16000])
cutlass need 0.037992436438798904 secs for data movement
cutlass needs 2.0442157983779906e-05 secs, with result shape torch.Size([1, 16000, 16000])
torch_int need 0.038599491119384766 secs for data movement
torch_int needs 2.2991560399532318e-05 secs, with result shapetorch.Size([1, 16000, 16000])
For weight 32000*32000
linear need 0.2859418913722038 secs for data movement
linear needs 6.164852529764176e-05 secs, with result shape torch.Size([1, 32000, 32000])
cutlass need 0.14255738258361816 secs for data movement
cutlass needs 2.057533711194992e-05 secs, with result shape torch.Size([1, 32000, 32000])
torch_int need 0.1388553399592638 secs for data movement
torch_int needs 2.332683652639389e-05 secs, with result shapetorch.Size([1, 32000, 32000])
'''

@torch.no_grad
def test_linear(test, data_scale, verbose):
    run_time = 10

    if test == "linear":
        linear = torch.nn.Linear(in_features=data_scale, out_features=data_scale, bias=True)
        x = torch.ones((1,data_scale, data_scale), dtype=torch.float16)
        start_time = time.perf_counter()
        x = x.cuda()
        if verbose:
            print(f"{test} need {time.perf_counter() - start_time} secs for data movement")
        linear.weight = torch.nn.Parameter(torch.ones((data_scale, data_scale), dtype=torch.float16).cuda())
        linear.bias = torch.nn.Parameter(torch.ones((data_scale, data_scale), dtype=torch.float16).cuda())
        for i in range(run_time):
            result = linear.forward(x)
        start_time = time.perf_counter()
        for _ in range(run_time):
            result = linear.forward(x)
        torch.cuda.synchronize()
        if verbose:
            print(f"{test} needs {(time.perf_counter() - start_time)/run_time} secs, with result shape {result.shape}")
        del linear
        del x

    elif test == "cutlass":
        x = torch.ones((1,data_scale, data_scale), dtype=torch.float32).to(dtype=torch.int8)
        start_time = time.perf_counter()
        x = x.cuda()
        if verbose:
            print(f"{test} need {time.perf_counter() - start_time} secs for data movement")
        weight = torch.ones((data_scale, data_scale), dtype=torch.float32).cuda().to(dtype=torch.int8)
        bias = torch.ones((data_scale), dtype=torch.float32).cuda().to(dtype=torch.int8)
        for i in range(10):
            result = linear_a8_w8_b8_o8(x, weight, bias, 1., 1.)
        start_time = time.perf_counter()
        for _ in range(10):
            result = linear_a8_w8_b8_o8(x, weight, bias, 1., 1.)
        torch.cuda.synchronize()
        if verbose:
            print(f"{test} needs {(time.perf_counter() - start_time)/10} secs, with result shape {result.shape}")
        del x
        del weight
        del bias

    elif test == "torch_int":
        linear = torch.nn.Linear(in_features=data_scale, out_features=data_scale, bias=True)
        x = torch.ones((1, data_scale, data_scale), dtype=torch.float32).to(dtype=torch.int8)
        start_time = time.perf_counter()
        x = x.cuda()
        if verbose:
            print(f"{test} need {time.perf_counter() - start_time} secs for data movement")
        linear.weight = torch.nn.Parameter(torch.ones((data_scale, data_scale), dtype=torch.float32).cuda())
        linear.bias = torch.nn.Parameter(torch.ones((data_scale), dtype=torch.float32).cuda())
        linear = W8A8B8O8Linear.from_float(linear, 1., 1.)
        for i in range(10):
            result = linear.forward(x)
        start_time = time.perf_counter()
        for _ in range(run_time):
            result = linear.forward(x)
        torch.cuda.synchronize()
        if verbose:
            print(f"{test} needs {(time.perf_counter() - start_time)/run_time} secs, with result shape{result.shape}")
        del linear
        del x



# warm up GPU
for i in range(2):
    for test in ["linear", "cutlass", "torch_int"]:
        test_linear(test, 1600, False)
        torch.cuda.empty_cache()


for data_scale in [16, 160, 1600]:
    print(f"For weight {data_scale}*{data_scale}")
    for test in ["linear", "cutlass", "torch_int"]:
        test_linear(test, data_scale, True)
        torch.cuda.empty_cache()