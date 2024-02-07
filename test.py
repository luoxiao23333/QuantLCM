'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2024-02-06 15:35:21
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2024-02-07 14:50:33
FilePath: /hyou37/xluo/LCM/test.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
import time
from torch_int._CUDA import linear_a8_w8_b8_o8
from torch.nn.functional import linear
from quant.text_encoder import replace_with_time_forward
from torch_int.nn.linear import W8A8B8O8Linear

'''
For weight 16*16
linear needs 2.635084092617035e-05 secs, with result shape torch.Size([1, 16, 16])
cutlass needs 2.323593944311142e-05 secs, with result shape torch.Size([1, 16, 16])
torch_int needs 2.2670440375804902e-05 secs, with result shapetorch.Size([1, 16, 16])
For weight 160*160
linear needs 2.6399828493595122e-05 secs, with result shape torch.Size([1, 160, 160])
cutlass needs 2.0581856369972228e-05 secs, with result shape torch.Size([1, 160, 160])
torch_int needs 2.3089349269866943e-05 secs, with result shapetorch.Size([1, 160, 160])
For weight 1600*1600
linear needs 3.5297684371471404e-05 secs, with result shape torch.Size([1, 1600, 1600])
cutlass needs 2.200677990913391e-05 secs, with result shape torch.Size([1, 1600, 1600])
torch_int needs 2.3466534912586212e-05 secs, with result shapetorch.Size([1, 1600, 1600])
For weight 16000*16000
linear needs 2.7489103376865388e-05 secs, with result shape torch.Size([1, 16000, 16000])
cutlass needs 2.1804310381412507e-05 secs, with result shape torch.Size([1, 16000, 16000])
torch_int needs 2.317987382411957e-05 secs, with result shapetorch.Size([1, 16000, 16000])
For weight 32000*32000
linear needs 6.108283996582031e-05 secs, with result shape torch.Size([1, 32000, 32000])
cutlass needs 2.1001137793064117e-05 secs, with result shape torch.Size([1, 32000, 32000])
torch_int needs 2.303328365087509e-05 secs, with result shapetorch.Size([1, 32000, 32000])
'''

@torch.no_grad
def test_linear(test, data_scale, verbose):
    run_time = 10

    if test == "linear":
        linear = torch.nn.Linear(in_features=data_scale, out_features=data_scale, bias=True)
        x = torch.ones((1,data_scale, data_scale), dtype=torch.float16).cuda()
        linear.weight = torch.nn.Parameter(torch.ones((data_scale, data_scale), dtype=torch.float16).cuda())
        linear.bias = torch.nn.Parameter(torch.ones((data_scale, data_scale), dtype=torch.float16).cuda())
        for i in range(run_time):
            result = linear.forward(x)
        start_time = time.perf_counter()
        for _ in range(run_time):
            result = linear.forward(x)
        if verbose:
            print(f"linear needs {(time.perf_counter() - start_time)/run_time} secs, with result shape {result.shape}")
        del linear
        del x

    elif test == "cutlass":
        x = torch.ones((1,data_scale, data_scale), dtype=torch.float32).cuda().to(dtype=torch.int8)
        weight = torch.ones((data_scale, data_scale), dtype=torch.float32).cuda().to(dtype=torch.int8)
        bias = torch.ones((data_scale), dtype=torch.float32).cuda().to(dtype=torch.int8)
        for i in range(10):
            result = linear_a8_w8_b8_o8(x, weight, bias, 1., 1.)
        start_time = time.perf_counter()
        for _ in range(10):
            result = linear_a8_w8_b8_o8(x, weight, bias, 1., 1.)
        if verbose:
            print(f"cutlass needs {(time.perf_counter() - start_time)/10} secs, with result shape {result.shape}")
        del x
        del weight
        del bias

    elif test == "torch_int":
        linear = torch.nn.Linear(in_features=data_scale, out_features=data_scale, bias=True)
        x = torch.ones((1, data_scale, data_scale), dtype=torch.float32).cuda().to(dtype=torch.int8)
        linear.weight = torch.nn.Parameter(torch.ones((data_scale, data_scale), dtype=torch.float32).cuda())
        linear.bias = torch.nn.Parameter(torch.ones((data_scale), dtype=torch.float32).cuda())
        linear = W8A8B8O8Linear.from_float(linear, 1., 1.)
        for i in range(10):
            result = linear.forward(x)
        start_time = time.perf_counter()
        for _ in range(run_time):
            result = linear.forward(x)
        if verbose:
            print(f"torch_int needs {(time.perf_counter() - start_time)/run_time} secs, with result shape{result.shape}")
        del linear
        del x



# warm up GPU
for i in range(2):
    for test in ["linear", "cutlass", "torch_int"]:
        test_linear(test, 1600, False)
        torch.cuda.empty_cache()


for data_scale in [16, 160, 1600, 16000, 32000]:
    print(f"For weight {data_scale}*{data_scale}")
    for test in ["linear", "cutlass", "torch_int"]:
        test_linear(test, data_scale, True)
        torch.cuda.empty_cache()