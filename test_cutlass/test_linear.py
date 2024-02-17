import torch
import time
from torch_int._CUDA import linear_a8_w8_b8_o8
from torch_int.nn.linear import W8A8B8O8Linear

'''
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


for data_scale in [16, 160, 1600]:
    print(f"For weight {data_scale}*{data_scale}")
    for test in ["linear", "cutlass", "torch_int"]:
        test_linear(test, data_scale, True)
        torch.cuda.empty_cache()