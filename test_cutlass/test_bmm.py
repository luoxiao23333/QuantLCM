'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2024-02-28 16:27:07
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2024-02-28 16:46:35
FilePath: /xluo/LCM/test_cutlass/test_bmm.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
import time
from torch_int._CUDA import bmm_s8t_s8n_s8t
from torch_int.nn.bmm import BMM_S8T_S8N_S8T

'''
For BMNK (1, 16, 16, 16)
linear needs 0.012497999705374241 ms, with result shape torch.Size([1, 16, 16])
cutlass needs 0.008647999493405223 ms, with result shape torch.Size([1, 16, 16])
torch_int needs 0.013975996989756823 ms, with result shapetorch.Size([1, 16, 16])
For BMNK (1, 160, 160, 160)
linear needs 0.012851099018007517 ms, with result shape torch.Size([1, 160, 160])
cutlass needs 0.011273002019152045 ms, with result shape torch.Size([1, 160, 160])
torch_int needs 0.01387899974361062 ms, with result shapetorch.Size([1, 160, 160])
For BMNK (1, 1280, 1280, 1280)
linear needs 0.06864420138299465 ms, with result shape torch.Size([1, 1280, 1280])
cutlass needs 0.033541000448167324 ms, with result shape torch.Size([1, 1280, 1280])
torch_int needs 0.03400309942662716 ms, with result shapetorch.Size([1, 1280, 1280])
For BMNK (1, 1600, 1600, 1600)
linear needs 0.13072420260868967 ms, with result shape torch.Size([1, 1600, 1600])
cutlass needs 0.0762991956435144 ms, with result shape torch.Size([1, 1600, 1600])
torch_int needs 0.07668720209039748 ms, with result shapetorch.Size([1, 1600, 1600])
'''

@torch.no_grad
def test_linear(test, data_scale, verbose):
    B, M, N, K = data_scale
    run_time = 10

    if test == "linear":
        mat1 = torch.ones((B,M, K), dtype=torch.float16).cuda()
        mat2 = torch.ones((B,N, K), dtype=torch.float16).cuda().transpose(1,2)
        for i in range(run_time):
            result = torch.bmm(mat1,mat2)
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        for _ in range(run_time):
            result = torch.bmm(mat1,mat2)
        torch.cuda.synchronize()
        if verbose:
            print(f"{test} needs {1000*(time.perf_counter() - start_time)/run_time} ms, with result shape {result.shape}")

    elif test == "cutlass":
        mat1 = torch.ones((B,M, K), dtype=torch.float16).cuda().to(dtype=torch.int8)
        mat2 = torch.ones((B,N, K), dtype=torch.float16).cuda().to(dtype=torch.int8)
        for i in range(10):
            result = bmm_s8t_s8n_s8t(mat1, mat2, 1.)
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        for _ in range(10):
            result = bmm_s8t_s8n_s8t(mat1, mat2, 1.)
        torch.cuda.synchronize()
        if verbose:
            print(f"{test} needs {1000*(time.perf_counter() - start_time)/10} ms, with result shape {result.shape}")

    elif test == "torch_int":
        mat1 = torch.ones((B,M, K), dtype=torch.float16).cuda().to(dtype=torch.int8)
        mat2 = torch.ones((B,N, K), dtype=torch.float16).cuda().to(dtype=torch.int8)
        bmm = BMM_S8T_S8N_S8T.from_scale(1., 1., 1.)
        for i in range(10):
            result = bmm(mat1, mat2)
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        for _ in range(run_time):
            result = bmm(mat1, mat2)
        torch.cuda.synchronize()
        if verbose:
            print(f"{test} needs {1000*(time.perf_counter() - start_time)/run_time} ms, with result shape{result.shape}")



# BMNK
data_range = [(1, 16, 16, 16), (1, 160, 160, 160), (1, 1280, 1280, 1280), (1, 1600, 1600, 1600)]

# warm up GPU
for i in range(2):
    for test in ["linear", "cutlass", "torch_int"]:
        test_linear(test, data_range[-1], False)
        torch.cuda.empty_cache()


for data_scale in data_range:
    print(f"For BMNK {data_scale}")
    for test in ["linear", "cutlass", "torch_int"]:
        test_linear(test, data_scale, True)
        torch.cuda.empty_cache()