'''
// options[0-1]: stride
// options[2-5]: padding
// options[6-7]: dilation 
torch::Tensor conv2D(torch::Tensor input,  // INT8
                    torch::Tensor filter, // INT8
                    torch::Tensor bias,    // INT8
                    torch::Tensor options, // INT8
                    float alpha,          // FP32
                    float beta            // FP32
){
'''

'''
For H*W: 16*16
torch needs 0.4151124507188797 ms, with result shape torch.Size([2, 1280, 18, 18])
cutlass needs 0.15731602907180786 ms, with result shape torch.Size([2, 1280, 18, 18])
torch_int needs 0.16611628234386444 ms, with result shape torch.Size([2, 1280, 18, 18])
torch_int_test needs 0.15522073954343796 ms, with result shape torch.Size([2, 1280, 18, 18])
For H*W: 160*160
torch needs 11.02035716176033 ms, with result shape torch.Size([2, 1280, 162, 162])
cutlass needs 1.583610475063324 ms, with result shape torch.Size([2, 1280, 162, 162])
torch_int needs 1.7341524362564087 ms, with result shape torch.Size([2, 1280, 162, 162])
torch_int_test needs 1.7383145168423653 ms, with result shape torch.Size([2, 1280, 162, 162])
For H*W: 240*240
torch needs 25.06272252649069 ms, with result shape torch.Size([2, 1280, 242, 242])
cutlass needs 3.3794602379202843 ms, with result shape torch.Size([2, 1280, 242, 242])
torch_int needs 3.726254403591156 ms, with result shape torch.Size([2, 1280, 242, 242])
torch_int_test needs 3.7241240963339806 ms, with result shape torch.Size([2, 1280, 242, 242])
'''

import torch
import time
from torch_int._CUDA import conv2D16, conv2D8
from torch_int.nn.conv import W8A8B8O8Conv2D16, W8A8B8O8Conv2D8, TestW8A8B8O8Conv2D16

'''
input: NHWC
'''

@torch.no_grad
def test_conv(test, data_scale, verbose):
    in_channels = 1280
    out_channels = 1280
    kernel_size = (1,1)
    run_time = 10

    if test == "torch":
        conv = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=(1,1), padding=(1,1), dilation=(1,1), bias=True).cuda()
        x = torch.ones((2,in_channels, data_scale, data_scale), dtype=torch.float32)
        start_time = time.perf_counter()
        x = x.cuda()
        for i in range(run_time):
            result = conv(x)
        start_time = time.perf_counter()
        for _ in range(run_time):
            result = conv(x)
        torch.cuda.synchronize()
        if verbose:
            print(f"torch needs {1000*(time.perf_counter() - start_time)/run_time} ms, with result shape {result.shape}")
        del conv
        del x

    elif test == "torch_int_test":
        conv = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=(1,1), padding=(1,1), dilation=(1,1), bias=True).cuda()
        conv = TestW8A8B8O8Conv2D16.from_float(conv, 1., 1.)
        x = torch.ones((2,in_channels, data_scale, data_scale), dtype=torch.float32).to(dtype=torch.int8)
        start_time = time.perf_counter()
        x = x.cuda()
        for i in range(run_time):
            result = conv(x)
        start_time = time.perf_counter()
        for _ in range(run_time):
            result = conv(x)
        torch.cuda.synchronize()
        if verbose:
            print(f"torch_int_test needs {1000*(time.perf_counter() - start_time)/run_time} ms, with result shape {result.shape}")
        del conv
        del x

    elif test == "torch_int":
        conv = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=(1,1), padding=(1,1), dilation=(1,1), bias=True).cuda()
        conv = W8A8B8O8Conv2D16.from_float(conv, 1., 1.)
        x = torch.ones((2,in_channels, data_scale, data_scale), dtype=torch.float32).to(dtype=torch.int8)
        start_time = time.perf_counter()
        x = x.cuda()
        for i in range(run_time):
            result = conv(x)
        start_time = time.perf_counter()
        for _ in range(run_time):
            result = conv(x)
        torch.cuda.synchronize()
        if verbose:
            print(f"torch_int needs {1000*(time.perf_counter() - start_time)/run_time} ms, with result shape {result.shape}")
        del conv
        del x

    elif test == "cutlass":
        x = torch.ones((2,in_channels, data_scale, data_scale), dtype=torch.float32).to(dtype=torch.int8)
        start_time = time.perf_counter()
        x = x.cuda()
        filter = torch.ones((in_channels, kernel_size[0], kernel_size[1], out_channels), dtype=torch.float32).cuda().to(dtype=torch.int8)
        bias = torch.ones((data_scale), dtype=torch.float32).cuda().to(dtype=torch.int8)
        options = torch.tensor((1,1,1,1,1,1,1,1), dtype=torch.int8).cuda()
        for i in range(10):
            result = conv2D16(x, filter, bias, options, 1, 1)
        start_time = time.perf_counter()
        for _ in range(10):
            result = conv2D16(x, filter, bias, options, 1, 1)
        if verbose:
            print(f"cutlass needs {1000*(time.perf_counter() - start_time)/10} ms, with result shape {result.shape}")
        del x
        del filter
        del bias



# warm up GPU
for i in range(2):
    for test in ["torch", "cutlass", "torch_int"]:
        test_conv(test, 160, False)
        torch.cuda.empty_cache()


for data_scale in [16, 160, 240]:
    print(f"For H*W: {data_scale}*{data_scale}")
    for test in ["torch", "cutlass", "torch_int", "torch_int_test"]:
        test_conv(test, data_scale, True)
        torch.cuda.empty_cache()