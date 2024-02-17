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
For H*W: 8*8
torch needs 0.05554426461458206 ms, with result shape torch.Size([2, 320, 10, 10]), with input shape torch.Size([2, 320, 8, 8])
cutlass needs 0.01187305897474289 ms, with result shape torch.Size([2, 320, 10, 10]), with input shape torch.Size([2, 320, 8, 8])
torch_int needs 0.01787915825843811 ms, with result shape torch.Size([2, 320, 10, 10]), with input shape torch.Size([2, 320, 8, 8])
torch_int_test needs 0.02370402216911316 ms, with result shape torch.Size([2, 320, 10, 10]), with input shape torch.Size([2, 320, 8, 8])
For H*W: 16*16
torch needs 0.054454803466796875 ms, with result shape torch.Size([2, 320, 18, 18]), with input shape torch.Size([2, 320, 16, 16])
cutlass needs 0.01206863671541214 ms, with result shape torch.Size([2, 320, 18, 18]), with input shape torch.Size([2, 320, 16, 16])
torch_int needs 0.01778826117515564 ms, with result shape torch.Size([2, 320, 18, 18]), with input shape torch.Size([2, 320, 16, 16])
torch_int_test needs 0.027112290263175964 ms, with result shape torch.Size([2, 320, 18, 18]), with input shape torch.Size([2, 320, 16, 16])
For H*W: 32*32
torch needs 0.06997324526309967 ms, with result shape torch.Size([2, 320, 34, 34]), with input shape torch.Size([2, 320, 32, 32])
cutlass needs 0.013723596930503845 ms, with result shape torch.Size([2, 320, 34, 34]), with input shape torch.Size([2, 320, 32, 32])
torch_int needs 0.01816563308238983 ms, with result shape torch.Size([2, 320, 34, 34]), with input shape torch.Size([2, 320, 32, 32])
torch_int_test needs 0.0242697075009346 ms, with result shape torch.Size([2, 320, 34, 34]), with input shape torch.Size([2, 320, 32, 32])
For H*W: 64*64
torch needs 0.2784827724099159 ms, with result shape torch.Size([2, 320, 66, 66]), with input shape torch.Size([2, 320, 64, 64])
cutlass needs 0.011872686445713043 ms, with result shape torch.Size([2, 320, 66, 66]), with input shape torch.Size([2, 320, 64, 64])
torch_int needs 0.04059839993715286 ms, with result shape torch.Size([2, 320, 66, 66]), with input shape torch.Size([2, 320, 64, 64])
torch_int_test needs 0.03406815230846405 ms, with result shape torch.Size([2, 320, 66, 66]), with input shape torch.Size([2, 320, 64, 64])
For H*W: 160*160
torch needs 1.5172263607382774 ms, with result shape torch.Size([2, 320, 162, 162]), with input shape torch.Size([2, 320, 160, 160])
cutlass needs 0.011635199189186096 ms, with result shape torch.Size([2, 320, 162, 162]), with input shape torch.Size([2, 320, 160, 160])
torch_int needs 0.30288510024547577 ms, with result shape torch.Size([2, 320, 162, 162]), with input shape torch.Size([2, 320, 160, 160])
torch_int_test needs 0.2984853461384773 ms, with result shape torch.Size([2, 320, 162, 162]), with input shape torch.Size([2, 320, 160, 160])
For H*W: 240*240
torch needs 3.2652487978339195 ms, with result shape torch.Size([2, 320, 242, 242]), with input shape torch.Size([2, 320, 240, 240])
cutlass needs 0.012731924653053284 ms, with result shape torch.Size([2, 320, 242, 242]), with input shape torch.Size([2, 320, 240, 240])
torch_int needs 0.6390843540430069 ms, with result shape torch.Size([2, 320, 242, 242]), with input shape torch.Size([2, 320, 240, 240])
torch_int_test needs 0.633692741394043 ms, with result shape torch.Size([2, 320, 242, 242]), with input shape torch.Size([2, 320, 240, 240])
'''

import torch
import time
from torch_int._CUDA import conv2D16, conv2D8, conv2D16Small
from torch_int.nn.conv import W8A8B8O8Conv2D16, W8A8B8O8Conv2D8, TestW8A8B8O8Conv2D16

'''
input: NHWC
'''

@torch.no_grad
def test_conv(test, data_scale, verbose):
    in_channels = 320
    out_channels = 320
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
            print(f"torch needs {1000*(time.perf_counter() - start_time)/run_time} ms, with result shape {result.shape}, with input shape {x.shape}")
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
            print(f"torch_int_test needs {1000*(time.perf_counter() - start_time)/run_time} ms, with result shape {result.shape}, with input shape {x.shape}")
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
            print(f"torch_int needs {1000*(time.perf_counter() - start_time)/run_time} ms, with result shape {result.shape}, with input shape {x.shape}")
        del conv
        del x

    elif test == "cutlass":
        x = torch.ones((2,in_channels, data_scale, data_scale), dtype=torch.float32).to(dtype=torch.int8)
        start_time = time.perf_counter()
        x = x.cuda()
        filter = torch.ones((in_channels, kernel_size[0], kernel_size[1], out_channels), dtype=torch.float32).cuda().to(dtype=torch.int8)
        bias = torch.ones((data_scale), dtype=torch.float32).cuda().to(dtype=torch.int8)
        options = 0
        for i in range(10):
            result = conv2D16(x, filter, bias, options, 1, 1)
        start_time = time.perf_counter()
        for _ in range(10):
            result = conv2D16(x, filter, bias, options, 1, 1)
        if verbose:
            print(f"cutlass needs {1000*(time.perf_counter() - start_time)/10} ms, with result shape {result.shape}, with input shape {x.shape}")
        del x
        del filter
        del bias



# warm up GPU
for i in range(2):
    for test in ["torch", "cutlass", "torch_int"]:
        test_conv(test, 160, False)
        torch.cuda.empty_cache()


# in unet, data_scale range is [8~64]
for data_scale in [8, 16, 32, 64, 160, 240]:
    print(f"For H*W: {data_scale}*{data_scale}")
    for test in ["torch", "cutlass", "torch_int", "torch_int_test"]:
        test_conv(test, data_scale, True)
        torch.cuda.empty_cache()