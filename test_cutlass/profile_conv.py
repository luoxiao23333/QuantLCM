

import torch
import time
from torch_int._CUDA import conv2D16, conv2D8, conv2D16Small
from torch_int.nn.conv import W8A8B8O8Conv2D16, W8A8B8O8Conv2D8, TestW8A8B8O8Conv2D16
from copy import deepcopy

'''
input: NHWC
'''

test = "cutlass"


def get_trace_handler(test):
    def trace_handler(prof):
        file = open("./trace/conv/" + test + ".txt", "w")
        print(prof.key_averages().table(
            sort_by="self_cuda_time_total", row_limit=-1), file=file)
        prof.export_chrome_trace("./trace/conv/" + test + ".json")
    return trace_handler

@torch.no_grad
def test_conv(test, data_scale, verbose):
    in_channels = 320
    out_channels = 320
    kernel_size = (1,1)
    run_time = 10

    if test == "torch":

        conv = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                            stride=(1,1), padding=(1,1), dilation=(1,1), bias=True).cuda()
        x = torch.ones((2,in_channels, data_scale, data_scale), dtype=torch.float32).cuda()

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],

            record_shapes=True,
            with_flops=True,
            with_stack=True,
            profile_memory=True,
            with_modules=True,

            schedule=torch.profiler.schedule(
            wait=2,
            warmup=2,
            active=6,
            repeat=1),
            on_trace_ready=get_trace_handler(test)
            # on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')
            # used when outputting for tensorboard
        ) as p:
            for _ in range(run_time):
                result = conv(x)
                torch.cuda.synchronize()
                p.step()

    elif test == "torch_int_test":
        conv = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=(1,1), padding=(1,1), dilation=(1,1), bias=True).cuda()
        conv = TestW8A8B8O8Conv2D16.from_float(conv, 1., 1.)
        x = torch.ones((2,in_channels, data_scale, data_scale), dtype=torch.float32).to(dtype=torch.int8).cuda()
        inputs = [deepcopy(x) for _ in range(10)]

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],

            record_shapes=True,
            with_flops=True,
            with_stack=True,
            profile_memory=True,
            with_modules=True,

            schedule=torch.profiler.schedule(
            wait=2,
            warmup=2,
            active=6,
            repeat=1),
            on_trace_ready=get_trace_handler(test)
            # on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')
            # used when outputting for tensorboard
        ) as p:
            for iter in range(run_time):
                result = conv(inputs[iter])
                torch.cuda.synchronize()
                p.step()

        del conv
        del x

    elif test == "torch_int":
        conv = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=(1,1), padding=(1,1), dilation=(1,1), bias=True).cuda()
        conv = W8A8B8O8Conv2D16.from_float(conv, 1., 1.)
        x = torch.ones((2,in_channels, data_scale, data_scale), dtype=torch.float32).to(dtype=torch.int8).cuda()
        inputs = [deepcopy(x) for _ in range(10)]

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],

            record_shapes=True,
            with_flops=True,
            with_stack=True,
            profile_memory=True,
            with_modules=True,

            schedule=torch.profiler.schedule(
            wait=2,
            warmup=2,
            active=6,
            repeat=1),
            on_trace_ready=get_trace_handler(test)
            # on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')
            # used when outputting for tensorboard
        ) as p:
            for iter in range(run_time):
                result = conv(inputs[iter])
                torch.cuda.synchronize()
                p.step()

        del conv
        del x

    elif test == "cutlass":
        x = torch.ones((2,in_channels, data_scale, data_scale), dtype=torch.float32).to(dtype=torch.int8).cuda()
        filter = torch.ones((in_channels, kernel_size[0], kernel_size[1], out_channels), dtype=torch.float32).cuda().to(dtype=torch.int8)
        bias = torch.ones((data_scale), dtype=torch.float32).cuda().to(dtype=torch.int8)
        options = 0

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],

            record_shapes=True,
            with_flops=True,
            with_stack=True,
            profile_memory=True,
            with_modules=True,

            schedule=torch.profiler.schedule(
            wait=2,
            warmup=2,
            active=6,
            repeat=1),
            on_trace_ready=get_trace_handler(test)
            # on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')
            # used when outputting for tensorboard
        ) as p:
            for iter in range(run_time):
                result = conv2D16Small(x, filter, bias, options, 1, 1)
                torch.cuda.synchronize()
                p.step()
        
        del x
        del filter
        del bias



test_conv(test, data_scale=64, verbose=False)