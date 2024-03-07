

import torch
import time
from torch_int._CUDA import linear_a8_w8_b8_o8
from torch_int.nn.linear import W8A8B8O8Linear
from copy import deepcopy

'''
input: NHWC
'''

test = "cutlass"


def get_trace_handler(test):
    def trace_handler(prof):
        file = open("./trace/linear/" + test + ".txt", "w+")
        print(prof.key_averages().table(
            sort_by="self_cuda_time_total", row_limit=-1), file=file)
        prof.export_chrome_trace("./trace/linear/" + test + ".json")
    return trace_handler

@torch.no_grad
def test_linear(test, data_scale, verbose):
    in_features = 768
    out_features = 768
    batch = 100
    run_time = 100

    if test == "torch":
        linear = torch.nn.Linear(in_features=in_features, out_features=out_features).cuda()
        x = torch.ones((batch, in_features, out_features), dtype=torch.float32).cuda()

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
                result = linear(x)
                torch.cuda.synchronize()
                p.step()

    elif test == "torch_int":
        linear = torch.nn.Linear(in_features=in_features, out_features=out_features).cuda()
        linear = W8A8B8O8Linear.from_float(linear, 1., 1.)
        x = torch.ones((batch,in_features, out_features), dtype=torch.float32).to(dtype=torch.int8).cuda()

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
                result = linear(x)
                torch.cuda.synchronize()
                p.step()

        del linear
        del x

    elif test == "cutlass":
        x = torch.ones((batch,data_scale, data_scale), dtype=torch.float32).to(dtype=torch.int8)
        x = x.cuda()
        weight = torch.ones((data_scale, data_scale), dtype=torch.float32).cuda().to(dtype=torch.int8)
        bias = torch.ones((data_scale), dtype=torch.float32).cuda().to(dtype=torch.int8)

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
                result = linear_a8_w8_b8_o8(x, weight, bias,1,1)
                torch.cuda.synchronize()
                p.step()
        
        del x
        del weight
        del bias



test_linear(test, data_scale=768, verbose=False)

