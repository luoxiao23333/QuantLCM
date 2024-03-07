

import torch
import time
from torch_int._CUDA import bmm_s8t_s8n_s8t
from torch_int.nn.bmm import BMM_S8T_S8N_S8T
from copy import deepcopy

'''
input: NHWC
'''

test = "cutlass"


def get_trace_handler(test):
    def trace_handler(prof):
        file = open("./trace/bmm/" + test + ".txt", "w+")
        print(prof.key_averages().table(
            sort_by="self_cuda_time_total", row_limit=-1), file=file)
        prof.export_chrome_trace("./trace/bmm/" + test + ".json")
    return trace_handler

@torch.no_grad
def test_linear(test, data_scale, verbose):
    in_features = 768
    out_features = 768
    batch = 100
    run_time = 100

    if test == "torch":
        mat1 = torch.ones((batch, in_features, out_features)).cuda()
        mat2 = mat1.transpose(1,2)

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
                result = torch.bmm(mat1, mat2)
                torch.cuda.synchronize()
                p.step()

    elif test == "torch_int":
        mat1 = torch.ones((batch, in_features, out_features)).cuda()
        mat2 = mat1.clone()
        bmm = BMM_S8T_S8N_S8T.from_scale(1.,1.,1.)

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
                bmm(mat1, mat2)
                torch.cuda.synchronize()
                p.step()

        del linear
        del x

    elif test == "cutlass":
        x = torch.ones((batch,data_scale, in_features), dtype=torch.float32).to(dtype=torch.int8)
        x = x.cuda()
        weight = torch.ones((batch, out_features, in_features), dtype=torch.float32).cuda().to(dtype=torch.int8)
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
                result = bmm_s8t_s8n_s8t(x, weight,1)
                # torch.cuda.synchronize()
                p.step()
        
        del x
        del weight
        del bias



test_linear(test, data_scale=768, verbose=False)

