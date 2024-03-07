
# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os
import torch
from diffusers import DiffusionPipeline
from typing import List
from argparse import ArgumentParser
import time
import datetime
from copy import deepcopy
from transformers.models.clip import CLIPTextModel
from diffusers.models.unet_2d_condition import UNet2DConditionModel
from diffusers.pipelines.latent_consistency_models.pipeline_latent_consistency_text2img import LatentConsistencyModelPipeline
from quant.utils import LatencyLogger
import tvm
import tvm.contrib.dlpack as dlpack

import numpy as np

from tvm.contrib.download import download_testdata

def get_trace_handler(test):
    def trace_handler(prof):
        file = open("./tvm_trace/" + test + ".txt", "w")
        print(prof.key_averages().table(
            sort_by="self_cuda_time_total", row_limit=-1), file=file)
        prof.export_chrome_trace("./tvm_trace/" + test + ".json")
    return trace_handler



class Predictor:
    def setup(self, args) -> None:
        """Load the model into memory to make running multiple predictions efficient"""

        # # Official LCM Pipeline supported now.
        # self.pipe = DiffusionPipeline.from_pretrained(
        #     "SimianLuo/LCM_Dreamshaper_v7",
        #     cache_dir="model_cache",
        #     local_files_only=True,
        # )

        # Want to use older ones, need to add "revision="fb9c5d1"
        # self.pipe = DiffusionPipeline.from_pretrained(
        #     "SimianLuo/LCM_Dreamshaper_v7",
        #     custom_pipeline="latent_consistency_txt2img",
        #     custom_revision="main",
        #     revision="fb9c5d1",
        #     cache_dir="model_cache",
        #     local_files_only=True,
        # )
        self.pipe = DiffusionPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7", local_files_only=True)
        self.pipe.to(torch_device="cuda", torch_dtype=torch.float32)

        self.record_path = {
            "itext": f"{args.forward_timed_dir}/itext.txt",
            "otext": f"{args.forward_timed_dir}/otext.txt",
            "iunet": f"{args.forward_timed_dir}/iunet.txt",
            "ounet": f"{args.forward_timed_dir}/ounet.txt"
        }

        if args.quant_text == True:
            self.quant_text(args.record_text)
        elif args.record_text == True:
            self.time_forward("text")

        if args.quant_unet == True:
            self.quant_unet(args.record_unet)
        elif args.record_unet == True:
            self.time_forward("unet")


    def predict(
        self,
        args
    ) -> List[str]:
        """Run a single prediction on the model"""
        seed = args.seed if args.seed is not None else int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")
        torch.manual_seed(seed)

        torch.cuda.synchronize()
        start_time = time.perf_counter()
        result = self.pipe(
            prompt=args.prompt,
            width=args.width,
            height=args.height,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            num_images_per_prompt=args.num_images,
            lcm_origin_steps=50,
            output_type="pil",
        ).images
        torch.cuda.synchronize()
        print(f"Inference Time: {(time.perf_counter()-start_time)*1000} ms")

        output_paths = []
        result_dir = f"./sample/{datetime.datetime.now()}"
        os.makedirs(result_dir)
        for i, sample in enumerate(result):
            output_path = f"{result_dir}/out-{i}.png"
            sample.save(output_path)
            output_paths.append(output_path)
        
        if args.record_unet or args.record_text:
            LatencyLogger.write_all()

        return output_paths
    
    def profile(
        self,
        args
    ) -> None:
        """Run a single prediction on the model"""
        seed = args.seed if args.seed is not None else int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")
        torch.manual_seed(seed)

        torch.cuda.synchronize()

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
                repeat=1
            ),
            on_trace_ready=get_trace_handler("original")
            # on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')
            # used when outputting for tensorboard
        ) as p:
            for _ in range(10):
                result = self.pipe(
                    prompt=args.prompt,
                    width=args.width,
                    height=args.height,
                    guidance_scale=args.guidance_scale,
                    num_inference_steps=args.num_inference_steps,
                    num_images_per_prompt=args.num_images,
                    lcm_origin_steps=50,
                    output_type="pil",
                ).images
                torch.cuda.synchronize()
                p.step()

    def quant_text(self, forward_timed):
        from quant.utils import replace_with_time_forward
        from quant.text_encoder import INT8CLIPTextModel
        components = getattr(self.pipe, 'components')
        text_encoder = deepcopy(components['text_encoder'])

        int8_text_encoder = INT8CLIPTextModel.from_float(text_encoder, encoder_layer_scales=[{
            "attn_input_scale": 1,
            "q_output_scale": 1,
            "k_output_scale": 1,
            "v_output_scale": 1,
            "out_input_scale": 1,
            "fc1_input_scale": 1,
            "fc2_input_scale": 1,
        } for _ in range(12)])
        int8_text_encoder.eval()
        #replace_with_time_forward(int8_text_encoder)
        if forward_timed:
            replace_with_time_forward(int8_text_encoder, self.record_path["itext"])
        int8_text_encoder.eval()
        print(int8_text_encoder.device)
        input()
        components['text_encoder'] = int8_text_encoder

        self.pipe = LatentConsistencyModelPipeline(**components, requires_safety_checker=False)
    
    def quant_unet(self, forward_timed):
        from quant.utils import replace_with_time_forward
        from quant.unet import replace_unet_conv
        components = getattr(self.pipe, 'components')
        unet = deepcopy(components['unet'])

        replace_unet_conv(unet)
        if forward_timed:
            replace_with_time_forward(unet, self.record_path["iunet"])
        unet.eval()
        components['unet'] = unet

        self.pipe = LatentConsistencyModelPipeline(**components, requires_safety_checker=False)

    def time_forward(self, model_name):
        from quant.utils import replace_with_time_forward
        from quant.unet import replace_unet_conv
        components = getattr(self.pipe, 'components')
        model = deepcopy(components[model_name])

        replace_with_time_forward(model, self.record_path[f"o{model_name}"])
        model.eval()
        components[model_name] = model

        self.pipe = LatentConsistencyModelPipeline(**components, requires_safety_checker=False)

    def save_model(self, directory = "./model_weights"):
        components = self.pipe.components
        text, unet, vae, tokenizer = \
            components["text_encoder"], components["unet"], components["vae"], components["tokenizer"]
        text.save_pretrained(f"{directory}/text/")
        unet.save_pretrained(f"{directory}/unet/")
        vae.save_pretrained(f"{directory}/vae/")

    def compile_text(self, directory = "./model_weights", tune=False, run_tune=False):
        input_data = torch.tensor([[49406,  1322,   268,  5352,  2870,  3086,   267,   320,  1215, 36650,
           593,  3878,  2225,   267,   279,   330, 49407, 49407, 49407, 49407,
         49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
         49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
         49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
         49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
         49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
         49407, 49407, 49407, 49407, 49407, 49407, 49407]], dtype=torch.int64).cuda()
        input_name = "input0"
        # target = tvm.target.cuda(model="A5000", arch="sm_86")
        shape = input_data.shape
        model = CLIPTextModel.from_pretrained(f"{directory}/text/", return_dict=False).cuda()
        shape_list = [(input_name, shape)]
        scripted_model = torch.jit.trace(model, input_data).eval()
        mod, params = tvm.relay.frontend.from_pytorch(scripted_model, shape_list)

        target = tvm.target.Target.from_device("cuda")

        if tune:
            import tvm.auto_scheduler as auto_scheduler
            from tvm.autotvm.tuner import XGBTuner
            from tvm import autotvm
            number = 10
            repeat = 1
            min_repeat_ms = 150  # 150ms is usually setted for NVIDIA GPU
            timeout = 1000000  # in seconds
            # create a TVM runner
            runner = autotvm.LocalRunner(
                number=number,
                repeat=repeat,
                timeout=timeout,
                min_repeat_ms=min_repeat_ms,
                enable_cpu_cache_flush=True,
            )
            tuning_option = {
                "tuner": "xgb",
                "trials": 2000,
                "early_stopping": 600,
                "measure_option": autotvm.measure_option(
                    builder=autotvm.LocalBuilder(build_func="default"), runner=runner
                ),
                "tuning_records": "text-autotuning.json",
            }
            if run_tune:

                # begin by extracting the tasks from the onnx model
                tasks = autotvm.task.extract_from_program(mod["main"], target=target, params=params)

                # Tune the extracted tasks sequentially.
                for i, task in enumerate(tasks):
                    prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))

                    # choose tuner
                    tuner = "xgb"

                    # create tuner
                    if tuner == "xgb":
                        tuner_obj = XGBTuner(task, loss_type="reg")
                    elif tuner == "xgb_knob":
                        tuner_obj = XGBTuner(task, loss_type="reg", feature_type="knob")
                    elif tuner == "xgb_itervar":
                        tuner_obj = XGBTuner(task, loss_type="reg", feature_type="itervar")
                    elif tuner == "xgb_curve":
                        tuner_obj = XGBTuner(task, loss_type="reg", feature_type="curve")
                    elif tuner == "xgb_rank":
                        tuner_obj = XGBTuner(task, loss_type="rank")
                    elif tuner == "xgb_rank_knob":
                        tuner_obj = XGBTuner(task, loss_type="rank", feature_type="knob")
                    elif tuner == "xgb_rank_itervar":
                        tuner_obj = XGBTuner(task, loss_type="rank", feature_type="itervar")
                    elif tuner == "xgb_rank_curve":
                        tuner_obj = XGBTuner(task, loss_type="rank", feature_type="curve")
                    elif tuner == "xgb_rank_binary":
                        tuner_obj = XGBTuner(task, loss_type="rank-binary")
                    elif tuner == "xgb_rank_binary_knob":
                        tuner_obj = XGBTuner(task, loss_type="rank-binary", feature_type="knob")
                    elif tuner == "xgb_rank_binary_itervar":
                        tuner_obj = XGBTuner(task, loss_type="rank-binary", feature_type="itervar")
                    elif tuner == "xgb_rank_binary_curve":
                        tuner_obj = XGBTuner(task, loss_type="rank-binary", feature_type="curve")
                    elif tuner == "ga":
                        tuner_obj = GATuner(task, pop_size=50)
                    elif tuner == "random":
                        tuner_obj = RandomTuner(task)
                    elif tuner == "gridsearch":
                        tuner_obj = GridSearchTuner(task)
                    else:
                        raise ValueError("Invalid tuner: " + tuner)

                    tuner_obj.tune(
                        n_trial=min(tuning_option["trials"], len(task.config_space)),
                        early_stopping=tuning_option["early_stopping"],
                        measure_option=tuning_option["measure_option"],
                        callbacks=[
                            autotvm.callback.progress_bar(tuning_option["trials"], prefix=prefix),
                            autotvm.callback.log_to_file(tuning_option["tuning_records"]),
                        ],
                    )
            with autotvm.apply_history_best(tuning_option["tuning_records"]):
                with tvm.transform.PassContext(opt_level=3, config={}):
                    # mod, params = tvm.relay.optimize(mod, target=target, params=params)
                    lib = tvm.relay.build(mod, target=target, params=params)

        else:
            # with tvm.transform.PassContext(opt_level=3, disabled_pass=["OpFusion"]):
            with tvm.transform.PassContext(opt_level=3):
                # mod = tvm.relay.transform.FuseOps(fuse_opt_level=0)(mod)
                # mod, params = tvm.relay.optimize(mod, target=target, params=params)
                lib = tvm.relay.build(mod, target=target, params=params)

        lib.export_library("tvm_lib/text_encoder.so")


    '''
        def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        down_intrablock_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[UNet2DConditionOutput, Tuple]:
    '''
    def compile_unet(self, directory = "./model_weights", tune=False, run_tune=False):

        input_dir = "/data/hyou37/xluo/LCM/sample_input/unet"
        timestep=torch.tensor([279], dtype=torch.int64).cuda()
        return_dict=False
        encoder_hidden_states = torch.load(f"{input_dir}/encoder_hidden_states.tensor").cuda().to(dtype=torch.float32)
        sample = torch.load(f"{input_dir}/sample.tensor").cuda().to(dtype=torch.float32)
        timestep_cond = torch.load(f"{input_dir}/timestep_cond.tensor").cuda().to(dtype=torch.float32)
        none_tensor = torch.tensor([0]).cuda()

        input_data = (sample, timestep, encoder_hidden_states, none_tensor, timestep_cond, none_tensor, none_tensor, none_tensor, none_tensor, none_tensor, none_tensor, none_tensor, none_tensor)
        # target = tvm.target.cuda(model="A5000", arch="sm_86")
        # model = UNet2DConditionModel.from_pretrained(f"{directory}/unet/", return_dict=False).cuda()
        # model = model.to(dtype=torch.float16)
        model = self.pipe.components["unet"]
        # shape_list = [(input_name, len(input_data))]
        shape_list = [("sample", sample.shape), 
                      ("timestep", timestep.shape), 
                      ("encoder_hidden_states", encoder_hidden_states.shape),
                      ("class_labels", none_tensor.shape),
                      ("timestep_cond", timestep_cond.shape),
                      ("attention_mask", none_tensor.shape),
                      ("cross_attention_kwargs", none_tensor.shape),
                      ("added_cond_kwargs", none_tensor.shape),
                      ("down_block_additional_residuals", none_tensor.shape),
                      ("mid_block_additional_residual", none_tensor.shape),
                      ("down_intrablock_additional_residuals", none_tensor.shape),
                      ("encoder_attention_mask", none_tensor.shape),
                      ("return_dict", none_tensor.shape)]
        scripted_model = torch.jit.trace(model, input_data).eval()
        mod, params = tvm.relay.frontend.from_pytorch(scripted_model, shape_list)

        target = tvm.target.Target.from_device("cuda")

        if tune:
            import tvm.auto_scheduler as auto_scheduler
            from tvm.autotvm.tuner import XGBTuner
            from tvm import autotvm
            number = 10
            repeat = 1
            min_repeat_ms = 150  # 150ms is usually setted for NVIDIA GPU
            timeout = 1000000  # in seconds
            # create a TVM runner
            runner = autotvm.LocalRunner(
                number=number,
                repeat=repeat,
                timeout=timeout,
                min_repeat_ms=min_repeat_ms,
                enable_cpu_cache_flush=True,
            )
            tuning_option = {
                "tuner": "xgb",
                "trials": 2000,
                "early_stopping": 600,
                "measure_option": autotvm.measure_option(
                    builder=autotvm.LocalBuilder(build_func="default"), runner=runner
                ),
                "tuning_records": "unet-autotuning.json",
            }
            if run_tune:

                # begin by extracting the tasks from the onnx model
                tasks = autotvm.task.extract_from_program(mod["main"], target=target, params=params)

                # Tune the extracted tasks sequentially.
                for i, task in enumerate(tasks):
                    prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))

                    # choose tuner
                    tuner = "xgb"

                    # create tuner
                    if tuner == "xgb":
                        tuner_obj = XGBTuner(task, loss_type="reg")
                    elif tuner == "xgb_knob":
                        tuner_obj = XGBTuner(task, loss_type="reg", feature_type="knob")
                    elif tuner == "xgb_itervar":
                        tuner_obj = XGBTuner(task, loss_type="reg", feature_type="itervar")
                    elif tuner == "xgb_curve":
                        tuner_obj = XGBTuner(task, loss_type="reg", feature_type="curve")
                    elif tuner == "xgb_rank":
                        tuner_obj = XGBTuner(task, loss_type="rank")
                    elif tuner == "xgb_rank_knob":
                        tuner_obj = XGBTuner(task, loss_type="rank", feature_type="knob")
                    elif tuner == "xgb_rank_itervar":
                        tuner_obj = XGBTuner(task, loss_type="rank", feature_type="itervar")
                    elif tuner == "xgb_rank_curve":
                        tuner_obj = XGBTuner(task, loss_type="rank", feature_type="curve")
                    elif tuner == "xgb_rank_binary":
                        tuner_obj = XGBTuner(task, loss_type="rank-binary")
                    elif tuner == "xgb_rank_binary_knob":
                        tuner_obj = XGBTuner(task, loss_type="rank-binary", feature_type="knob")
                    elif tuner == "xgb_rank_binary_itervar":
                        tuner_obj = XGBTuner(task, loss_type="rank-binary", feature_type="itervar")
                    elif tuner == "xgb_rank_binary_curve":
                        tuner_obj = XGBTuner(task, loss_type="rank-binary", feature_type="curve")
                    elif tuner == "ga":
                        tuner_obj = GATuner(task, pop_size=50)
                    elif tuner == "random":
                        tuner_obj = RandomTuner(task)
                    elif tuner == "gridsearch":
                        tuner_obj = GridSearchTuner(task)
                    else:
                        raise ValueError("Invalid tuner: " + tuner)

                    tuner_obj.tune(
                        n_trial=min(tuning_option["trials"], len(task.config_space)),
                        early_stopping=tuning_option["early_stopping"],
                        measure_option=tuning_option["measure_option"],
                        callbacks=[
                            autotvm.callback.progress_bar(tuning_option["trials"], prefix=prefix),
                            autotvm.callback.log_to_file(tuning_option["tuning_records"]),
                        ],
                    )
            with autotvm.apply_history_best(tuning_option["tuning_records"]):
                with tvm.transform.PassContext(opt_level=3, config={}):
                    # mod, params = tvm.relay.optimize(mod, target=target, params=params)
                    lib = tvm.relay.build(mod, target=target, params=params)

        else:
            # with tvm.transform.PassContext(opt_level=3, disabled_pass=["OpFusion"]):
            with tvm.transform.PassContext(opt_level=3):
                # mod = tvm.relay.transform.FuseOps(fuse_opt_level=0)(mod)
                # mod, params = tvm.relay.optimize(mod, target=target, params=params)
                lib = tvm.relay.build(mod, target=target, params=params)

        lib.export_library("tvm_lib/unet.so")


    def run_lib(self, lib_name):
        lib_path = {
            "text": "tvm_lib/text_encoder.so"
        }
        lib = tvm.runtime.load_module(lib_path[lib_name])
        input_data = torch.tensor([[49406,  1322,   268,  5352,  2870,  3086,   267,   320,  1215, 36650,
           593,  3878,  2225,   267,   279,   330, 49407, 49407, 49407, 49407,
         49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
         49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
         49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
         49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
         49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
         49407, 49407, 49407, 49407, 49407, 49407, 49407]], dtype=torch.int64).cuda()
        dev = tvm.cuda()
        from tvm.contrib import graph_executor
        m = graph_executor.GraphModule(lib["default"](dev))
        
        input_name = "input0"
        model = CLIPTextModel.from_pretrained(f"./model_weights/text/", return_dict=False).eval().cuda()

        m.set_input(input_name, tvm.nd.from_dlpack(torch.utils.dlpack.to_dlpack(input_data)))
        for _ in range(10):
            m.run()
            model(input_data)

        
        torch.cuda.synchronize()
        st = time.time()
        # m.set_input(input_name, tvm.nd.from_dlpack(torch.utils.dlpack.to_dlpack(input_data)))
        m.run()
        #tvm_output = (m.get_output(0), m.get_output(1))
        torch.cuda.synchronize()
        print(f"tvm runs {time.time()-st} secs")
        st = time.time()
        
        ans = model(input_data)
        torch.cuda.synchronize()
        print(f"torch runs {time.time()-st} secs")
        # print((tvm_output[0].numpy() - ans[0].detach().cpu().numpy())/ans[0].detach().cpu().numpy())
        # print((tvm_output[1].numpy() - ans[1].detach().cpu().numpy())/ans[1].detach().cpu().numpy())

    def profile_text(self, test_target):
        lib_path = "tvm_lib/text_encoder.so"
        lib = tvm.runtime.load_module(lib_path)
        input_data = torch.tensor([[49406,  1322,   268,  5352,  2870,  3086,   267,   320,  1215, 36650,
           593,  3878,  2225,   267,   279,   330, 49407, 49407, 49407, 49407,
         49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
         49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
         49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
         49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
         49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
         49407, 49407, 49407, 49407, 49407, 49407, 49407]], dtype=torch.int64).cuda()
        dev = tvm.cuda()
        from tvm.contrib import graph_executor
        m = graph_executor.GraphModule(lib["default"](dev))
        
        input_name = "input0"
        model = CLIPTextModel.from_pretrained(f"./model_weights/text/", return_dict=False).eval().cuda()

        tvm_input = tvm.nd.from_dlpack(torch.utils.dlpack.to_dlpack(input_data))
        m.set_input(input_name, tvm_input)
        for _ in range(10):
            if test_target == "tvm":
                m.run()
            if test_target == "vanilla":
                model(input_data)

        tvm.cuda(0).sync()

        run_time = 10
        torch.cuda.synchronize()
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
            on_trace_ready=get_trace_handler(test_target)
            # on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')
            # used when outputting for tensorboard
        ) as p:
            for iter in range(run_time):
                if test_target == "tvm":
                    m.run()
                if test_target == "vanilla":
                    model(input_data)
                torch.cuda.synchronize()
                p.step()

    def profile_unet(self, test_target):
        input_dir = "/data/hyou37/xluo/LCM/sample_input/unet"
        timestep=torch.tensor([279], dtype=torch.int64).cuda()
        return_dict=False
        encoder_hidden_states = torch.load(f"{input_dir}/encoder_hidden_states.tensor").cuda().to(dtype=torch.float32)
        sample = torch.load(f"{input_dir}/sample.tensor").cuda().to(dtype=torch.float32)
        timestep_cond = torch.load(f"{input_dir}/timestep_cond.tensor").cuda().to(dtype=torch.float32)
        none_tensor = torch.tensor([0]).cuda()

        input_data = (sample, timestep, encoder_hidden_states, none_tensor, timestep_cond, none_tensor, none_tensor, none_tensor, none_tensor, none_tensor, none_tensor, none_tensor, none_tensor)
        # target = tvm.target.cuda(model="A5000", arch="sm_86")

        lib_path = "tvm_lib/unet.so"
        lib = tvm.runtime.load_module(lib_path)

        dev = tvm.cuda()
        from tvm.contrib import graph_executor
        m = graph_executor.GraphModule(lib["default"](dev))
        
        input_data = [("sample", sample), 
            ("timestep", timestep), 
            ("encoder_hidden_states", encoder_hidden_states),
            ("class_labels", none_tensor),
            ("timestep_cond", timestep_cond),
            ("attention_mask", none_tensor),
            ("cross_attention_kwargs", none_tensor),
            ("added_cond_kwargs", none_tensor),
            ("down_block_additional_residuals", none_tensor),
            ("mid_block_additional_residual", none_tensor),
            ("down_intrablock_additional_residuals", none_tensor),
            ("encoder_attention_mask", none_tensor),
            ("return_dict", none_tensor)]
        model = self.pipe.components["unet"]

        # tvm_input = tvm.nd.from_dlpack(torch.utils.dlpack.to_dlpack(input_data))
        tvm_input = {}
        model_input = {}
        for key, value in input_data:
            tvm_input[key] = tvm.nd.from_dlpack(torch.utils.dlpack.to_dlpack(value))
            model_input[key] = value
        m.set_input(**tvm_input)
        for _ in range(10):
            if test_target == "tvm":
                m.run()
            if test_target == "vanilla":
                model(**model_input)

        tvm.cuda(0).sync()

        run_time = 10
        torch.cuda.synchronize()
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
            on_trace_ready=get_trace_handler(f"unet/{test_target}")
            # on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')
            # used when outputting for tensorboard
        ) as p:
            for iter in range(run_time):
                if test_target == "tvm":
                    m.run()
                if test_target == "vanilla":
                    model(**model_input)
                torch.cuda.synchronize()
                p.step()


    

def parse_argument():
    parser = ArgumentParser()
    parser.add_argument("--prompt", type=str, default="Self-portrait oil painting, a beautiful cyborg with golden hair, 8k",
                        help="Input prompt for the image generation.")
    parser.add_argument("--width", type=int, default=512,
                        help="Width of the output image. Lower the setting if out of memory.")
    parser.add_argument("--height", type=int, default=512,
                        help="Height of the output image. Lower the setting if out of memory.")
    parser.add_argument("--num_images", type=int, default=1, choices=range(1, 5),
                        help="Number of images to output. Range: 1-4.")
    parser.add_argument("--num_inference_steps", type=int, default=4, choices=range(1, 51),
                        help="Number of denoising steps. Recommend: 1~8 steps.")
    parser.add_argument("--guidance_scale", type=float, default=8.0, 
                        help="Scale for classifier-free guidance. Range: 1-20.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed. Leave blank to randomize the seed.")
    
    parser.add_argument("--repeat", type=int, default=1,
                    help="how many inference you like to do")
    parser.add_argument("--quant_text", type=bool, default=False,
                    help="quant text encoder")
    parser.add_argument("--record_text", type=bool, default=False,
                    help="record text encoder")
    parser.add_argument("--quant_unet", type=bool, default=False,
                    help="quant unet")
    parser.add_argument("--record_unet", type=bool, default=False,
                    help="record unet")
    parser.add_argument("--forward_timed_dir", type=str, default="./module_latency",
                    help="whether to save record result")
    
    parser.add_argument("--profile", type=bool, default=False,
                help="enable profile")

    return parser.parse_args()


args = parse_argument()
predictor = Predictor()
predictor.setup(args)
#predictor.save_model()
# predictor.compile_text(tune=True, run_tune=False)
# predictor.run_lib("text")
# predictor.profile_lib("tvm")
# predictor.profile_lib("vanilla")
#predictor.compile_unet(tune=False, run_tune=False)
predictor.profile_unet(test_target="tvm")
predictor.profile_unet(test_target="vanilla")
if args.profile:
    predictor.profile(args)
    exit()

# for _ in range(args.repeat):
#     predictor.predict(args)