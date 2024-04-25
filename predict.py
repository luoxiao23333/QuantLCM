
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
from diffusers.pipelines.latent_consistency_models.pipeline_latent_consistency_text2img import LatentConsistencyModelPipeline
from quant.utils import LatencyLogger
from quant.transformer_blocks import INTUNet2DConditionModel


def get_trace_handler(test, cpu=False):
    def trace_handler(prof):
        sort_key = "self_cuda_time_total" if not cpu else "self_cpu_time_total"
        if cpu:
            file_name = f"./trace/{test}_self_cpu"
        else:
            file_name = f"./trace/{test}"

        file = open(f"{file_name}.txt", "w+")
        print(prof.key_averages().table(
            sort_by=sort_key, row_limit=-1), file=file)
        prof.export_chrome_trace(f"{file_name}.json")
    return trace_handler


def scale_channel(scale_factor: float, model: torch.nn.Module):
    for name, module in model.named_children():
        if hasattr(module, "weight"):
            if len(module.weight.shape) >= 2:
                module.weight = torch.nn.Parameter(module.weight[:int(module.weight.shape[0]*scale_factor)][:int(module.weight.shape[1]*scale_factor)], requires_grad=False)
            else:
                module.weight = torch.nn.Parameter(module.weight[:int(module.weight.shape[0]*scale_factor)], requires_grad=False)
        if hasattr(module, "bias") and module.bias is not None:
            module.bias = torch.nn.Parameter(module.bias[:int(module.bias.shape[0]*scale_factor)], requires_grad=False)

        scale_channel(scale_factor, module)


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
        self.pipe = DiffusionPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7", local_files_only=False, cache_dir="./weights")
        
        if args.base_precision == "fp16":
            self.pipe.to(torch_device="cuda", torch_dtype=torch.float16)
        elif args.base_precision == "fp32":
            pass
        else:
            assert False, f"{args.base_precision} is not supportted, only support [fp16, fp32]"

        self.record_path = {
            "itext": f"{args.forward_timed_dir}/itext.txt",
            "otext": f"{args.forward_timed_dir}/otext.txt",
            "iunet": f"{args.forward_timed_dir}/iunet.txt",
            "ounet": f"{args.forward_timed_dir}/ounet.txt"
        }
        if args.channel_scale is not None:
            for key in self.record_path.keys():
                self.record_path[key] += f"_channelscale{args.channel_scale}"

        if args.quant_text == True:
            self.quant_text(args.record_text)
        elif args.record_text == True:
            self.time_forward("text")

        if args.quant_unet == True:
            self.quant_unet(args.record_unet)
        elif args.record_unet == True:
            self.time_forward("unet")

        if args.channel_scale is not None:
            assert False, "Not Implemented"
            if args.record_unet is False:
                assert False
            scale_channel(args.channel_scale, self.pipe.components["unet"])


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
        if args.save_image:
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
        profile_name = "quant_unet" if args.quant_unet else "original"
        if args.channel_scale is not None:
            profile_name += f"channelscale_{args.channel_scale}"

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
                active=1,
                repeat=1
            ),
            on_trace_ready=get_trace_handler(profile_name, args.sort_by_cpu)
            # on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')
            # used when outputting for tensorboard
        ) as p:
            for _ in range(5):
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
        components = getattr(self.pipe, 'components')
        unet = deepcopy(components['unet'])

        unet = INTUNet2DConditionModel.from_float(unet)
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
    
    parser.add_argument("--repeat", type=int, default=4,
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
    parser.add_argument("--sort_by_cpu", type=bool, default=False,
                help="profile table is sortted by self cpu or self gpu")
    parser.add_argument("--save_image", type=bool, default=False,
                help="whether to save the result")
    
    parser.add_argument("--intensive_infer", type=bool, default=False,
                help="intensive infer and profile")
    parser.add_argument("--base_precision", type=str, default="fp16", choices=["fp16", "fp32"],
                help="base LCM model precision")

    def range_limited_float_type(arg):
        try:
            value = float(arg)
        except ValueError:
            raise parser.ArgumentTypeError("Value must be a floating point number")
        if value <= 0 or value > 1:
            raise parser.ArgumentTypeError("Value must be within the range (0, 1]")
        return value
    parser.add_argument("--channel_scale", type=range_limited_float_type, default=None,
                help="set channel_scale")

    return parser.parse_args()


args = parse_argument()
predictor = Predictor()
predictor.setup(args)

if args.intensive_infer:
    while True:
        predictor.predict(args)
        time.sleep(0.1)

if args.profile:
    predictor.profile(args)
    exit()

for _ in range(args.repeat):
    predictor.predict(args)