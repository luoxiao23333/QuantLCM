
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


def get_trace_handler(test):
    def trace_handler(prof):
        file = open("./trace/" + test + ".txt", "w+")
        print(prof.key_averages().table(
            sort_by="self_cuda_time_total", row_limit=-1), file=file)
        prof.export_chrome_trace("./trace/" + test + ".json")
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
        self.pipe.to(torch_device="cuda", torch_dtype=torch.float16)

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
    parser.add_argument("--quant_unet", type=bool, default=True,
                    help="quant unet")
    parser.add_argument("--record_unet", type=bool, default=False,
                    help="record unet")
    parser.add_argument("--forward_timed_dir", type=str, default="./module_latency",
                    help="whether to save record result")
    
    parser.add_argument("--profile", type=bool, default=True,
                help="enable profile")

    return parser.parse_args()


args = parse_argument()
predictor = Predictor()
predictor.setup(args)

if args.profile:
    predictor.profile(args)
    exit()

for _ in range(args.repeat):
    predictor.predict(args)