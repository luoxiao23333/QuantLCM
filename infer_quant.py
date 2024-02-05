from __future__ import annotations

import os
import random
import time

import gradio as gr
import numpy as np
import PIL.Image
import torch
try:
    import intel_extension_for_pytorch as ipex
except:
    pass

from diffusers import DiffusionPipeline
import torch

import os
import torch
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor
import uuid
from copy import deepcopy

DESCRIPTION = '''# Latent Consistency Model
Distilled from [Dreamshaper v7](https://huggingface.co/Lykon/dreamshaper-7) fine-tune of [Stable Diffusion v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5) with only 4,000 training iterations (~32 A100 GPU Hours). [Project page](https://latent-consistency-models.github.io)
'''
if torch.cuda.is_available():
    DESCRIPTION += "\n<p>Running on CUDA 😀</p>"
elif hasattr(torch, 'xpu') and torch.xpu.is_available():
    DESCRIPTION += "\n<p>Running on XPU 🤓</p>"
else:
    DESCRIPTION += "\n<p>Running on CPU 🥶 This demo does not work on CPU.</p>"

MAX_SEED = np.iinfo(np.int32).max
CACHE_EXAMPLES = torch.cuda.is_available() and os.getenv("CACHE_EXAMPLES") == "1"
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "768"))
USE_TORCH_COMPILE = os.getenv("USE_TORCH_COMPILE") == "1"

def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed

def save_image(img, profile: gr.OAuthProfile | None, metadata: dict, root_path='./'):
    unique_name = str(uuid.uuid4()) + '.png'
    unique_name = os.path.join(root_path, unique_name)
    img.save(unique_name)
    # gr_user_history.save_image(label=metadata["prompt"], image=img, profile=profile, metadata=metadata)
    return unique_name

def save_images(image_array, profile: gr.OAuthProfile | None, metadata: dict):
    paths = []
    root_path = './images/'
    os.makedirs(root_path, exist_ok=True)
    with ThreadPoolExecutor() as executor:
        paths = list(executor.map(save_image, image_array, [profile]*len(image_array), [metadata]*len(image_array), [root_path]*len(image_array)))
    return paths


def generate(
    prompt: str,
    seed: int = 0,
    width: int = 512,
    height: int = 512,
    guidance_scale: float = 8.0,
    num_inference_steps: int = 4,
    num_images: int = 4,
    randomize_seed: bool = False,
    param_dtype='torch.float16',
    progress = gr.Progress(track_tqdm=True),
    profile: gr.OAuthProfile | None = None,
) -> PIL.Image.Image:
    seed = randomize_seed_fn(seed, randomize_seed)
    torch.manual_seed(seed)
    pipe.to(torch_device=device, torch_dtype=torch.float16 if param_dtype == 'torch.float16' else torch.float32)
    start_time = time.time()
    result = pipe(
        prompt=prompt,
        width=width,
        height=height,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        num_images_per_prompt=num_images,
        lcm_origin_steps=50,
        output_type="pil",
    ).images
    paths = save_images(result, profile, metadata={"prompt": prompt, "seed": seed, "width": width, "height": height, "guidance_scale": guidance_scale, "num_inference_steps": num_inference_steps})
    print(time.time() - start_time)
    return paths, seed

"""
   Operation System Options:
      If you are using MacOS, please set the following (device="mps") ;
      If you are using Linux & Windows with Nvidia GPU, please set the device="cuda";
      If you are using Linux & Windows with Intel Arc GPU, please set the device="xpu";
"""
# device = "mps"    # MacOS
# device = "xpu"    # Intel Arc GPU
device = "cuda"   # Linux & Windows

examples = [
    "portrait photo of a girl, photograph, highly detailed face, depth of field, moody light, golden hour, style by Dan Winters, Russell James, Steve McCurry, centered, extremely detailed, Nikon D850, award winning photography",
    "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k",
    "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
    "A photo of beautiful mountain with realistic sunset and blue lake, highly detailed, masterpiece",
]


"""
   DTYPE Options:
      To reduce GPU memory you can set "DTYPE=torch.float16",
      but image quality might be compromised
"""
DTYPE = torch.float16  # torch.float16 works as well, but pictures seem to be a bit worse

pipe = DiffusionPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7")
# pipe = DiffusionPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7", custom_pipeline="latent_consistency_txt2img", custom_revision="main")
pipe.to(torch_device=device, torch_dtype=DTYPE)

from model_analyze import analyze, model_memory_usage, print_tensor_dtypes
from model_analyze import write_model_arch
from quant.text_encoder import INT8CLIPTextModel
text_encoder = deepcopy(pipe.components['text_encoder'])
'''
attn_input_scale: float,
q_output_scale: float,
k_output_scale: float,
v_output_scale: float,
out_input_scale: float,
fc1_input_scale: float,
fc2_input_scale: float
'''

int8_text_encoder = INT8CLIPTextModel.from_float(text_encoder, encoder_layer_scales=[{
    "attn_input_scale": 1,
    "q_output_scale": 1,
    "k_output_scale": 1,
    "v_output_scale": 1,
    "out_input_scale": 1,
    "fc1_input_scale": 1,
    "fc2_input_scale": 1,
} for _ in range(12)])
def encode(text_encoder):
    tokenizer = pipe.components["tokenizer"]
    prompt = examples[0]
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
    if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            text_input_ids, untruncated_ids
        ):
            removed_text = tokenizer.batch_decode(
                untruncated_ids[:, tokenizer.model_max_length - 1 : -1]
            )

    if hasattr(text_encoder.config, "use_attention_mask") and text_encoder.config.use_attention_mask:
        attention_mask = text_inputs.attention_mask.to(device)
    else:
        attention_mask = None

    prompt_embeds = text_encoder(
        text_input_ids.to(device),
        attention_mask=attention_mask,
    )
    prompt_embeds = prompt_embeds[0]
    return prompt_embeds


fp16 = encode(text_encoder)
pipe.components['text_encoder'] = int8_text_encoder
i8 = encode(int8_text_encoder)
print(fp16)
print(i8)
#print(int8_text_encoder)
#print(f"int8: {model_memory_usage(int8_text_encoder)}")
#print(f"f32: {model_memory_usage(text_encoder)}")
# generate("portrait photo of a girl, photograph, highly detailed face, depth of field, moody light, golden hour, style by Dan Winters, Russell James, Steve McCurry, centered, extremely detailed, Nikon D850, award winning photography")
# copy_and_report_attributes(text_encoder, int8_text_encoder)







