#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import os
import sys
import math
import random
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from mmengine.config import Config

import torch
import torch.utils.checkpoint
from torchvision.utils import save_image

from diffusers import FluxTransformer2DModel
from diffusers.utils import check_min_version
from diffusers.configuration_utils import FrozenDict
from diffusers.loaders.peft import _SET_ADAPTER_SCALE_FN_MAPPING

from custom_model_mmdit import CustomFluxTransformer2DModel
from custom_model_transp_vae import AutoencoderKLTransformerTraining as CustomVAE
from custom_pipeline import CustomFluxPipelineCfg

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.31.0.dev0")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def parse_config(path=None):
    
    if path is None:
        parser = argparse.ArgumentParser()
        parser.add_argument('config_dir', type=str)
        args = parser.parse_args()
        path = args.config_dir
    config = Config.fromfile(path)
    
    config.config_dir = path

    if "LOCAL_RANK" in os.environ:
        config.local_rank = int(os.environ["LOCAL_RANK"])
    elif "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ:
        config.local_rank = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
    else:
        config.local_rank = -1

    return config


def initialize_pipeline(config, args):

    # Load the original Transformer model from the pretrained model
    transformer_orig = FluxTransformer2DModel.from_pretrained(
        config.transformer_varient if hasattr(config, "transformer_varient") else config.pretrained_model_name_or_path, 
        subfolder="" if hasattr(config, "transformer_varient") else "transformer", 
        revision=config.revision, 
        variant=config.variant,
        torch_dtype=torch.bfloat16,
        cache_dir=config.get("cache_dir", None),
    )
    
    # Configure the custom Transformer model
    mmdit_config = dict(transformer_orig.config)
    mmdit_config["_class_name"] = "CustomSD3Transformer2DModel"
    mmdit_config["max_layer_num"] = config.max_layer_num
    mmdit_config = FrozenDict(mmdit_config)
    transformer = CustomFluxTransformer2DModel.from_config(mmdit_config).to(dtype=torch.bfloat16)
    missing_keys, unexpected_keys = transformer.load_state_dict(transformer_orig.state_dict(), strict=False)

    # Fuse initial LoRA weights
    if args.pre_fuse_lora_dir is not None:
        lora_state_dict = CustomFluxPipelineCfg.lora_state_dict(args.pre_fuse_lora_dir)
        CustomFluxPipelineCfg.load_lora_into_transformer(lora_state_dict, None, transformer)
        transformer.fuse_lora(safe_fusing=True)
        transformer.unload_lora() # Unload LoRA parameters

    # Load layer_pe weights
    layer_pe_path = os.path.join(args.ckpt_dir, "layer_pe.pth")
    layer_pe = torch.load(layer_pe_path)
    missing_keys, unexpected_keys = transformer.load_state_dict(layer_pe, strict=False)

    # Initialize the custom pipeline
    pipeline_type = CustomFluxPipelineCfg
    pipeline = pipeline_type.from_pretrained(
        config.pretrained_model_name_or_path,
        transformer=transformer,
        revision=config.revision,
        variant=config.variant,
        torch_dtype=torch.bfloat16,
        cache_dir=config.get("cache_dir", None),
    ).to(torch.device("cuda", index=args.gpu_id))
    pipeline.enable_model_cpu_offload(gpu_id=args.gpu_id) # Save GPU memory

    # Load LoRA weights
    pipeline.load_lora_weights(args.ckpt_dir, adapter_name="layer")

    # Load additional LoRA weights
    if args.extra_lora_dir is not None:
        _SET_ADAPTER_SCALE_FN_MAPPING["CustomFluxTransformer2DModel"] = _SET_ADAPTER_SCALE_FN_MAPPING["FluxTransformer2DModel"]
        pipeline.load_lora_weights(args.extra_lora_dir, adapter_name="extra")
        pipeline.set_adapters(["layer", "extra"], adapter_weights=[1.0, 0.5])

    return pipeline

def get_fg_layer_box(list_layer_pt):
    list_layer_box = []
    for layer_pt in list_layer_pt:
        alpha_channel = layer_pt[:, 3:4]

        if layer_pt.shape[1] == 3:
            list_layer_box.append(
                (0, 0, layer_pt.shape[3], layer_pt.shape[2])
            )
            continue

        # Step 1: Find the non-zero indices
        _, _, rows, cols = torch.nonzero(alpha_channel + 1, as_tuple=True)

        if (rows.numel() == 0) or (cols.numel() == 0):
            # If there are no non-zero indices, we can skip this layer
            list_layer_box.append(None)
            continue

        # Step 2: Get the minimum and maximum indices for rows and columns
        min_row, max_row = rows.min().item(), rows.max().item()
        min_col, max_col = cols.min().item(), cols.max().item()

        # Step 3: Quantize the minimum values down to the nearest multiple of 16
        quantized_min_row = (min_row // 16) * 16
        quantized_min_col = (min_col // 16) * 16

        # Step 4: Quantize the maximum values up to the nearest multiple of 16 outside of the max
        quantized_max_row = ((max_row // 16) + 1) * 16
        quantized_max_col = ((max_col // 16) + 1) * 16
        list_layer_box.append(
            (quantized_min_col, quantized_min_row, quantized_max_col, quantized_max_row)
        )
    return list_layer_box


def adjust_coordinate(value, floor_or_ceil, k=16, min_val=0, max_val=1024):
    # Round the value to the nearest multiple of k
    if floor_or_ceil == "floor":
        rounded_value = math.floor(value / k) * k
    else:
        rounded_value = math.ceil(value / k) * k
    # Clamp the value between min_val and max_val
    return max(min_val, min(rounded_value, max_val))
    

def test(args):

    if args.seed is not None:
        seed_everything(args.seed)

    cfg_path = args.cfg_path
    config = parse_config(cfg_path)

    if args.variant is not None: args.save_dir += '_' + args.variant

    # Initialize pipeline
    pipeline = initialize_pipeline(config, args)

    # load multi-layer-transparent-vae-decoder
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    transp_vae = CustomVAE()
    transp_vae_path = args.transp_vae_ckpt
    missing, unexpected = transp_vae.load_state_dict(torch.load(transp_vae_path)['model'], strict=False)
    transp_vae.eval()

    test_samples = [
        {
            "index": "reso512_1",
            "wholecaption": "The image shows a collection of luggage items on a carpeted floor. There are three main pieces of luggage: a large suitcase, a smaller suitcase, and a duffel bag. The large suitcase is positioned in the center, with the smaller suitcase to its left and the duffel bag to its right. The luggage appears to be packed and ready for travel. In the foreground, there is a plastic bag containing what looks like a pair of shoes. The background features a white curtain, suggesting that the setting might be indoors, possibly a hotel room or a similar temporary accommodation. The image is in black and white, which gives it a timeless or classic feel.",
            "layout": [(0, 0, 512, 512), (0, 0, 512, 512), (281, 203, 474, 397), (94, 22, 294, 406), (190, 327, 379, 471)],
        },
        {
            "index": "reso512_2",
            "wholecaption": "The image features a logo for a flower shop named ”Estelle Darcy Flower Shop.” The logo is designed with a stylized flower,  which appears to be a rose, in shades of pink and green. The flower is positioned to the left of the text, which is written in a cursive font. The text is in a brown color, and the overall style of the image is simple and elegant, with a clean, light background that does not distract  from the logo itself. The logo conveys a sense of freshness and natural beauty, which is fitting for a flower shop.",
            "layout": [(0, 0, 512, 512), (0, 0, 512, 512), (320, 160, 432, 352), (128, 240, 368, 320), (128, 304, 352, 336)],
        },
    ]

    for idx, batch in tqdm(enumerate(test_samples)):

        generator = torch.Generator(device=torch.device("cuda", index=args.gpu_id)).manual_seed(args.seed) if args.seed else None

        this_index = batch["index"]

        validation_prompt = batch["wholecaption"]
        validation_box_raw = batch["layout"]
        validation_box = [
            (
                adjust_coordinate(rect[0], floor_or_ceil="floor"), 
                adjust_coordinate(rect[1], floor_or_ceil="floor"), 
                adjust_coordinate(rect[2], floor_or_ceil="ceil"), 
                adjust_coordinate(rect[3], floor_or_ceil="ceil"), 
            )
            for rect in validation_box_raw
        ]
        if len(validation_box) > 52:
            validation_box = validation_box[:52]
        
        output, rgba_output, _, _ = pipeline(
            prompt=validation_prompt,
            validation_box=validation_box,
            generator=generator,
            height=config.resolution,
            width=config.resolution,
            num_layers=len(validation_box),
            guidance_scale=args.cfg,
            num_inference_steps=args.steps,
            transparent_decoder=transp_vae,
        )
        images = output.images   # list of PIL, len=layers
        rgba_images = [Image.fromarray(arr, 'RGBA') for arr in rgba_output]

        os.makedirs(os.path.join(args.save_dir, this_index), exist_ok=True)
        os.system(f"rm -rf {os.path.join(args.save_dir, this_index)}/*")
        for frame_idx, frame_pil in enumerate(images):
            frame_pil.save(os.path.join(args.save_dir, this_index, f"layer_{frame_idx}.png"))
            if frame_idx == 0:
                frame_pil.save(os.path.join(args.save_dir, this_index, "merged.png"))
        merged_pil = images[1].convert('RGBA')
        for frame_idx, frame_pil in enumerate(rgba_images):
            if frame_idx < 2:
                frame_pil = images[frame_idx].convert('RGBA') # merged and background
            else:
                merged_pil = Image.alpha_composite(merged_pil, frame_pil)
            frame_pil.save(os.path.join(args.save_dir, this_index, f"layer_{frame_idx}_rgba.png"))
        
        merged_pil = merged_pil.convert('RGB')
        merged_pil.save(os.path.join(args.save_dir, this_index, "merged_rgba.png"))

    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", type=str)
    parser.add_argument("--ckpt_dir", type=str)
    parser.add_argument("--transp_vae_ckpt", type=str)
    parser.add_argument("--pre_fuse_lora_dir", type=str)
    parser.add_argument("--extra_lora_dir", type=str, default=None)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--variant", type=str, default="None")
    parser.add_argument("--cfg", type=float, default=4.0)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--gpu_id", type=int, default=0)

    args = parser.parse_args()

    test(args)