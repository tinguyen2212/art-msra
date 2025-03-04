import os
import math
import torch
import argparse
from PIL import Image
from multi_layer_gen.custom_model_mmdit import CustomFluxTransformer2DModel
from multi_layer_gen.custom_model_transp_vae import AutoencoderKLTransformerTraining as CustomVAE
from multi_layer_gen.custom_pipeline import CustomFluxPipelineCfg

def test_sample(pipeline, transp_vae, batch, args):

    def adjust_coordinate(value, floor_or_ceil, k=16, min_val=0, max_val=1024):
        # Round the value to the nearest multiple of k
        if floor_or_ceil == "floor":
            rounded_value = math.floor(value / k) * k
        else:
            rounded_value = math.ceil(value / k) * k
        # Clamp the value between min_val and max_val
        return max(min_val, min(rounded_value, max_val))

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
    
    generator = torch.Generator(device=torch.device("cuda", index=args.gpu_id)).manual_seed(args.seed) if args.seed else None
    output, rgba_output, _, _ = pipeline(
        prompt=validation_prompt,
        validation_box=validation_box,
        generator=generator,
        height=args.resolution,
        width=args.resolution,
        num_layers=len(validation_box),
        guidance_scale=args.cfg,
        num_inference_steps=args.steps,
        transparent_decoder=transp_vae,
    )
    images = output.images   # list of PIL, len=layers
    rgba_images = [Image.fromarray(arr, 'RGBA') for arr in rgba_output]

    os.makedirs(os.path.join(args.save_dir, this_index), exist_ok=True)
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


args = dict(
    save_dir="output/",
    resolution=512,
    cfg=4.0,
    steps=28,
    seed=41,
    gpu_id=0,
)
args = argparse.Namespace(**args)

transformer = CustomFluxTransformer2DModel.from_pretrained("ART-Release/ART_v1.0", subfolder="transformer", torch_dtype=torch.bfloat16)
transp_vae = CustomVAE.from_pretrained("ART-Release/ART_v1.0", subfolder="transp_vae", torch_dtype=torch.float32)
pipeline = CustomFluxPipelineCfg.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    transformer=transformer,
    torch_dtype=torch.bfloat16,
).to(torch.device("cuda", index=args.gpu_id))
pipeline.enable_model_cpu_offload(gpu_id=args.gpu_id) # Save GPU memory

sample = {
    "index": "reso512_3",
    "wholecaption": 'Floral wedding invitation: green leaves, white flowers; circular border. Center: "JOIN US CELEBRATING OUR WEDDING" (cursive), "DONNA AND HARPER" (bold), "03 JUNE 2023" (small bold). White, green color scheme, elegant, natural.',
    "layout": [(0, 0, 512, 512), (0, 0, 512, 512), (0, 0, 512, 352), (144, 384, 368, 448), (160, 192, 352, 432), (368, 0, 512, 144), (0, 0, 144, 144), (128, 80, 384, 208), (128, 448, 384, 496), (176, 48, 336, 80)],
}

test_sample(pipeline=pipeline, transp_vae=transp_vae, batch=sample, args=args)

del pipeline
if torch.cuda.is_available():
    torch.cuda.empty_cache()