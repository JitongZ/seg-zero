import os, pdb

import argparse
import numpy as np
import torch
import requests
import glob
from PIL import Image

from diffusers import DDIMScheduler
from utils.ddim_inv import DDIMInversion
from utils.edit_directions import construct_direction
from utils.edit_pipeline import EditingPipeline

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

mask_dimensions = [64, 32, 16, 8] # all the dimensions required in our unet

# Returns binary mask where 0 is black, and 1 is white (the mask of interest)
def image_to_binary_masks(image_path):
    # Load the image
    with Image.open(image_path) as img:
        # Convert image to grayscale
        img = img.convert('L')
        masks = {}

        for dim in mask_dimensions:
            resized_img = img.resize((dim, dim))

            # Convert image to a NumPy array
            img_array = np.array(resized_img)

            # Apply a threshold to create a binary matrix (0 for black, 1 for white)
            # Assuming the threshold for differentiating black and white is 128
            binary_matrix = (img_array > 128).astype(int)

            masks[dim] = torch.tensor(binary_matrix).to(device).reshape(-1)

        return masks

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--inversion', required=True)
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--task_name', type=str, default='cat2dog')
    parser.add_argument('--results_folder', type=str, default='output/test_cat')
    parser.add_argument('--num_ddim_steps', type=int, default=50) # fanpu: could reduce this to make it faster
    parser.add_argument('--model_path', type=str, default='CompVis/stable-diffusion-v1-4')
    parser.add_argument('--xa_guidance', default=0.1, type=float)
    parser.add_argument('--negative_guidance_scale', default=5.0, type=float) # classifier-free guidance
    parser.add_argument('--use_float_16', action='store_true')
    parser.add_argument('--mask', type=str)
    parser.add_argument('--mask_outside_scaling_factor', type=float, default=1.5)
    parser.add_argument('--mask_inside_scaling_factor', type=float, default=0.7)

    # fanpu: set true to create a "attention.pkl" file that can
    # be visualized using visualize_attn.py
    parser.add_argument('--dump_attention', action='store_true')

    args = parser.parse_args()

    os.makedirs(os.path.join(args.results_folder, "edit"), exist_ok=True)
    os.makedirs(os.path.join(args.results_folder, "reconstruction"), exist_ok=True)

    if args.use_float_16:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    # if the inversion is a folder, the prompt should also be a folder
    assert (os.path.isdir(args.inversion)==os.path.isdir(args.prompt)), "If the inversion is a folder, the prompt should also be a folder"
    if os.path.isdir(args.inversion):
        l_inv_paths = sorted(glob.glob(os.path.join(args.inversion, "*.pt")))
        l_bnames = [os.path.basename(x) for x in l_inv_paths]
        l_prompt_paths = [os.path.join(args.prompt, x.replace(".pt",".txt")) for x in l_bnames]
    else:
        l_inv_paths = [args.inversion]
        l_prompt_paths = [args.prompt]

    # Load mask if it exists
    if args.mask:
        masks = image_to_binary_masks(args.mask)
    else:
        masks = None

    # Make the editing pipeline
    pipe = EditingPipeline.from_pretrained(args.model_path, torch_dtype=torch_dtype).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)


    for inv_path, prompt_path in zip(l_inv_paths, l_prompt_paths):
        prompt_str = open(prompt_path).read().strip()
        # CR fanpu: don't really get why they think using the unedited prompt
        # for the negative prompt is the right thing to do here
        rec_pil, edit_pil = pipe(prompt_str,
                num_inference_steps=args.num_ddim_steps,
                x_in=torch.load(inv_path).unsqueeze(0),
                edit_dir=construct_direction(args.task_name),
                guidance_amount=args.xa_guidance,
                guidance_scale=args.negative_guidance_scale,
                negative_prompt=prompt_str, # use the unedited prompt for the negative prompt
                dump_attention=args.dump_attention,
                masks=masks,
                mask_outside_scaling_factor=args.mask_outside_scaling_factor,
                mask_inside_scaling_factor=args.mask_inside_scaling_factor
        )
        
        bname = os.path.basename(args.inversion).split(".")[0]
        if args.mask is None:
            edit_pil[0].save(os.path.join(args.results_folder, f"edit/{bname}_no_mask.png"))
        else:
            edit_pil[0].save(os.path.join(args.results_folder, f"edit/{bname}_outside_{args.mask_outside_scaling_factor}_inside_{args.mask_inside_scaling_factor}.png"))
        rec_pil[0].save(os.path.join(args.results_folder, f"reconstruction/{bname}.png"))
