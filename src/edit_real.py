import os, pdb

import argparse
import numpy as np
import torch
import requests
import glob
from PIL import Image
from tqdm import tqdm

from diffusers import DDIMScheduler
from utils.ddim_inv import DDIMInversion
from utils.edit_directions import construct_direction
from utils.edit_pipeline import EditingPipeline

import matplotlib.pyplot as plt

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

mask_dimensions = [64, 32, 16, 8] # all the dimensions required in our unet

def max_pooling(image, pool_size):
    # Determine the shape of the image
    height, width = image.shape

    # Calculate the shape of the output matrix
    new_height = height // pool_size
    new_width = width // pool_size

    # Initialize the output matrix
    pooled_image = np.zeros((new_height, new_width))

    # Perform max pooling
    for i in range(new_height):
        for j in range(new_width):
            h_start = i * pool_size
            h_end = h_start + pool_size
            w_start = j * pool_size
            w_end = w_start + pool_size

            # Extract the current block and find its max value
            pooled_image[i, j] = np.max(image[h_start:h_end, w_start:w_end])

    return pooled_image

# Returns binary mask where 0 is black, and 1 is white (the mask of interest)
def image_to_binary_masks(image_paths):
    # Load the image
    masks = {}

    counter = 0

    for image_path in image_paths:
        with Image.open(image_path) as img:
            # Convert image to grayscale
            img = img.convert('L')

            for dim in mask_dimensions:
                resized_img = img.resize((1024, 1024), Image.BICUBIC)
                # resized_img = img.resize((dim, dim), Image.BICUBIC)
                img_array = max_pooling(np.array(resized_img), 1024 // dim)

                # Convert image to a NumPy array
                # img_array = np.array(resized_img)

                # Apply a threshold to create a binary matrix (0 for black, 1 for white)
                # Assuming the threshold for differentiating black and white is 128
                binary_matrix = (img_array > 128).astype(int)

                gaussian_blur = False

                if gaussian_blur:
                    # adding gaussian blur
                    import cv2
                    
                    sigma = max(binary_matrix.shape) / 32.0
                    # Calculate an appropriate kernel size (an odd integer)
                    kernel_size = int(6 * sigma) | 1
                    # Convert the binary mask to a float32 image
                    binary_matrix_float = binary_matrix.astype(np.float32)
                    # Apply Gaussian blur
                    blurred_mask = cv2.GaussianBlur(binary_matrix_float, (kernel_size, kernel_size), sigma)
                    # Normalize the blurred mask to the range [0, 1]
                    blurred_mask = blurred_mask / np.max(blurred_mask)
                    
                    # binary_mask_img = binary_matrix.astype(np.float32)
                    # binary_mask_img = (binary_mask_img * 255).astype(np.uint8)

                    # blurred_mask_img = (blurred_mask * 255).astype(np.uint8)
                    # print(" ???", kernel_size, max(binary_matrix.shape))
                    # cv2.imwrite(f'./temp/binary_mask_{counter}.png', binary_mask_img)
                    # cv2.imwrite(f'./temp/blurred_mask_{counter}.png', blurred_mask_img)
                    # counter += 1

                    binary_matrix = blurred_mask
                

                if dim not in masks:
                    masks[dim] = torch.tensor(binary_matrix).to(device)
                else:
                    masks[dim] = torch.maximum(masks[dim], torch.tensor(binary_matrix).to(device))

    # for mask in masks.values():
    #     plt.imshow(mask.cpu())
    #     plt.show()
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
    parser.add_argument('--masks', type=str, nargs='+', help="List of mask images to be used, which will be unioned together")
    parser.add_argument('--mask_outside_scaling_factor', type=float, default=1.5)
    parser.add_argument('--mask_inside_scaling_factor', type=float, default=0.7)
    parser.add_argument('--guidance_steps', type=int, nargs='+')
    parser.add_argument('--mask_explore_radius', type=float)
    parser.add_argument('--mask_explore_steps', type=int)

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
    if args.masks:
        masks = image_to_binary_masks(args.masks)
    else:
        masks = None

    # Make the editing pipeline
    pipe = EditingPipeline.from_pretrained(args.model_path, torch_dtype=torch_dtype).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    bname = os.path.basename(args.inversion).split(".")[0]

    if args.mask_explore_radius is None or args.mask_explore_steps is None:
        print("No hyperparameter search, assuming this is baseline")
        mask_outside_scaling_factors = [args.mask_outside_scaling_factor]
        mask_inside_scaling_factors = [args.mask_inside_scaling_factor]
    else:
        r = args.mask_explore_radius
        s = args.mask_explore_steps
        mo = args.mask_outside_scaling_factor
        mi = args.mask_inside_scaling_factor
        mask_outside_scaling_factors = np.linspace(mo - r, mo + r, s)[::-1]
        mask_inside_scaling_factors = np.linspace(mi - r, mi + r, s)[::-1]

    guidance_steps_list = args.guidance_steps
    if guidance_steps_list is None:
        guidance_steps_list = [ 1 ]

    for guidance_steps in tqdm(guidance_steps_list):
        for mask_outside_scaling_factor in tqdm(mask_outside_scaling_factors):
            for mask_inside_scaling_factor in tqdm(mask_inside_scaling_factors):
                mask_outside_scaling_factor = round(mask_outside_scaling_factor, 2)
                mask_inside_scaling_factor = round(mask_inside_scaling_factor, 2)
                print("Trying guidance_steps", guidance_steps, "mask_outside_scaling_factor", mask_outside_scaling_factor, "mask_inside_scaling_factor", mask_inside_scaling_factor)

                if args.masks is None:
                    dest_file = os.path.join(args.results_folder, f"edit/{bname}_{args.task_name}_{args.xa_guidance}_no_mask.png")
                else:
                    dest_file = os.path.join(args.results_folder, f"edit/{bname}_{args.task_name}_guidance_{guidance_steps}_{args.xa_guidance}_outside_{mask_outside_scaling_factor}_inside_{mask_inside_scaling_factor}.png")

                if os.path.exists(dest_file):
                    print("Skipping as edit file already exists")
                    continue

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
                            mask_outside_scaling_factor=mask_outside_scaling_factor,
                            mask_inside_scaling_factor=mask_inside_scaling_factor,
                            guidance_steps=guidance_steps
                    )
                    
                    edit_pil[0].save(dest_file)
                    rec_pil[0].save(os.path.join(args.results_folder, f"reconstruction/{bname}.png"))