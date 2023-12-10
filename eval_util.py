from PIL import Image
import numpy as np
import os
import torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.multimodal.clip_score import CLIPScore
from statistics import mean

def resized_image_as_array(image_path, new_size=(512, 512)):
    image = Image.open(image_path)
    if image.size != (512, 512):
        image = image.resize((512,512), Image.Resampling.LANCZOS)
    image = np.array(image)
    return image

def mse(image1_path, image2_path, mask=None):
    image1 = resized_image_as_array(image1_path)
    image2 = resized_image_as_array(image2_path)

    # Mask the edited area if necessary
    if mask is not None:
        image1[mask] = 0
        image2[mask] = 0

    # Calculate the squared difference in the unmasked area
    squared_diff = np.square(image1 - image2)

    # Calculate the mean of the squared differences
    mse_value = np.mean(squared_diff)
    return mse_value

def LPIPS(image1_path, image2_path, mask=None):
    # make sure both images has shape (512, 512, 3)
    image1 = resized_image_as_array(image1_path)
    image2 = resized_image_as_array(image2_path)

    # LPIPS needs the images to be in the [-1, 1] range.
    image1 = image1 / 255.0 * 2 - 1
    image2 = image2 / 255.0 * 2 - 1

    # Mask the edited area if necessary
    if mask is not None:
        image1[mask] = 0
        image2[mask] = 0

    # tensors with shape [N, 3, H, W], current [H, W, 3]
    image1 = np.transpose(image1, (2, 0, 1))
    image2 = np.transpose(image2, (2, 0, 1))
    image1 = np.expand_dims(image1, axis=0)
    image2 = np.expand_dims(image2, axis=0)

    # need to turn np array into tensor
    image1 = torch.from_numpy(image1).float()
    image2 = torch.from_numpy(image2).float()    

    lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex')

    return lpips(image1, image2).item()

def CLIP(image1_path, prompt, mask=None):
    # make sure both images has shape (512, 512, 3)
    image1 = resized_image_as_array(image1_path)

    # Mask the edited area if necessary
    assert mask is None, "CLIP does not involve masking"

    # tensors with shape [N, 3, H, W], current [H, W, 3]
    image1 = np.transpose(image1, (2, 0, 1))
    image1 = np.expand_dims(image1, axis=0)

    # need to turn np array into tensor
    image1 = torch.from_numpy(image1).float()

    clip = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")
    score = clip(image1, prompt)
    score.detach()
    return score.item()

def resize_and_combine_masks(mask_lists, new_size=(512, 512)):
    combined_masks = []

    for mask_list in mask_lists:
        # Store the resized masks to combine later if necessary
        resized_masks = []

        for mask_path in mask_list:
            # Read and resize the mask
            mask = Image.open(mask_path)
            mask = mask.convert('L')
            mask = mask.resize(new_size)

            # Convert mask to binary array (0 and 1)
            mask_array = np.array(mask)
            # print("sum of pixels:", np.sum(mask_array > 230))
            mask_array = (mask_array > 127).astype(int)  # Assuming the mask is grayscale

            # Append to list of resized masks
            resized_masks.append(mask_array)

        # Combine masks by taking the element-wise maximum if there's more than one
        if len(resized_masks) > 1:
            combined_mask = np.maximum.reduce(resized_masks)
        else:
            combined_mask = resized_masks[0]

        combined_masks.append(combined_mask)

    return combined_masks

# func takes in two image paths, one mask (optional) and 
# returns a list of [file_name, eval_result]
def get_evals(func, path_list1, path_list2, mask_list=None):
    result = []
    if mask_list is None:
        for image1, image2 in zip(path_list1, path_list2):
            eval_result = func(image1, image2)
            file_name = image1.split('.')[-2].split('/')[-1]
            # print(f"MSE of {file_name}: {round(eval_result, 2)}")
            res = [file_name, eval_result]
            result.append(res)
    else:
        for image1, image2, mask in zip(path_list1, path_list2, mask_list):
            eval_result = func(image1, image2, mask)
            file_name = image1.split('.')[-2].split('/')[-1]
            # print(f"MSE of {file_name}: {round(eval_result, 2)}")
            res = [file_name, eval_result]
            result.append(res)
    return result

def print_result(ours, baseline):
    col_widths = [15, 10, 10, 10]
    print(f'{"Label":<{col_widths[0]}} {"Ours":<{col_widths[1]}} {"Baseline":<{col_widths[2]}} {"Diff":<{col_widths[3]}}')
    statistics = {"labels":[], "ours":[], "baseline":[], "diff":[]}
    for line in zip(ours, baseline):
        assert line[0][0] == line[1][0]
        label = line[0][0]
        ours_value = line[0][1]
        baseline_value = line[1][1]
        statistics["labels"].append(label)
        statistics["ours"].append(ours_value)
        statistics["baseline"].append(baseline_value)
        statistics["diff"].append(ours_value-baseline_value)
        print(f'{label:<{col_widths[0]}} {ours_value:<{col_widths[1]}.2f} {baseline_value:<{col_widths[2]}.2f} {ours_value-baseline_value:<{col_widths[3]}.2f}')
    # print average of ours and baseline
    ours_mean = mean([x[1] for x in ours])
    baseline_mean = mean([x[1] for x in baseline])
    print(f"\nAverage:")
    print(f'{"ours:":<{15}} {ours_mean:<{20}.2f}')
    print(f'{"baseline:":<{15}} {baseline_mean:<{20}.2f}')
    print(f'{"diff:":<{15}} {ours_mean-baseline_mean:<{20}.2f}')
    statistics["ours_mean"] = ours_mean
    statistics["baseline_mean"] = baseline_mean
    statistics["diff_mean"] = ours_mean-baseline_mean
    return statistics

def write_to_csv(statistics, file_name="lpips"):
    import csv
    with open(os.path.join(output_path, f'{file_name}.csv'), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Label", "Ours", "Baseline", "Diff"])
        for line in zip(statistics["labels"], statistics["ours"], statistics["baseline"], statistics["diff"]):
            label = line[0]
            ours_value = line[1]
            baseline_value = line[2]
            diff_value = line[3]
            writer.writerow([label, ours_value, baseline_value, diff_value])
        writer.writerow(["Average", statistics["ours_mean"], statistics["baseline_mean"], statistics["diff_mean"]])

if __name__ == "__main__":
    # hardcode the filenames for now
    output_path = "assets/eval_results"
    images_path = "assets/eval_images"
    file_names = [
        "0_no_smile.png",
        "13_smiling.png",
        "17_no_bangs.png",
        "stefano_bangs.png",
        "stefano_glasses.png",
        "stefano_shaven.png",
        "stefano_young.png"
        ]
    ours = [os.path.join(images_path, "ours", f) for f in file_names]
    p2p = [os.path.join(images_path, "pix2pix-baseline", f) for f in file_names]
    # rec files in png
    rec_file_names = [
        "0.png",
        "13.png",
        "17.png",
        "stefano.png",
        "stefano.png",
        "stefano.png",
        "stefano.png"
    ]
    # original files in jpg
    original_file_names = [
        "0.jpg",
        "13.jpg",
        "17.jpg",
        "stefano.jpg",
        "stefano.jpg",
        "stefano.jpg",
        "stefano.jpg"
    ]
    rec = [os.path.join(images_path, "rec", f) for f in rec_file_names]
    original = [os.path.join(images_path, "original", f) for f in original_file_names]
    masks_names = [
        ["00000_mouth.png"],
        ["00013_l_lip.png", "00013_u_lip.png"],
        ["00017_hair.png"],
        ["stefano_bangs_mask.jpg"],
        ["stefano_glasses_mask.jpg"],
        ["stefano_beard_mask.jpg"],
        ["stefano_age_mask.jpg"],
    ]
    masks_list = [[os.path.join(images_path, "masks", file) for file in sublist] for sublist in masks_names]
    combined_masks = resize_and_combine_masks(masks_list)
    src_descriptions = [
        "mouth open wide smiling",
        "not smiling",
        "white woman with blonde hair bangs",
        "white man with black hair without bangs",
        "without glasses",
        "beard",
        "middle aged white man"
    ]

    dst_descriptions = [
        "mouth closed not smiling",
        "smiling",
        "white woman with blonde hair without bangs",
        "white man with black hair bangs",
        "wearing glasses",
        "clean shaven face",
        "young adult white man"
    ]

    prompts = [
        "a woman wearing a tiara and smiling at the camera",
        "a woman with long blonde hair and blue eyes",
        "a woman with long blonde hair and blue eyes",
        "a close up of a person wearing a shirt and tie",
        "a close up of a person wearing a shirt and tie",
        "a close up of a person wearing a shirt and tie",
        "a close up of a person wearing a shirt and tie"
    ]

    dst_sentences = [x[0]+", "+x[1] for x in zip(prompts, dst_descriptions)]

    # # Calculate MSE
    # print("----------------------------------------")
    # print("Masked MSE")
    # print("Ours vs Rec, P2P vs Rec")
    # ours_vs_res = get_evals(mse,ours,rec,combined_masks)
    # p2p_vs_res = get_evals(mse,p2p,rec,combined_masks)
    # ours_vs_p2p_rec_masked = print_result(ours_vs_res, p2p_vs_res)

    # print()
    # print("Ours vs Original, P2P vs Original")
    # ours_vs_orig = get_evals(mse,ours,original,combined_masks)
    # p2p_vs_orig = get_evals(mse,p2p,original,combined_masks)
    # ours_vs_p2p_orig_masked = print_result(ours_vs_orig, p2p_vs_orig)
    # write_to_csv(ours_vs_p2p_orig_masked, "mse_masked")

    # print("----------------------------------------")
    # print("Unmasked MSE")
    # print("Ours vs Rec, P2P vs Rec")
    # ours_vs_res = get_evals(mse,ours,rec)
    # p2p_vs_res = get_evals(mse,p2p,rec)
    # ours_vs_p2p_rec_unmasked = print_result(ours_vs_res, p2p_vs_res)

    # print()
    # print("Ours vs Original, P2P vs Original")
    # ours_vs_orig = get_evals(mse,ours,original)
    # p2p_vs_orig = get_evals(mse,p2p,original)
    # ours_vs_p2p_orig_unmasked = print_result(ours_vs_orig, p2p_vs_orig)
    # write_to_csv(ours_vs_p2p_orig_unmasked, "mse_unmasked")


    
    # # Calculate LPIPS
    # print("----------------------------------------")
    # print("Masked LPIPS")
    # print("Ours vs Rec, P2P vs Rec")
    # ours_vs_res = get_evals(LPIPS,ours,rec,combined_masks)
    # p2p_vs_res = get_evals(LPIPS,p2p,rec,combined_masks)
    # ours_vs_p2p_rec_masked = print_result(ours_vs_res, p2p_vs_res)

    # print()
    # print("Ours vs Original, P2P vs Original")
    # ours_vs_orig = get_evals(LPIPS,ours,original,combined_masks)
    # p2p_vs_orig = get_evals(LPIPS,p2p,original,combined_masks)
    # ours_vs_p2p_orig_masked = print_result(ours_vs_orig, p2p_vs_orig)
    # write_to_csv(ours_vs_p2p_orig_masked, "lpips_masked")

    # print("----------------------------------------")
    # print("Unmasked LPIPS")
    # print("Ours vs Rec, P2P vs Rec")
    # ours_vs_res = get_evals(LPIPS,ours,rec)
    # p2p_vs_res = get_evals(LPIPS,p2p,rec)
    # ours_vs_p2p_rec_unmasked = print_result(ours_vs_res, p2p_vs_res)
    
    # print()
    # print("Ours vs Original, P2P vs Original")
    # ours_vs_orig = get_evals(LPIPS,ours,original)
    # p2p_vs_orig = get_evals(LPIPS,p2p,original)
    # ours_vs_p2p_orig_unmasked = print_result(ours_vs_orig, p2p_vs_orig)
    # write_to_csv(ours_vs_p2p_orig_unmasked, "lpips_unmasked")

    
    # Calculate CLIP
    print("----------------------------------------")
    print("CLIP")
    print("Ours vs P2P vs Original")
    ours_clip = get_evals(CLIP,ours,dst_sentences)
    p2p_clip = get_evals(CLIP,p2p,dst_sentences)
    original_clip = get_evals(CLIP,original,dst_sentences)
    # print("ours", ours_clip)
    # print("p2p", p2p_clip)
    # print("original", original_clip)
    ours_vs_p2p_clip = print_result(ours_clip, p2p_clip)
    write_to_csv(ours_vs_p2p_clip, "clip")
    # p2p = get_evals(CLIP,p2p,rec,combined_masks)
    # original = get_evals(CLIP,p2p,rec,combined_masks)



    


    

