from PIL import Image
import numpy as np
import os

# def mse(imageA, imageB):
#     # Ensure the images have the same dimensions
#     assert imageA.shape == imageB.shape, "Images must have the same dimensions"

#     # Compute the mean squared error between the two images
#     err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
#     err /= float(imageA.shape[0] * imageA.shape[1])
    
#     return err

def mse_masked(image1_path, image2_path, mask):
    image1 = np.array(Image.open(image1_path))
    image2 = np.array(Image.open(image2_path))
    # Ensure the mask is boolean for indexing
    mask = mask.astype(bool)
    
    # Invert the mask to focus on the unmasked area
    unmasked_area = ~mask

    # Calculate the squared difference in the unmasked area
    squared_diff = np.square(image1[unmasked_area] - image2[unmasked_area])
    # print("mask sum: ", np.sum(unmasked_area))
    # print("squared_diff shape", squared_diff.shape)

    # Calculate the mean of the squared differences
    mse_value = np.mean(squared_diff)
    return mse_value

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

def print_mse(path_list1, path_list2, mask_list):
    for image1, image2, mask in zip(path_list1, path_list2, mask_list):
        mse_val = mse_masked(image1, image2, mask)
        file_name = image1.split('.')[-2].split('/')[-1]
        print(f"MSE of {file_name}: {round(mse_val, 2)}")


if __name__ == "__main__":
    # hardcode the filenames for now
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
    rec_file_names = [
        "0.png",
        "13.png",
        "17.png",
        "stefano.png",
        "stefano.png",
        "stefano.png",
        "stefano.png"
    ]
    rec = [os.path.join(images_path, "rec", f) for f in rec_file_names]
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

    # Calculate MSE
    print("Ours vs Rec")
    print_mse(ours,rec,combined_masks)
    print("P2P vs Rec")
    print_mse(p2p,rec,combined_masks)
    

