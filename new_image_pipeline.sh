#!/bin/bash

image_num=13
image_prefix=00013
source="neutral expression"
target="smiling"
mask_outside_scaling_factor=1.0
mask_inside_scaling_factor=0.8
mask_explore_radius=0.6
mask_explore_step=6
xa_guidance=0.01
xa_guidance_baseline=0.01 # default 0.1
guidance_steps=1 #"1 2 3 5 7 10 20 30"
# mask_type choices: 
# cloth  l_brow  l_eye  mouth  nose    r_eye  u_lip 
# hair   l_ear   l_lip  neck   r_brow  skin
mask_types=("skin")

# When performing exploration for images in CelebAHQ-mask, these are the
# attributes you can use:
# ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
# 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry',
# 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses',
# 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male',
# 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face',
# 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns',
# 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat',
# 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']

task="${source}2${target}"
# masks="assets/masks/${image_prefix}_u_lip.png assets/masks/${image_prefix}_l_lip.png" 
masks=""
for mask_type in "${mask_types[@]}"; do
    masks+="assets/masks/${image_prefix}_${mask_type}.png " 
done

./grab_data.sh "${image_num}.jpg" "${image_prefix}"

echo "Task ${task}"
echo "Using masks ${masks}"
echo "Image ${image_num} ${image_prefix}"

echo "Generating sentences"


if [ ! -f "assets/sentences/${source}.txt" ]; then
    touch "assets/sentences/${source}.txt"
    echo -n ${source} > "assets/sentences/${source}.txt"
fi

if [ ! -f "assets/sentences/${target}.txt" ]; then
    touch "assets/sentences/${target}.txt"
    echo -n ${target} > "assets/sentences/${target}.txt"
fi

# this will burn money
# python src/generate_sentences.py \
#     --source "${source}" \
#     --target "${target}" \
#     --num_sentences 60 \
#     --backend "gpt4"

echo "Making edit directions"
python src/make_edit_direction.py \
    --file_source_sentences "assets/sentences/${source}.txt" \
    --file_target_sentences "assets/sentences/${target}.txt" \
    --output_folder assets/embeddings_sd_1.4

echo "Running inversion"

python src/inversion.py  \
        --input_image "assets/custom/${image_num}.jpg" \
        --results_folder "output/test_custom" \
        --num_ddim_steps 50 \
        --use_float_16

echo "Running our edits"

python src/edit_real.py \
    --inversion "output/test_custom/inversion/${image_num}.pt" \
    --prompt "output/test_custom/prompt/${image_num}.txt" \
    --task_name "${task}" \
    --results_folder "output/test_custom/" \
    --num_ddim_steps 50 \
    --masks ${masks} \
    --mask_outside_scaling_factor $mask_outside_scaling_factor \
    --mask_inside_scaling_factor $mask_inside_scaling_factor \
    --guidance_steps=$guidance_steps \
    --xa_guidance=$xa_guidance \
    --mask_explore_radius=$mask_explore_radius \
    --mask_explore_step=$mask_explore_step \
    --use_float_16

echo "Running baseline"

python src/edit_real.py \
    --inversion "output/test_custom/inversion/${image_num}.pt" \
    --prompt "output/test_custom/prompt/${image_num}.txt" \
    --task_name "${task}" \
    --results_folder "output/test_custom/" \
    --num_ddim_steps 50 \
    --xa_guidance=$xa_guidance_baseline \
    --use_float_16