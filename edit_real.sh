#!/bin/bash

# Experiment for adding lipstick for 4
python src/edit_real.py \
    --inversion "output/test_custom/inversion/4.pt" \
    --prompt "output/test_custom/prompt/4.txt" \
    --task_name "natural lips2lipstick" \
    --results_folder "output/test_custom/" \
    --num_ddim_steps 50 \
    --use_float_16 \
    --masks "assets/masks/00004_u_lip.png" "assets/masks/00004_l_lip.png" \
    --mask_outside_scaling_factor 1.5 \
    --mask_inside_scaling_factor 0.55

# Adding lipstick baseline
# python src/edit_real.py \
#     --inversion "output/test_custom/inversion/4.pt" \
#     --prompt "output/test_custom/prompt/4.txt" \
#     --task_name "natural lips2lipstick" \
#     --results_folder "output/test_custom/" \
#     --num_ddim_steps 50 \
#     --use_float_16

# Experiment for removing facial hair for 180
# python src/edit_real.py \
#     --inversion "output/test_custom/inversion/180.pt" \
#     --prompt "output/test_custom/prompt/180.txt" \
#     --task_name "beard2clean shaven" \
#     --results_folder "output/test_custom/" \
#     --num_ddim_steps 50 \
#     --use_float_16 \
#     --masks "assets/masks/00180_skin.png" \
#     --mask_outside_scaling_factor 1.5 \
#     --mask_inside_scaling_factor 0.55

# Experiment for removing facial hair for 180 using original pix2pix-zero
# python src/edit_real.py \
#     --inversion "output/test_custom/inversion/180.pt" \
#     --prompt "output/test_custom/prompt/180.txt" \
#     --task_name "beard2clean shaven" \
#     --results_folder "output/test_custom/" \
#     --num_ddim_steps 50 \
#     --use_float_16