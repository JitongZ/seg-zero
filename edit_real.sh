#!/bin/bash

# Experiment for removing facial hair for 180
python src/edit_real.py \
    --inversion "output/test_custom/inversion/180.pt" \
    --prompt "output/test_custom/prompt/180.txt" \
    --task_name "beard2clean shaven" \
    --results_folder "output/test_custom/" \
    --num_ddim_steps 50 \
    --use_float_16 \
    --mask "assets/masks/00180_skin.png" \
    --mask_outside_scaling_factor 1.5 \
    --mask_inside_scaling_factor 0.55

# Experiment for removing facial hair for 180 using original pix2pix-zero
# python src/edit_real.py \
#     --inversion "output/test_custom/inversion/180.pt" \
#     --prompt "output/test_custom/prompt/180.txt" \
#     --task_name "beard2clean shaven" \
#     --results_folder "output/test_custom/" \
#     --num_ddim_steps 50 \
#     --use_float_16