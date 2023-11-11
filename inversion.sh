#!/bin/bash

# For no lipstick to lipstick
python src/inversion.py  \
        --input_image "assets/custom/4.jpg" \
        --results_folder "output/test_custom" \
        --num_ddim_steps 50 \
        --use_float_16


# For facial hair to clean shaven
# python src/inversion.py  \
#         --input_image "assets/custom/180.jpg" \
#         --results_folder "output/test_custom" \
#         --num_ddim_steps 50 \
#         --use_float_16

