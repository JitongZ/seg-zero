#!/bin/bash

python src/inversion.py  \
        --input_image "assets/custom/180.jpg" \
        --results_folder "output/test_custom" \
        --num_ddim_steps 50 \
        --use_float_16

