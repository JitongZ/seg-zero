#!/bin/bash

python src/edit_real.py \
    --inversion "output/test_custom/inversion/180.pt" \
    --prompt "output/test_custom/prompt/180.txt" \
    --task_name "beard2clean shaven" \
    --results_folder "output/test_custom/" \
    --num_ddim_steps 50 \
    --use_float_16