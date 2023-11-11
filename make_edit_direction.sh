#!/bin/bash

python src/make_edit_direction.py \
    --file_source_sentences "assets/sentences/natural lips.txt" \
    --file_target_sentences "assets/sentences/lipstick.txt" \
    --output_folder assets/embeddings_sd_1.4

# python src/make_edit_direction.py \
#     --file_source_sentences "assets/sentences/beard.txt" \
#     --file_target_sentences "assets/sentences/clean shaven.txt" \
#     --output_folder assets/embeddings_sd_1.4