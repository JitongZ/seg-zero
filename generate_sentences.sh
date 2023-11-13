#!/bin/bash

python src/generate_sentences.py \
    --source "beard" \
    --target "clean shaven" \
    --num_sentences 60 \
    --backend "gpt4"
