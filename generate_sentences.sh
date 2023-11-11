#!/bin/bash

python src/generate_sentences.py \
    --source "natural lips" \
    --target "lipstick" \
    --num_sentences 60 \
    --backend "gpt4"
