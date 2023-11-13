#!/bin/bash

python src/generate_sentences.py \
    --source "man without glasses" \
    --target "man wearing glasses" \
    --num_sentences 60 \
    --backend "gpt4"
