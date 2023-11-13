#!/bin/bash

python src/generate_sentences.py \
    --source "uncovered forehead" \
    --target "hair bangs" \
    --num_sentences 60 \
    --backend "gpt4"
