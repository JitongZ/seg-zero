#!/bin/bash

python src/generate_sentences.py \
    --source "a young adult" \
    --target "a man with short hair" \
    --num_sentences 60 \
    --backend "gpt4"
