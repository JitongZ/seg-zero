#!/bin/bash

image = "29954.jpg"
mask_prefix = "29954"

cp "CelebAMask-HQ/CelebA-HQ-img/${image}" assets/custom/
cp "CelebAMask-HQ/CelebAMask-HQ-mask-anno/${image}*" assets/masks
