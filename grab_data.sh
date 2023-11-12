#!/bin/bash

image=$1 # "29954.jpg"
mask_prefix=$2 #"29954"

copy_img="cp CelebAMask-HQ/CelebA-HQ-img/${image} assets/custom/"
echo $copy_img
$copy_img

find CelebAMask-HQ/CelebAMask-HQ-mask-anno -type f -regex ".*${mask_prefix}.*" | xargs -I {} cp {} assets/masks/