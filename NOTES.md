# NOTES

## 10/28
author: jitongz & zhiyuxie
### environment
```
conda create -n pix2pix-zero python=3.8
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirement.txt
```
Note:  pip install diffusers==0.12.0

### run tasks
1. Need to generate 1000 sentences for custom edit directions (some pre-defined edit directions are included in `asset` directory)
    * https://github.com/pix2pixzero/pix2pix-zero/issues/3
    * Use models like chatgpt: "Generate many sentences that describe a picture containing a {word}."
    * Can use multiple words
2. Run `src/make_edit_direction.py` to get `embeddings_sd_1.4/[word].pt` file (embeddings)
    * Calculate the average sentence embedding
    * e.g. apple2orange -> apple.pt & orange.pt 
3. Run `src/inversion.py` to generate `prompt/cat_1.txt` and DDIM inversion (noises) `inversion/cat_1.pt`
    * On RTX3080, the runtime is approximately 5 min.
4. Run `src/edit_real.py` to generate `edit/cat_1.png` and reconstruction image `reconstruction/cat_1.png` (same as the input image)
    ```
    python src/edit_real.py \
    --inversion "output/test_cat/inversion/cat_1.pt" \
    --prompt "output/test_cat/prompt/cat_1.txt" \
    --task_name "cat2dog" \
    --results_folder "output/test_cat/" 
    ```
    * On RTX3080, the runtime is approximately 1.5h.