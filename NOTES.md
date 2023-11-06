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


### UNet Structure
```
UNet2DConditionModel(
  (conv_in): Conv2d(4, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (time_proj): Timesteps()
  (time_embedding): TimestepEmbedding(
    (linear_1): Linear(in_features=320, out_features=1280, bias=True)
    (act): SiLU()
    (linear_2): Linear(in_features=1280, out_features=1280, bias=True)
  )
  (down_blocks): ModuleList(
    (0): CrossAttnDownBlock2D(
      (attentions): ModuleList(
        (0-1): 2 x Transformer2DModel(
          (norm): GroupNorm(32, 320, eps=1e-06, affine=True)
          (proj_in): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1))
          (transformer_blocks): ModuleList(
            (0): BasicTransformerBlock(
              (attn1): CrossAttention(
                (to_q): Linear(in_features=320, out_features=320, bias=False)
                (to_k): Linear(in_features=320, out_features=320, bias=False)
                (to_v): Linear(in_features=320, out_features=320, bias=False)
                (to_out): ModuleList(
                  (0): Linear(in_features=320, out_features=320, bias=True)
                  (1): Dropout(p=0.0, inplace=False)
                )
              )
              (ff): FeedForward(
                (net): ModuleList(
                  (0): GEGLU(
                    (proj): Linear(in_features=320, out_features=2560, bias=True)
                  )
                  (1): Dropout(p=0.0, inplace=False)
                  (2): Linear(in_features=1280, out_features=320, bias=True)
                )
              )
              (attn2): CrossAttention(
                (to_q): Linear(in_features=320, out_features=320, bias=False)
                (to_k): Linear(in_features=768, out_features=320, bias=False)
                (to_v): Linear(in_features=768, out_features=320, bias=False)
                (to_out): ModuleList(
                  (0): Linear(in_features=320, out_features=320, bias=True)
                  (1): Dropout(p=0.0, inplace=False)
                )
              )
              (norm1): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
              (norm2): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
              (norm3): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
            )
          )
          (proj_out): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1))
        )
      )
      (resnets): ModuleList(
        (0-1): 2 x ResnetBlock2D(
          (norm1): GroupNorm(32, 320, eps=1e-05, affine=True)
          (conv1): Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (time_emb_proj): Linear(in_features=1280, out_features=320, bias=True)
          (norm2): GroupNorm(32, 320, eps=1e-05, affine=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (conv2): Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (nonlinearity): SiLU()
        )
      )
      (downsamplers): ModuleList(
        (0): Downsample2D(
          (conv): Conv2d(320, 320, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        )
      )
    )
    (1): CrossAttnDownBlock2D(
      (attentions): ModuleList(
        (0-1): 2 x Transformer2DModel(
          (norm): GroupNorm(32, 640, eps=1e-06, affine=True)
          (proj_in): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1))
          (transformer_blocks): ModuleList(
            (0): BasicTransformerBlock(
              (attn1): CrossAttention(
                (to_q): Linear(in_features=640, out_features=640, bias=False)
                (to_k): Linear(in_features=640, out_features=640, bias=False)
                (to_v): Linear(in_features=640, out_features=640, bias=False)
                (to_out): ModuleList(
                  (0): Linear(in_features=640, out_features=640, bias=True)
                  (1): Dropout(p=0.0, inplace=False)
                )
              )
              (ff): FeedForward(
                (net): ModuleList(
                  (0): GEGLU(
                    (proj): Linear(in_features=640, out_features=5120, bias=True)
                  )
                  (1): Dropout(p=0.0, inplace=False)
                  (2): Linear(in_features=2560, out_features=640, bias=True)
                )
              )
              (attn2): CrossAttention(
                (to_q): Linear(in_features=640, out_features=640, bias=False)
                (to_k): Linear(in_features=768, out_features=640, bias=False)
                (to_v): Linear(in_features=768, out_features=640, bias=False)
                (to_out): ModuleList(
                  (0): Linear(in_features=640, out_features=640, bias=True)
                  (1): Dropout(p=0.0, inplace=False)
                )
              )
              (norm1): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
              (norm2): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
              (norm3): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
            )
          )
          (proj_out): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1))
        )
      )
      (resnets): ModuleList(
        (0): ResnetBlock2D(
          (norm1): GroupNorm(32, 320, eps=1e-05, affine=True)
          (conv1): Conv2d(320, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (time_emb_proj): Linear(in_features=1280, out_features=640, bias=True)
          (norm2): GroupNorm(32, 640, eps=1e-05, affine=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (conv2): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (nonlinearity): SiLU()
          (conv_shortcut): Conv2d(320, 640, kernel_size=(1, 1), stride=(1, 1))
        )
        (1): ResnetBlock2D(
          (norm1): GroupNorm(32, 640, eps=1e-05, affine=True)
          (conv1): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (time_emb_proj): Linear(in_features=1280, out_features=640, bias=True)
          (norm2): GroupNorm(32, 640, eps=1e-05, affine=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (conv2): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (nonlinearity): SiLU()
        )
      )
      (downsamplers): ModuleList(
        (0): Downsample2D(
          (conv): Conv2d(640, 640, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        )
      )
    )
    (2): CrossAttnDownBlock2D(
      (attentions): ModuleList(
        (0-1): 2 x Transformer2DModel(
          (norm): GroupNorm(32, 1280, eps=1e-06, affine=True)
          (proj_in): Conv2d(1280, 1280, kernel_size=(1, 1), stride=(1, 1))
          (transformer_blocks): ModuleList(
            (0): BasicTransformerBlock(
              (attn1): CrossAttention(
                (to_q): Linear(in_features=1280, out_features=1280, bias=False)
                (to_k): Linear(in_features=1280, out_features=1280, bias=False)
                (to_v): Linear(in_features=1280, out_features=1280, bias=False)
                (to_out): ModuleList(
                  (0): Linear(in_features=1280, out_features=1280, bias=True)
                  (1): Dropout(p=0.0, inplace=False)
                )
              )
              (ff): FeedForward(
                (net): ModuleList(
                  (0): GEGLU(
                    (proj): Linear(in_features=1280, out_features=10240, bias=True)
                  )
                  (1): Dropout(p=0.0, inplace=False)
                  (2): Linear(in_features=5120, out_features=1280, bias=True)
                )
              )
              (attn2): CrossAttention(
                (to_q): Linear(in_features=1280, out_features=1280, bias=False)
                (to_k): Linear(in_features=768, out_features=1280, bias=False)
                (to_v): Linear(in_features=768, out_features=1280, bias=False)
                (to_out): ModuleList(
                  (0): Linear(in_features=1280, out_features=1280, bias=True)
                  (1): Dropout(p=0.0, inplace=False)
                )
              )
              (norm1): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
              (norm2): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
              (norm3): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
            )
          )
          (proj_out): Conv2d(1280, 1280, kernel_size=(1, 1), stride=(1, 1))
        )
      )
      (resnets): ModuleList(
        (0): ResnetBlock2D(
          (norm1): GroupNorm(32, 640, eps=1e-05, affine=True)
          (conv1): Conv2d(640, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (time_emb_proj): Linear(in_features=1280, out_features=1280, bias=True)
          (norm2): GroupNorm(32, 1280, eps=1e-05, affine=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (conv2): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (nonlinearity): SiLU()
          (conv_shortcut): Conv2d(640, 1280, kernel_size=(1, 1), stride=(1, 1))
        )
        (1): ResnetBlock2D(
          (norm1): GroupNorm(32, 1280, eps=1e-05, affine=True)
          (conv1): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (time_emb_proj): Linear(in_features=1280, out_features=1280, bias=True)
          (norm2): GroupNorm(32, 1280, eps=1e-05, affine=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (conv2): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (nonlinearity): SiLU()
        )
      )
      (downsamplers): ModuleList(
        (0): Downsample2D(
          (conv): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        )
      )
    )
    (3): DownBlock2D(
      (resnets): ModuleList(
        (0-1): 2 x ResnetBlock2D(
          (norm1): GroupNorm(32, 1280, eps=1e-05, affine=True)
          (conv1): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (time_emb_proj): Linear(in_features=1280, out_features=1280, bias=True)
          (norm2): GroupNorm(32, 1280, eps=1e-05, affine=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (conv2): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (nonlinearity): SiLU()
        )
      )
    )
  )
  (up_blocks): ModuleList(
    (0): UpBlock2D(
      (resnets): ModuleList(
        (0-2): 3 x ResnetBlock2D(
          (norm1): GroupNorm(32, 2560, eps=1e-05, affine=True)
          (conv1): Conv2d(2560, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (time_emb_proj): Linear(in_features=1280, out_features=1280, bias=True)
          (norm2): GroupNorm(32, 1280, eps=1e-05, affine=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (conv2): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (nonlinearity): SiLU()
          (conv_shortcut): Conv2d(2560, 1280, kernel_size=(1, 1), stride=(1, 1))
        )
      )
      (upsamplers): ModuleList(
        (0): Upsample2D(
          (conv): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
      )
    )
    (1): CrossAttnUpBlock2D(
      (attentions): ModuleList(
        (0-2): 3 x Transformer2DModel(
          (norm): GroupNorm(32, 1280, eps=1e-06, affine=True)
          (proj_in): Conv2d(1280, 1280, kernel_size=(1, 1), stride=(1, 1))
          (transformer_blocks): ModuleList(
            (0): BasicTransformerBlock(
              (attn1): CrossAttention(
                (to_q): Linear(in_features=1280, out_features=1280, bias=False)
                (to_k): Linear(in_features=1280, out_features=1280, bias=False)
                (to_v): Linear(in_features=1280, out_features=1280, bias=False)
                (to_out): ModuleList(
                  (0): Linear(in_features=1280, out_features=1280, bias=True)
                  (1): Dropout(p=0.0, inplace=False)
                )
              )
              (ff): FeedForward(
                (net): ModuleList(
                  (0): GEGLU(
                    (proj): Linear(in_features=1280, out_features=10240, bias=True)
                  )
                  (1): Dropout(p=0.0, inplace=False)
                  (2): Linear(in_features=5120, out_features=1280, bias=True)
                )
              )
              (attn2): CrossAttention(
                (to_q): Linear(in_features=1280, out_features=1280, bias=False)
                (to_k): Linear(in_features=768, out_features=1280, bias=False)
                (to_v): Linear(in_features=768, out_features=1280, bias=False)
                (to_out): ModuleList(
                  (0): Linear(in_features=1280, out_features=1280, bias=True)
                  (1): Dropout(p=0.0, inplace=False)
                )
              )
              (norm1): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
              (norm2): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
              (norm3): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
            )
          )
          (proj_out): Conv2d(1280, 1280, kernel_size=(1, 1), stride=(1, 1))
        )
      )
      (resnets): ModuleList(
        (0-1): 2 x ResnetBlock2D(
          (norm1): GroupNorm(32, 2560, eps=1e-05, affine=True)
          (conv1): Conv2d(2560, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (time_emb_proj): Linear(in_features=1280, out_features=1280, bias=True)
          (norm2): GroupNorm(32, 1280, eps=1e-05, affine=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (conv2): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (nonlinearity): SiLU()
          (conv_shortcut): Conv2d(2560, 1280, kernel_size=(1, 1), stride=(1, 1))
        )
        (2): ResnetBlock2D(
          (norm1): GroupNorm(32, 1920, eps=1e-05, affine=True)
          (conv1): Conv2d(1920, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (time_emb_proj): Linear(in_features=1280, out_features=1280, bias=True)
          (norm2): GroupNorm(32, 1280, eps=1e-05, affine=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (conv2): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (nonlinearity): SiLU()
          (conv_shortcut): Conv2d(1920, 1280, kernel_size=(1, 1), stride=(1, 1))
        )
      )
      (upsamplers): ModuleList(
        (0): Upsample2D(
          (conv): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
      )
    )
    (2): CrossAttnUpBlock2D(
      (attentions): ModuleList(
        (0-2): 3 x Transformer2DModel(
          (norm): GroupNorm(32, 640, eps=1e-06, affine=True)
          (proj_in): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1))
          (transformer_blocks): ModuleList(
            (0): BasicTransformerBlock(
              (attn1): CrossAttention(
                (to_q): Linear(in_features=640, out_features=640, bias=False)
                (to_k): Linear(in_features=640, out_features=640, bias=False)
                (to_v): Linear(in_features=640, out_features=640, bias=False)
                (to_out): ModuleList(
                  (0): Linear(in_features=640, out_features=640, bias=True)
                  (1): Dropout(p=0.0, inplace=False)
                )
              )
              (ff): FeedForward(
                (net): ModuleList(
                  (0): GEGLU(
                    (proj): Linear(in_features=640, out_features=5120, bias=True)
                  )
                  (1): Dropout(p=0.0, inplace=False)
                  (2): Linear(in_features=2560, out_features=640, bias=True)
                )
              )
              (attn2): CrossAttention(
                (to_q): Linear(in_features=640, out_features=640, bias=False)
                (to_k): Linear(in_features=768, out_features=640, bias=False)
                (to_v): Linear(in_features=768, out_features=640, bias=False)
                (to_out): ModuleList(
                  (0): Linear(in_features=640, out_features=640, bias=True)
                  (1): Dropout(p=0.0, inplace=False)
                )
              )
              (norm1): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
              (norm2): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
              (norm3): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
            )
          )
          (proj_out): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1))
        )
      )
      (resnets): ModuleList(
        (0): ResnetBlock2D(
          (norm1): GroupNorm(32, 1920, eps=1e-05, affine=True)
          (conv1): Conv2d(1920, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (time_emb_proj): Linear(in_features=1280, out_features=640, bias=True)
          (norm2): GroupNorm(32, 640, eps=1e-05, affine=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (conv2): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (nonlinearity): SiLU()
          (conv_shortcut): Conv2d(1920, 640, kernel_size=(1, 1), stride=(1, 1))
        )
        (1): ResnetBlock2D(
          (norm1): GroupNorm(32, 1280, eps=1e-05, affine=True)
          (conv1): Conv2d(1280, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (time_emb_proj): Linear(in_features=1280, out_features=640, bias=True)
          (norm2): GroupNorm(32, 640, eps=1e-05, affine=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (conv2): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (nonlinearity): SiLU()
          (conv_shortcut): Conv2d(1280, 640, kernel_size=(1, 1), stride=(1, 1))
        )
        (2): ResnetBlock2D(
          (norm1): GroupNorm(32, 960, eps=1e-05, affine=True)
          (conv1): Conv2d(960, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (time_emb_proj): Linear(in_features=1280, out_features=640, bias=True)
          (norm2): GroupNorm(32, 640, eps=1e-05, affine=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (conv2): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (nonlinearity): SiLU()
          (conv_shortcut): Conv2d(960, 640, kernel_size=(1, 1), stride=(1, 1))
        )
      )
      (upsamplers): ModuleList(
        (0): Upsample2D(
          (conv): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
      )
    )
    (3): CrossAttnUpBlock2D(
      (attentions): ModuleList(
        (0-2): 3 x Transformer2DModel(
          (norm): GroupNorm(32, 320, eps=1e-06, affine=True)
          (proj_in): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1))
          (transformer_blocks): ModuleList(
            (0): BasicTransformerBlock(
              (attn1): CrossAttention(
                (to_q): Linear(in_features=320, out_features=320, bias=False)
                (to_k): Linear(in_features=320, out_features=320, bias=False)
                (to_v): Linear(in_features=320, out_features=320, bias=False)
                (to_out): ModuleList(
                  (0): Linear(in_features=320, out_features=320, bias=True)
                  (1): Dropout(p=0.0, inplace=False)
                )
              )
              (ff): FeedForward(
                (net): ModuleList(
                  (0): GEGLU(
                    (proj): Linear(in_features=320, out_features=2560, bias=True)
                  )
                  (1): Dropout(p=0.0, inplace=False)
                  (2): Linear(in_features=1280, out_features=320, bias=True)
                )
              )
              (attn2): CrossAttention(
                (to_q): Linear(in_features=320, out_features=320, bias=False)
                (to_k): Linear(in_features=768, out_features=320, bias=False)
                (to_v): Linear(in_features=768, out_features=320, bias=False)
                (to_out): ModuleList(
                  (0): Linear(in_features=320, out_features=320, bias=True)
                  (1): Dropout(p=0.0, inplace=False)
                )
              )
              (norm1): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
              (norm2): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
              (norm3): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
            )
          )
          (proj_out): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1))
        )
      )
      (resnets): ModuleList(
        (0): ResnetBlock2D(
          (norm1): GroupNorm(32, 960, eps=1e-05, affine=True)
          (conv1): Conv2d(960, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (time_emb_proj): Linear(in_features=1280, out_features=320, bias=True)
          (norm2): GroupNorm(32, 320, eps=1e-05, affine=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (conv2): Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (nonlinearity): SiLU()
          (conv_shortcut): Conv2d(960, 320, kernel_size=(1, 1), stride=(1, 1))
        )
        (1-2): 2 x ResnetBlock2D(
          (norm1): GroupNorm(32, 640, eps=1e-05, affine=True)
          (conv1): Conv2d(640, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (time_emb_proj): Linear(in_features=1280, out_features=320, bias=True)
          (norm2): GroupNorm(32, 320, eps=1e-05, affine=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (conv2): Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (nonlinearity): SiLU()
          (conv_shortcut): Conv2d(640, 320, kernel_size=(1, 1), stride=(1, 1))
        )
      )
    )
  )
  (mid_block): UNetMidBlock2DCrossAttn(
    (attentions): ModuleList(
      (0): Transformer2DModel(
        (norm): GroupNorm(32, 1280, eps=1e-06, affine=True)
        (proj_in): Conv2d(1280, 1280, kernel_size=(1, 1), stride=(1, 1))
        (transformer_blocks): ModuleList(
          (0): BasicTransformerBlock(
            (attn1): CrossAttention(
              (to_q): Linear(in_features=1280, out_features=1280, bias=False)
              (to_k): Linear(in_features=1280, out_features=1280, bias=False)
              (to_v): Linear(in_features=1280, out_features=1280, bias=False)
              (to_out): ModuleList(
                (0): Linear(in_features=1280, out_features=1280, bias=True)
                (1): Dropout(p=0.0, inplace=False)
              )
            )
            (ff): FeedForward(
              (net): ModuleList(
                (0): GEGLU(
                  (proj): Linear(in_features=1280, out_features=10240, bias=True)
                )
                (1): Dropout(p=0.0, inplace=False)
                (2): Linear(in_features=5120, out_features=1280, bias=True)
              )
            )
            (attn2): CrossAttention(
              (to_q): Linear(in_features=1280, out_features=1280, bias=False)
              (to_k): Linear(in_features=768, out_features=1280, bias=False)
              (to_v): Linear(in_features=768, out_features=1280, bias=False)
              (to_out): ModuleList(
                (0): Linear(in_features=1280, out_features=1280, bias=True)
                (1): Dropout(p=0.0, inplace=False)
              )
            )
            (norm1): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
            (norm2): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
            (norm3): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
          )
        )
        (proj_out): Conv2d(1280, 1280, kernel_size=(1, 1), stride=(1, 1))
      )
    )
    (resnets): ModuleList(
      (0-1): 2 x ResnetBlock2D(
        (norm1): GroupNorm(32, 1280, eps=1e-05, affine=True)
        (conv1): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (time_emb_proj): Linear(in_features=1280, out_features=1280, bias=True)
        (norm2): GroupNorm(32, 1280, eps=1e-05, affine=True)
        (dropout): Dropout(p=0.0, inplace=False)
        (conv2): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (nonlinearity): SiLU()
      )
    )
  )
  (conv_norm_out): GroupNorm(32, 320, eps=1e-05, affine=True)
  (conv_act): SiLU()
  (conv_out): Conv2d(320, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
)
```

### Latent dimensions
The LDM paper had the best performing scale factor of 8, and so they similarly
have `self.vae_scale_factor=8` with latents of `64x64`, to generate final
images of dimensions `512x512`.

### 
Why 77? Because CLIP tokenizer can only handle sequences up to length 77
`self.tokenizer.model_max_length`

### Shape of d_ref_t2a

As stored in `d_ref_t2attn`:
```
down_blocks.0.attentions.0.transformer_blocks.0.attn2 torch.Size([16, 4096, 77])
down_blocks.0.attentions.1.transformer_blocks.0.attn2 torch.Size([16, 4096, 77])
down_blocks.1.attentions.0.transformer_blocks.0.attn2 torch.Size([16, 1024, 77])
down_blocks.1.attentions.1.transformer_blocks.0.attn2 torch.Size([16, 1024, 77])
down_blocks.2.attentions.0.transformer_blocks.0.attn2 torch.Size([16, 256, 77])
down_blocks.2.attentions.1.transformer_blocks.0.attn2 torch.Size([16, 256, 77])
up_blocks.1.attentions.0.transformer_blocks.0.attn2 torch.Size([16, 256, 77])
up_blocks.1.attentions.1.transformer_blocks.0.attn2 torch.Size([16, 256, 77])
up_blocks.1.attentions.2.transformer_blocks.0.attn2 torch.Size([16, 256, 77])
up_blocks.2.attentions.0.transformer_blocks.0.attn2 torch.Size([16, 1024, 77])
up_blocks.2.attentions.1.transformer_blocks.0.attn2 torch.Size([16, 1024, 77])
up_blocks.2.attentions.2.transformer_blocks.0.attn2 torch.Size([16, 1024, 77])
up_blocks.3.attentions.0.transformer_blocks.0.attn2 torch.Size([16, 4096, 77])
up_blocks.3.attentions.1.transformer_blocks.0.attn2 torch.Size([16, 4096, 77])
up_blocks.3.attentions.2.transformer_blocks.0.attn2 torch.Size([16, 4096, 77])
mid_block.attentions.0.transformer_blocks.0.attn2 torch.Size([16, 64, 77])
```