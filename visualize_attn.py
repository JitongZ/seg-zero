import pickle
import ipdb
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def get_attns(ref_attns):
    timesteps = list(ref_attns.keys())
    attn_blocks = ref_attns[timesteps[0]].keys()
    attns = {}
    for attn_block in attn_blocks:
        attns[attn_block] = []
        print(attn_block)
        for timestep in timesteps:
            # Get highest attention over all the tokens
            cross_attn, cross_attn_indices = ref_attns[timestep][attn_block].max(dim=-1)
            channels, pixels = cross_attn.shape
            dim = int(np.sqrt(pixels))
            cross_attn = cross_attn.reshape(channels, dim, dim)
            cross_attn = cross_attn.mean(dim=0)
            attns[attn_block].append(cross_attn)
    return attns

def plot_attns(attns):
    attn_blocks = attns.keys()
    for attn_block in attn_blocks:
        for i in range(len(attns[attn_block])):
            img = attns[attn_block][i]
            plt.imshow(img)
            dim, _ = img.shape
            plt.axis('off')
            dir_path = Path(f"""viz/{attn_block.replace(".", "_")}""")
            dir_path.mkdir(parents=True, exist_ok=True)
            plt.title(f"{attn_block} step {i} ({dim} x {dim})")
            plt.savefig(f"{dir_path}/{i}.png")

with open('attention.pkl', 'rb') as f:
    ref_attns = pickle.load(f)
    attns = get_attns(ref_attns)
    plot_attns(attns)