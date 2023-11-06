import os
import glob
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image

directories = [
    "viz/down_blocks_0_attentions_0_transformer_blocks_0_attn2",
    "viz/down_blocks_0_attentions_1_transformer_blocks_0_attn2",
    "viz/down_blocks_1_attentions_0_transformer_blocks_0_attn2",
    "viz/down_blocks_1_attentions_1_transformer_blocks_0_attn2",
    "viz/down_blocks_2_attentions_0_transformer_blocks_0_attn2",
    "viz/down_blocks_2_attentions_1_transformer_blocks_0_attn2",
    "viz/mid_block_attentions_0_transformer_blocks_0_attn2",
    "viz/up_blocks_1_attentions_0_transformer_blocks_0_attn2",
    "viz/up_blocks_1_attentions_1_transformer_blocks_0_attn2",
    "viz/up_blocks_1_attentions_2_transformer_blocks_0_attn2",
    "viz/up_blocks_2_attentions_0_transformer_blocks_0_attn2",
    "viz/up_blocks_2_attentions_1_transformer_blocks_0_attn2",
    "viz/up_blocks_2_attentions_2_transformer_blocks_0_attn2",
    "viz/up_blocks_3_attentions_0_transformer_blocks_0_attn2",
    "viz/up_blocks_3_attentions_1_transformer_blocks_0_attn2",
    "viz/up_blocks_3_attentions_2_transformer_blocks_0_attn2",
]

def create_gif(image_directory):
    # Set the directory containing your images
    image_pattern = os.path.join(image_directory, '*.png')  # Assumes images are .png

    # Retrieve all image paths
    image_paths = sorted(glob.glob(image_pattern), key=lambda x: int(os.path.split(x)[-1].split('.')[0]))

    # Ensure the images are sorted correctly by converting the filenames to integers
    image_paths = sorted(image_paths, key=lambda x: int(os.path.basename(x).split('.')[0]))

    # Create a figure for plotting
    fig = plt.figure()

    # Create a list to store the frames
    frames = []

    # Load and append each image to the frames list
    for image_path in image_paths:
        frame = Image.open(image_path)
        frames.append([plt.imshow(frame, animated=True, cmap='gray')])  # Remove cmap if images are RGB

    # Hide axes
    plt.axis('off')

    # Create and save the animation
    ani = animation.ArtistAnimation(fig, frames, interval=100, blit=True)  # interval controls frame timing
    # ani.save(f'{image_directory}/output.gif', writer='pillow', fps=14)
    ani.save(f'viz/output_{image_directory.replace("viz/", "")}.gif', writer='pillow', fps=14)

for directory in directories:
    create_gif(directory)