import os
import re
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Specify the directory path
directory_path = './stefano_glasses'

# List all files in the directory
all_files = os.listdir(directory_path)

# Filter for PNG files
png_files = [file for file in all_files if file.endswith('.png')]

pattern = r'outside_([0-9.]+)_inside_([0-9.]+)\.png'

images = {}

for png_file in png_files:
    match = re.search(pattern, png_file)
    if match:
        outside = float(match.group(1))
        inside = float(match.group(2))

        file_path = os.path.join(directory_path, png_file)
        image = Image.open(file_path)
        image_array = np.array(image)

        images[(outside, inside)] = image_array
        # print(f"File: {png_file} - Outside: {outside}, Inside: {inside}")
    else:
        raise ValueError(f"Invalid file name: {png_file}")

# Create a figure with 16x16 grid of subplots
fig, axes = plt.subplots(16, 16, figsize=(16, 16))

# Flatten the axes array for easy iteration
axes = axes.flatten()

ordering = np.array(sorted(images.keys()))
ordering = ordering.reshape(16, 16, 2)
ordering = ordering[::-1].reshape(-1, 2)

# Loop through images and plot each one
for i, key in enumerate(ordering):
    outside, inside = key
    axes[i].imshow(images[(outside, inside)])
    axes[i].axis('off') 

    print(key)

    # axes[i].set_title(f'Outside: {outside}, Inside: {inside}', fontsize=8)
    # Optionally, you can turn off the axis ticks if they're not needed
    # axes[i].set_xticks([inside])
    # axes[i].set_yticks([outside])

fig.suptitle('No Glasses to Glasses', fontsize=16)
fig.supxlabel('lambda_in (low to high)', fontsize=12)
fig.supylabel('lambda_out (low to high)', fontsize=12)

# Adjust layout
plt.subplots_adjust(wspace=0, hspace=0)

# Show the plot
plt.savefig('glasses_grid_plot.png')
plt.show()