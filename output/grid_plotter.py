import os
import re
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Specify the directory path
directory_path = './stefano_glasses_guidance'

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

plt.tight_layout(pad=3.0, h_pad=0.5, w_pad=0.5)

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

fig.suptitle('No Glasses to Glasses', fontsize=34, y=0.98)  # Adjust y for title position
fig.supxlabel('lambda_in (low to high)', fontsize=20, x=0.5)  # Adjust x for xlabel position
fig.supylabel('lambda_out (low to high)', fontsize=20, y=0.5)  # Adjust y for ylabel position

# Fine-tune subplot adjustments if needed
plt.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.95)

# fig.suptitle('No Glasses to Glasses', fontsize=34)
# fig.supxlabel('lambda_in (low to high)', fontsize=20)
# fig.supylabel('lambda_out (low to high)', fontsize=20)

# # Adjust layout
# plt.subplots_adjust(wspace=0, hspace=0)

# Show the plot
plt.savefig('glasses_guidance_grid_plot.png')
plt.show()
