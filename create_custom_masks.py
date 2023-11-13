import os
from PIL import Image

directory_path = 'assets/custom_masks_raw'
tol = 10

def process_image(file_path):
    # Open the image file
    with Image.open(f"{directory_path}/{file_path}") as img:
        # Convert the image to RGB if it's not
        img = img.convert('RGB')

        # Get the size of the image
        width, height = img.size

        # Process each pixel
        for x in range(width):
            for y in range(height):
                r, g, b = img.getpixel((x, y))
                # Check if the pixel is green (#0, 117, 0)
                if r < tol and b < tol and g > 117 - tol and g < 117 + tol:
                    new_color = (255, 255, 255)  # White
                else:
                    new_color = (0, 0, 0)  # Black
                img.putpixel((x, y), new_color)

        # Save the processed image
        img.save(f'assets/custom_masks/{file_path}')

for filename in os.listdir(directory_path):
    process_image(filename)