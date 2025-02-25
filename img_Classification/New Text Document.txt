import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import os

# Parameters for dataset generation
num_samples_per_digit = 500  # Number of samples per digit (0-9)
image_size = (28, 28)  # Image size (same as MNIST)
font_size = 24  # Font size for drawing digits

# Create a directory to save generated images
dataset_dir = "/mnt/data/generated_mnist"
os.makedirs(dataset_dir, exist_ok=True)

# Generate images for each digit (0-9)
for digit in range(10):
    digit_dir = os.path.join(dataset_dir, str(digit))
    os.makedirs(digit_dir, exist_ok=True)
    
    for i in range(num_samples_per_digit):
        img = Image.new("L", image_size, color=255)  # Create a white image
        draw = ImageDraw.Draw(img)
        
        # Draw digit
        draw.text((5, 2), str(digit), fill=0)  # Simple text drawing
        
        # Save image
        img_path = os.path.join(digit_dir, f"{digit}_{i}.png")
        img.save(img_path)

# Check a few generated images
sample_images = []
for digit in range(10):
    img_path = os.path.join(dataset_dir, str(digit), f"{digit}_0.png")
    img = Image.open(img_path)
    sample_images.append(img)

# Display sample images
fig, axes = plt.subplots(1, 10, figsize=(10, 2))
for ax, img, digit in zip(axes, sample_images, range(10)):
    ax.imshow(img, cmap="gray")
    ax.set_title(str(digit))
    ax.axis("off")

plt.show()
