import cv2
import os
import imgaug.augmenters as iaa
import numpy as np

# --- Configuration ---
TARGET_IMAGE_FOLDER = "data/train"
AUGMENTED_IMAGE_FOLDER = "data/train_aug"
NUM_AUGMENTED_IMAGES_PER_ORIGINAL = 3 # Adjust this number as needed

# --- Augmentation Pipeline ---
# You can enable/disable and adjust parameters for each augmenter
augmentation_pipeline = iaa.Sequential([
    # Geometric Transformations
    iaa.Fliplr(0.5),  # Flip horizontally 50% of the time
    iaa.Affine(
        scale={"x": (0.9, 1.1), "y": (0.9, 1.1)}, # Scale images by 90-110%
        translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, # Translate by -10% to +10%
        rotate=(-15, 15), # Rotate between -15 and 15 degrees
        shear=(-8, 8), # Shear between -8 and 8 degrees
        order=1, # Use bilinear interpolation (fast and good for most cases)
        cval=0, # Fill empty areas with 0 (black)
        mode="constant"
    ),

    # Photometric Transformations
    iaa.Multiply((0.8, 1.2)), # Change brightness by 80-120%
    iaa.ContrastNormalization((0.75, 1.5)), # Adjust contrast
    iaa.AddToHueAndSaturation((-20, 20)), # Change hue and saturation
    iaa.Sometimes(0.3, iaa.GaussianBlur(sigma=(0, 0.5))), # Apply Gaussian blur with a probability of 30%

    # Other Augmentations
    iaa.Sometimes(0.2, iaa.Crop(percent=(0, 0.1))), # Randomly crop up to 10% of the image
    iaa.Sometimes(0.2, iaa.pillike.FilterEmboss()), # Emboss the image
    iaa.Sometimes(0.2, iaa.AdditiveGaussianNoise(scale=(0, 0.05*255))), # Add gaussian noise
    #iaa.Sometimes(0.1, iaa.pillike.FilterSharpen(factor=1.2)), # Sharpen the image (using a single float value)
    iaa.Sometimes(0.1, iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.75, 1.5))), # Alternative sharpening augmenter

    # Occlusion Augmentation (Random Erasing)
    iaa.Sometimes(0.2, iaa.Cutout(nb_iterations=(1, 3), size=0.2, squared=False)) # Add random black rectangles
], random_order=True) # Apply augmenters in random order

# --- Data Augmentation Process ---
if not os.path.exists(AUGMENTED_IMAGE_FOLDER):
    os.makedirs(AUGMENTED_IMAGE_FOLDER)

print(f"[INFO] Starting data augmentation from {TARGET_IMAGE_FOLDER}...")

for image_name in os.listdir(TARGET_IMAGE_FOLDER):
    image_path = os.path.join(TARGET_IMAGE_FOLDER, image_name)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not read image {image_path}")
        continue

    print(f"  -> Augmenting {image_name}...")
    for i in range(NUM_AUGMENTED_IMAGES_PER_ORIGINAL):
        image_aug = augmentation_pipeline.augment_image(image)
        augmented_image_name = f"{os.path.splitext(image_name)[0]}_aug_{i}.jpg"
        augmented_image_path = os.path.join(AUGMENTED_IMAGE_FOLDER, augmented_image_name)
        cv2.imwrite(augmented_image_path, image_aug)

print("[INFO] Data augmentation complete.")

# --- Important Next Step ---
print(f"[INFO] Now, update the TARGET_IMAGE_FOLDER in your main face blurring script to include both '{TARGET_IMAGE_FOLDER}' and '{AUGMENTED_IMAGE_FOLDER}' to use all images for training embeddings.")