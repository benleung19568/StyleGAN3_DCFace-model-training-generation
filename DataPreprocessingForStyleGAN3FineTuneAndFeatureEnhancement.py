import os
from PIL import Image, ImageEnhance
import numpy as np
import cv2

# Paths
SOURCE_PATH = r"C:/Users/user/stylegan3/augmented_disgust"
TARGET_PATH = r"C:/Users/user/dcface/dcface/preprocessed_disgustdata"

os.makedirs(TARGET_PATH, exist_ok=True)

# Contrast enhancement factor (1.0 - 2.0 in range)
CONTRAST_FACTOR = 1.5

for i, filename in enumerate(os.listdir(SOURCE_PATH)):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(SOURCE_PATH, filename)
        img = Image.open(img_path).convert('RGB')
        img_resized = img.resize((256, 256), Image.LANCZOS)
        img_array = np.array(img_resized)

        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply histogram equalization to the L channel (lightness)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_eq = clahe.apply(l)
        
        # Merge equalized L channel with original A and B channels
        lab_eq = cv2.merge((l_eq, a, b))
        
        # Convert back to RGB
        img_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)
        img_eq_pil = Image.fromarray(img_eq)
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(img_eq_pil)
        img_enhanced = enhancer.enhance(CONTRAST_FACTOR)
        
        # Save to target path
        new_path = os.path.join(TARGET_PATH, f"angry_{i:05d}.png")
        img_enhanced.save(new_path, quality=100)
        
        if i % 100 == 0:
            print(f"Processed {i} images")

print(f"Preprocessing complete. Saved {i+1} images to {TARGET_PATH}")