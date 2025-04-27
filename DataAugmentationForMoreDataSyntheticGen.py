import os
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path
import albumentations as A
import random
import shutil

# Configuration
INPUT_DIR = r"C:\Users\user\stylegan3\preprocessed_disgust"  # raw string for Windows paths
OUTPUT_DIR = r"C:\Users\user\stylegan3\augmented_disgust"
NUM_AUGMENTATIONS = 3
ROTATION_RANGE = 15
BRIGHTNESS_RANGE = 0.2
CONTRAST_RANGE = 0.2
FLIP_PROB = 0.5
BLUR_PROB = 0.3
NOISE_PROB = 0.3
INCLUDE_ORIGINALS = True
RANDOM_SEED = 42

def create_augmentation_pipeline():
    return A.Compose([
        A.RandomRotate(limit=ROTATION_RANGE, p=0.8),
        A.RandomBrightnessContrast(
            brightness_limit=BRIGHTNESS_RANGE,
            contrast_limit=CONTRAST_RANGE,
            p=0.8
        ),
        A.HorizontalFlip(p=FLIP_PROB),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=(3, 7), p=1.0),
        ], p=BLUR_PROB),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1.0),
        ], p=NOISE_PROB),
        A.OneOf([
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
        ], p=0.3),
        A.OneOf([
            A.CLAHE(clip_limit=2, p=1.0),
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1.0),
        ], p=0.3),
    ])

def augment_images():
    input_dir = Path(INPUT_DIR)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set random seed for reproducibility
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    image_files = [f for f in input_dir.glob('**/*') if f.suffix.lower() in image_extensions]
    
    print(f"Found {len(image_files)} images in input directory")
    
    # Copy original images if requested
    if INCLUDE_ORIGINALS:
        print("Copying original images...")
        for img_path in tqdm(image_files):
            rel_path = img_path.relative_to(input_dir)
            dest_path = output_dir / rel_path
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_path, dest_path)
    
    # Create augmentation pipeline
    transform = create_augmentation_pipeline()
    
    # Augment images
    print(f"Generating {NUM_AUGMENTATIONS} augmentations per image...")
    total_generated = 0
    
    for img_path in tqdm(image_files):
        try:
            # Read image
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Warning: Could not read {img_path}")
                continue
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Get relative path to maintain directory structure
            rel_path = img_path.relative_to(input_dir)
            base_name = rel_path.stem
            extension = rel_path.suffix
            output_subdir = output_dir / rel_path.parent.relative_to(input_dir) if rel_path.parent != input_dir else output_dir
            output_subdir.mkdir(parents=True, exist_ok=True)
            
            # Generate augmentations
            for i in range(NUM_AUGMENTATIONS):
                augmented = transform(image=img)
                aug_img = augmented['image']
                aug_img = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
                
                # Save augmented image
                output_path = output_subdir / f"{base_name}_aug_{i+1}{extension}"
                cv2.imwrite(str(output_path), aug_img)
                total_generated += 1
                
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    print(f"Augmentation complete! Generated {total_generated} new images.")
    if INCLUDE_ORIGINALS:
        print(f"Total images in output directory: {total_generated + len(image_files)}")
    else:
        print(f"Total images in output directory: {total_generated}")

# Run the augmentation
if __name__ == "__main__":
    augment_images()