# -*- coding: utf-8 -*-
"""
DCFace Image Generator - Enhanced with ID Seeding and Style Mixing
For use in Spyder IDE - Fixed visualization code
"""

# ======================================================================
# CONFIGURATION - Edit these parameters as needed
# ======================================================================
CONFIG = {
    # Model paths
    "model_path": "C:/Users/user/dcface/dcface/disgust_checkpoints/disgust/dcface_disgust_final.pt",
    
    # Output directory
    "output_dir": "./generated_disgust_images",
    
    # Input image directories
    "id_dir": "C:/Users/user/dcface/dcface/sample_images/id_images",  # Path to ID face images directory. Set to None to use random generation
    "style_dir": "C:/Users/user/dcface/dcface/preprocessed_disgustdata",  # Path to style images directory. Set to None to use random styles
    
    # Generation parameters
    "num_images": 16,  # Number of images to generate in random mode
    "rows": 4,  # Number of rows for output grid
    "latent_dim": 512,  # Dimension of latent space
    "truncation": 0.7,  # Truncation factor (lower = better quality but less diversity)
    "alpha": 0.7,  # Style mixing strength (0.0 = only ID, 1.0 = only style)
    
    # Quality and enhancement
    "enhance": True,  # Apply image enhancement
    "contrast": 1.2,  # Contrast enhancement factor
    "brightness": 1.1,  # Brightness enhancement factor
    "sharpness": 1.3,  # Sharpness enhancement factor
    
    # Other settings
    "seed": 42,  # Random seed for reproducibility
    "resolution": (256, 256),  # Output image resolution
    "save_individual": True,  # Save individual images in addition to grid
}
# ======================================================================

import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
from torchvision import transforms
import time
from tqdm import tqdm
import glob
import random
import torchvision.utils as vutils
from pathlib import Path

# Set seed for reproducibility
def set_seed(seed):
    """Ensure reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Generator architecture (must match your training architecture exactly)
class Generator(nn.Module):
    """Generator Architecture (DCFace 5x5)"""
    def __init__(self, latent_dim=512, ngf=64):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.ngf = ngf
        
        # Initial projection from latent space
        self.init_size = 4
        self.l1 = nn.Sequential(
            nn.Linear(latent_dim, ngf * 16 * self.init_size ** 2)
        )
        
        # Main convolutional blocks
        self.conv_blocks = nn.Sequential(
            # State Size: (ngf*16) x 4 x 4
            nn.BatchNorm2d(ngf * 16),
            
            # Upsample to 8x8
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf * 16, ngf * 16, 5, stride=1, padding=2),
            nn.BatchNorm2d(ngf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Upsample to 16x16
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf * 16, ngf * 8, 5, stride=1, padding=2),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Upsample to 32x32
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf * 8, ngf * 4, 5, stride=1, padding=2),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Upsample to 64x64
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf * 4, ngf * 2, 5, stride=1, padding=2),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Upsample to 128x128
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf * 2, ngf, 5, stride=1, padding=2),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Upsample to 256x256
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf, 3, 5, stride=1, padding=2),
            nn.Tanh(),
        )
        
    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], self.ngf * 16, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

# Simple encoder for ID images
class SimpleEncoder(nn.Module):
    """Basic encoder to project images into latent space"""
    def __init__(self, latent_dim=512):
        super(SimpleEncoder, self).__init__()
        
        # Simplified encoder
        self.encoder = nn.Sequential(
            # 256x256 -> 128x128
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.LeakyReLU(0.2),
            
            # 128x128 -> 64x64
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            # 64x64 -> 32x32
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            # 32x32 -> 16x16
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            
            # 16x16 -> 8x8
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            
            # 8x8 -> 4x4
            nn.Conv2d(512, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            
            # Final layers
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, latent_dim)
        )
    
    def forward(self, x):
        return self.encoder(x)

# Load and preprocess an image
def load_image(image_path, target_size=(256, 256)):
    """Load and preprocess image to tensor in [-1, 1] range"""
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    try:
        image = Image.open(image_path).convert('RGB')
        tensor = transform(image).unsqueeze(0)
        return tensor
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

# Load all images from a directory
def load_image_batch(directory, target_size=(256, 256), limit=None):
    """Load multiple images from a directory"""
    if directory is None or not os.path.exists(directory):
        print(f"Directory does not exist: {directory}")
        return None, None
        
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_paths.extend(glob.glob(os.path.join(directory, ext)))
    
    if limit and len(image_paths) > limit:
        image_paths = image_paths[:limit]
    
    if not image_paths:
        print(f"No images found in directory: {directory}")
        return None, None
    
    images = []
    valid_paths = []
    
    for path in tqdm(image_paths, desc="Loading images"):
        img = load_image(path, target_size)
        if img is not None:
            images.append(img)
            valid_paths.append(path)
    
    if not images:
        return None, None
    
    return torch.cat(images, dim=0), valid_paths

# Image enhancement post-processing
def enhance_image(image, config):
    """Apply mild enhancement to the generated images"""
    # Convert to PIL
    if isinstance(image, torch.Tensor):
        # Convert from [-1, 1] to [0, 1] range
        if image.min() < 0:
            image = (image + 1) / 2.0
        # Convert to PIL
        transform = transforms.ToPILImage()
        image = transform(image)
    
    # Apply enhancement
    if config["enhance"]:
        # Contrast adjustment
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(config["contrast"])
        
        # Brightness adjustment
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(config["brightness"])
        
        # Sharpness/clarity adjustment
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(config["sharpness"])
    
    return image

# Truncation trick for improved quality
def generate_truncated_latent(latent_dim, truncation=0.7, batch_size=1):
    """Generate latent vectors using truncation trick for better quality"""
    # Generate random latent vectors
    latent = torch.randn(batch_size, latent_dim)
    
    # Apply truncation trick (clip values to improve quality)
    latent = torch.clamp(latent, -truncation, truncation)
    
    return latent

# Style mixing function
def mix_styles(id_latent, style_latent, alpha=0.5):
    """Mix two latent vectors with given alpha blend factor"""
    return id_latent * (1 - alpha) + style_latent * alpha

# Save a single image to file
def save_image(tensor, save_path, normalize=False, config=None):
    """Save a single image tensor to file"""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Convert from [-1, 1] to [0, 1] if needed
    if normalize:
        tensor = (tensor + 1) / 2.0
    
    # Apply enhancement if requested
    if config and config["enhance"]:
        img_pil = enhance_image(tensor.cpu(), config)
        img_pil.save(save_path)
    else:
        vutils.save_image(tensor.cpu(), save_path, normalize=False)
    
    return save_path

# Create a grid of images
def create_image_grid(images, nrow, save_path=None):
    """Create a grid from a list of images"""
    # Create directory if save_path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    grid = vutils.make_grid(images, nrow=nrow, padding=2, normalize=False)
    
    if save_path:
        vutils.save_image(grid, save_path)
    
    # Return the grid tensor
    return grid

# Main function to generate images
def generate_images(config=None):
    """Generate images based on configuration"""
    if config is None:
        config = CONFIG
    
    # Set random seed for reproducibility
    set_seed(config["seed"])
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(config["output_dir"], exist_ok=True)
    
    # Load generator model
    print(f"Loading generator from {config['model_path']}")
    
    try:
        # Set up generator
        netG = Generator(latent_dim=config["latent_dim"], ngf=64).to(device)
        
        # Load model weights
        checkpoint = torch.load(config["model_path"], map_location=device)
        
        if isinstance(checkpoint, dict) and 'generator' in checkpoint:
            netG.load_state_dict(checkpoint['generator'])
            print("Loaded generator from checkpoint dictionary")
        else:
            netG.load_state_dict(checkpoint)
            print("Loaded generator directly from checkpoint")
            
        print("Generator loaded successfully")
    except Exception as e:
        print(f"Error loading generator: {e}")
        return
    
    # Set model to eval mode
    netG.eval()
    
    # Create encoder (only used if ID or style directories are provided)
    encoder = SimpleEncoder(latent_dim=config["latent_dim"]).to(device)
    
    # Initialize the encoder with reasonable weights (since we don't have a pretrained one)
    for m in encoder.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.normal_(m.weight, 0.0, 0.02)
    
    # Check if we should use ID and style images
    use_id = config["id_dir"] is not None and os.path.exists(config["id_dir"])
    use_style = config["style_dir"] is not None and os.path.exists(config["style_dir"])
    
    # Load ID images if directory is provided
    id_latents = None
    id_images = None
    id_paths = None
    
    if use_id:
        print(f"Loading ID images from {config['id_dir']}")
        id_images, id_paths = load_image_batch(
            config["id_dir"], 
            target_size=config["resolution"], 
            limit=config["rows"]
        )
        
        if id_images is None or id_images.shape[0] == 0:
            print("No valid ID images found. Using random generation.")
            use_id = False
        else:
            print(f"Loaded {id_images.shape[0]} ID images")
            
            # Generate latent vectors for ID images
            encoder.eval()
            with torch.no_grad():
                id_latents = encoder(id_images.to(device))
                print("Encoded ID images to latent vectors")
    
    # Load style images if directory is provided
    style_latents = None
    style_images = None
    style_paths = None
    
    if use_style and use_id:  # Only load style if we have ID images
        print(f"Loading style images from {config['style_dir']}")
        style_images, style_paths = load_image_batch(
            config["style_dir"], 
            target_size=config["resolution"], 
            limit=config["rows"]
        )
        
        if style_images is None or style_images.shape[0] == 0:
            print("No valid style images found. Using random style vectors.")
            use_style = False
        else:
            print(f"Loaded {style_images.shape[0]} style images")
            
            # Generate latent vectors for style images
            with torch.no_grad():
                style_latents = encoder(style_images.to(device))
                print("Encoded style images to latent vectors")
    
    # Generate images based on the configuration
    if use_id and use_style:
        # Style mixing mode
        print("Generating style mixed images...")
        
        # Create directory for style mix images
        style_mix_dir = os.path.join(config["output_dir"], "style_mix")
        os.makedirs(style_mix_dir, exist_ok=True)
        
        # Number of id and style images
        n_ids = id_images.shape[0]
        n_styles = style_images.shape[0]
        
        # Generate original ID images
        with torch.no_grad():
            original_id_images = netG(id_latents)
            original_id_images = (original_id_images + 1) / 2.0  # Convert to [0, 1] range
        
        # Generate original style images
        with torch.no_grad():
            original_style_images = netG(style_latents)
            original_style_images = (original_style_images + 1) / 2.0  # Convert to [0, 1] range
        
        # Save original ID and style images
        if config["save_individual"]:
            for i in range(n_ids):
                id_name = os.path.basename(id_paths[i]).split('.')[0]
                save_path = os.path.join(style_mix_dir, f"id_{id_name}_original.png")
                save_image(original_id_images[i], save_path, config=config)
            
            for i in range(n_styles):
                style_name = os.path.basename(style_paths[i]).split('.')[0]
                save_path = os.path.join(style_mix_dir, f"style_{style_name}_original.png")
                save_image(original_style_images[i], save_path, config=config)
        
        # Generate and save style-mixed images
        all_mixed_images = []
        
        # Add original ID images to the grid
        all_mixed_images.extend([img for img in original_id_images])
        
        # Mix styles with different alpha values
        alphas = [0.3, config["alpha"], 0.9]  # Different mixing strengths
        
        for alpha in alphas:
            mixed_row = []
            
            for i in range(n_ids):
                for j in range(n_styles):
                    # Mix latents
                    mixed_latent = mix_styles(id_latents[i], style_latents[j], alpha)
                    
                    # Generate image
                    with torch.no_grad():
                        mixed_image = netG(mixed_latent.unsqueeze(0))
                        mixed_image = (mixed_image[0] + 1) / 2.0  # Convert to [0, 1] range
                        mixed_row.append(mixed_image)
                    
                    # Save individual mixed image
                    if config["save_individual"]:
                        id_name = os.path.basename(id_paths[i]).split('.')[0]
                        style_name = os.path.basename(style_paths[j]).split('.')[0]
                        
                        save_path = os.path.join(style_mix_dir, 
                                              f"id_{id_name}_style_{style_name}_alpha_{alpha:.1f}.png")
                        
                        save_image(mixed_image, save_path, config=config)
            
            # Add this row to all images
            all_mixed_images.extend(mixed_row)
        
        # Add original style images to the grid
        all_mixed_images.extend([img for img in original_style_images])
        
        # Create and save the grid
        grid_images = torch.stack(all_mixed_images)
        grid_path = os.path.join(config["output_dir"], "style_mix_grid.png")
        create_image_grid(grid_images, nrow=n_styles, save_path=grid_path)
        
        print(f"Style mixing grid saved to {grid_path}")
        
        # Display the grid in Spyder - FIXED CODE
        plt.figure(figsize=(16, 10))
        # Make a displayable grid then convert to numpy for matplotlib
        plt_grid = vutils.make_grid(grid_images, nrow=n_styles)
        plt.imshow(plt_grid.permute(1, 2, 0).cpu().numpy())
        plt.axis('off')
        plt.title("Style Mixing Grid")
        plt.show()
        
    elif use_id:
        # ID seeded generation
        print("Generating emotion images with ID seeds...")
        
        # Create directory for ID-seeded images
        id_seed_dir = os.path.join(config["output_dir"], "id_seed")
        os.makedirs(id_seed_dir, exist_ok=True)
        
        # Number of ID images
        n_ids = id_images.shape[0]
        
        # Generate original ID images
        with torch.no_grad():
            original_id_images = netG(id_latents)
            original_id_images = (original_id_images + 1) / 2.0  # Convert to [0, 1] range
        
        # Generate random emotion latents
        emotion_latents = torch.randn(n_ids, config["latent_dim"]).to(device)
        emotion_latents = torch.clamp(emotion_latents, -config["truncation"], config["truncation"])
        
        # Mix ID and emotion latents
        mixed_latents = mix_styles(id_latents, emotion_latents, config["alpha"])
        
        # Generate emotion-mixed images
        with torch.no_grad():
            emotion_images = netG(mixed_latents)
            emotion_images = (emotion_images + 1) / 2.0  # Convert to [0, 1] range
        
        # Save individual images
        if config["save_individual"]:
            for i in range(n_ids):
                id_name = os.path.basename(id_paths[i]).split('.')[0]
                
                # Save original ID
                save_path = os.path.join(id_seed_dir, f"{id_name}_original.png")
                save_image(original_id_images[i], save_path, config=config)
                
                # Save emotion version
                save_path = os.path.join(id_seed_dir, f"{id_name}_emotion.png")
                save_image(emotion_images[i], save_path, config=config)
        
        # Create and save comparison grid
        grid_images = torch.cat([original_id_images, emotion_images], dim=0)
        grid_path = os.path.join(config["output_dir"], "id_emotion_grid.png")
        create_image_grid(grid_images, nrow=n_ids, save_path=grid_path)
        
        print(f"ID-emotion comparison grid saved to {grid_path}")
        
        # Display the grid in Spyder - FIXED CODE
        plt.figure(figsize=(16, 6))
        # Make a displayable grid then convert to numpy for matplotlib
        plt_grid = vutils.make_grid(grid_images, nrow=n_ids)
        plt.imshow(plt_grid.permute(1, 2, 0).cpu().numpy())
        plt.axis('off')
        plt.title("ID vs Emotion Grid")
        plt.show()
        
    else:
        # Random generation
        print(f"Generating {config['num_images']} random emotion images...")
        
        # Create directory for random images
        random_dir = os.path.join(config["output_dir"], "random")
        os.makedirs(random_dir, exist_ok=True)
        
        # Generate random latents with truncation
        latents = []
        for _ in range(config["num_images"]):
            latent = generate_truncated_latent(
                config["latent_dim"], 
                truncation=config["truncation"]
            )
            latents.append(latent)
        
        latents = torch.cat(latents, dim=0).to(device)
        
        # Generate images
        with torch.no_grad():
            random_images = netG(latents)
            random_images = (random_images + 1) / 2.0  # Convert to [0, 1] range
        
        # Save individual images
        if config["save_individual"]:
            for i in range(config["num_images"]):
                save_path = os.path.join(random_dir, f"random_{i:04d}.png")
                save_image(random_images[i], save_path, config=config)
        
        # Create and save grid
        grid_path = os.path.join(config["output_dir"], "random_grid.png")
        grid = create_image_grid(random_images, nrow=config["rows"], save_path=grid_path)
        
        print(f"Random generation grid saved to {grid_path}")
        
        # Display the grid in Spyder - FIXED CODE
        plt.figure(figsize=(12, 12))
        # Grid is already in the right format so just use it directly
        plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
        plt.axis('off')
        plt.title("Randomly Generated Images")
        plt.show()
    
    print("Image generation complete!")
    return True

# Execute the generation function when run as a script
if __name__ == "__main__":
    # Print current configuration
    print("\n=== DCFace Image Generation Configuration ===")
    for key, value in CONFIG.items():
        print(f"{key}: {value}")
    print("=========================================\n")
    
    # Generate images with the current configuration
    generate_images(CONFIG)