import sys
sys.path.insert(0, "C:/Users/user/stylegan3")
import pickle
import os
import numpy as np
import PIL.Image
from IPython.display import Image
import matplotlib.pyplot as plt
import torch

os.environ['STYLEGAN3_NO_CUSTOM_OPS'] = '1'
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6'
device = torch.device('cuda')
torch.cuda.empty_cache()


def seed2vec(G, seed):
    """Generate latent vector from seed."""
    return np.random.RandomState(seed).randn(1, G.z_dim)

def display_image(image):
    """Display an image using matplotlib."""
    plt.axis('off')
    plt.imshow(image)
    plt.show()

def get_label(G, device, class_idx):
    """Create label for conditional networks."""
    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        if class_idx is None:
            raise ValueError("Must specify class label with --class when using a conditional network")
        label[:, class_idx] = 1
    return label

def generate_image(device, G, z, truncation_psi=1.0, noise_mode='const', class_idx=None):
    """Generate an image using the StyleGAN model."""
    z = torch.from_numpy(z).to(device)
    
    label = None
    if hasattr(G, 'c_dim') and G.c_dim > 0:
        if class_idx is not None:
            label = get_label(G, device, class_idx)
        else:
            label = torch.zeros([1, G.c_dim], device=device)
    
    try:
        with torch.no_grad():  # Add no_grad to save memory
            img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            return PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB')
    except RuntimeError as e:
        print(f"Error generating image: {str(e)}")
        # If we encounter an error, try clearing memory and try again
        torch.cuda.empty_cache()
        with torch.no_grad():
            img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            return PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB')

def save_image(image, save_path):
    """Save the generated image to the specified path."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Create directories if needed
    image.save(save_path)
    print(f"Image saved at: {save_path}")

# Safely load the StyleGAN model with error handling
try:
    print("Loading model...")
    with open('C:/Users/user/stylegan3/training_results_fear/network-snapshot-000070.pkl', 'rb') as f:
        checkpoint = pickle.load(f)
        
    # Check if G_ema is available in the checkpoint
    if 'G_ema' in checkpoint:
        G = checkpoint['G_ema'].to(device)
    else:
        # Fall back to G if G_ema not available
        G = checkpoint['G'].to(device)
    
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    sys.exit(1)

# Define parameters like seed range
SEED_FROM = 37000
SEED_TO = 38000
SAVE_DIRECTORY = "C:/Users/user/generated_images_sad_new/"  # Set your desired save location

# Create the save directory if it doesn't exist
os.makedirs(SAVE_DIRECTORY, exist_ok=True)

# Generate, display, and save images
total_images = SEED_TO - SEED_FROM
print(f"Starting generation of {total_images} images...")

for i in range(SEED_FROM, SEED_TO):
    try:
        print(f"Generating image for seed {i} ({i-SEED_FROM+1}/{total_images})")
        z = seed2vec(G, i)
        img = generate_image(device, G, z)
        
        # Skip if image generation failed
        if img is None:
            print(f"Skipping seed {i} due to image generation failure")
            continue
            
        save_path = os.path.join(SAVE_DIRECTORY, f"seed_{i}.png")
        save_image(img, save_path)
        
        # Periodically clear CUDA cache to prevent memory issues
        if (i - SEED_FROM + 1) % 10 == 0:
            torch.cuda.empty_cache()
            
    except Exception as e:
        print(f"Error processing seed {i}: {str(e)}")
        # Continue with the next seed instead of stopping
        continue

print(f"Image generation complete. Generated images saved to {SAVE_DIRECTORY}")

