"""
DCFace Emotion Fine-tuning Framework - Complete Version with Enhanced Visualization
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
import torchvision.utils as vutils
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
plt.ioff()  # Turn off interactive mode
from tqdm import tqdm
import random
import glob
from torchvision.transforms.functional import to_tensor, normalize
import time
from contextlib import nullcontext
import warnings

# Fix Intel OpenMP conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Import for safe model loading
import torch.serialization
try:
    from omegaconf import DictConfig, OmegaConf
    from omegaconf.base import ContainerMetadata
    import typing
    # Add more safe globals - including dict which is needed
    torch.serialization.add_safe_globals([DictConfig, ContainerMetadata, typing.Any, dict])
    print("Added all required classes to safe globals for secure model loading")
except ImportError:
    print("omegaconf or typing not found, will use weights_only=False if needed")

# Filter warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

#############################################
# CONFIGURATION
#############################################

CONFIG = {
    # Dataset parameters
    "data_root": "C:/Users/user/dcface/dcface/preprocessed_disgustdata",
    "emotion": "disgust",
    "sample_limit": None,
    
    # Model parameters
    "latent_dim": 512,
    "ngf": 64,
    "ndf": 64,
    
    # Training parameters
    "epochs": 500,
    "batch_size": 64,
    "lr_g": 1e-06,
    "lr_d": 1e-06,
    "beta1": 0.5,
    "beta2": 0.999,
    "workers": 4,
    "n_critic": 1,
    "gpu": 0,
    "seed": 42,
    "mixed_precision": True,
    "clip_grad": True,
    "max_grad_norm": 1.0,
    "label_smoothing": True,
    
    
    # Loading and saving
    #"pretrained": "C:/Users/user/dcface/dcface/pretrained_models/dcface_5x5.ckpt",
    #"resume": None,
    #"reset_disc": False,
    #"checkpoint_dir": "./disgust_checkpoints",
    #"allow_unsafe_load": True,
    
    
    # Loading and saving - CHANGE THIS PART
    "pretrained": None,  # Set to None since we're resuming
    "resume": "C:/Users/user/dcface/dcface/disgust_checkpoints/disgust/dcface_disgust_epoch_339.pt",  # Point to your last checkpoint
    "reset_disc": False,
    "checkpoint_dir": "./disgust_checkpoints",
    "allow_unsafe_load": True,
    
    # Visualization
    "save_interval": 5,
    "plot_interval": 10,
    "vis_interval": 400,
    "num_samples": 16,
    "sample_rows": 4,
    "interactive_display": False,
}

#############################################
# MODEL DEFINITIONS
#############################################

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

class Discriminator(nn.Module):
    """Discriminator Architecture"""
    def __init__(self, ndf=64):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            # Input: 3 x 256 x 256
            nn.Conv2d(3, ndf, 5, stride=2, padding=2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # State size: (ndf) x 128 x 128
            nn.Conv2d(ndf, ndf * 2, 5, stride=2, padding=2, bias=False),
            nn.LayerNorm([ndf * 2, 64, 64]),
            nn.LeakyReLU(0.2, inplace=True),
            
            # State size: (ndf*2) x 64 x 64
            nn.Conv2d(ndf * 2, ndf * 4, 5, stride=2, padding=2, bias=False),
            nn.LayerNorm([ndf * 4, 32, 32]),
            nn.LeakyReLU(0.2, inplace=True),
            
            # State size: (ndf*4) x 32 x 32
            nn.Conv2d(ndf * 4, ndf * 8, 5, stride=2, padding=2, bias=False),
            nn.LayerNorm([ndf * 8, 16, 16]),
            nn.LeakyReLU(0.2, inplace=True),
            
            # State size: (ndf*8) x 16 x 16
            nn.Conv2d(ndf * 8, ndf * 16, 5, stride=2, padding=2, bias=False),
            nn.LayerNorm([ndf * 16, 8, 8]),
            nn.LeakyReLU(0.2, inplace=True),
            
            # State size: (ndf*16) x 8 x 8
            nn.Conv2d(ndf * 16, ndf * 32, 5, stride=2, padding=2, bias=False),
            nn.LayerNorm([ndf * 32, 4, 4]),
            nn.LeakyReLU(0.2, inplace=True),
            
            # State size: (ndf*32) x 4 x 4
            nn.Flatten(),
            nn.Linear(ndf * 32 * 4 * 4, 1),
        )
        
    def forward(self, img):
        return self.model(img)

class EmotionDataset(Dataset):
    """Custom Dataset for emotion images"""
    def __init__(self, root_dir, emotion, transform=None, sample_limit=None):
        self.root_dir = root_dir
        self.emotion = emotion
        self.transform = transform
        
        # Support both directory structures
        potential_paths = [
            os.path.join(root_dir, emotion),
            root_dir
        ]
        
        # Find all image files
        self.image_paths = []
        for path in potential_paths:
            if os.path.exists(path):
                for ext in ['*.jpg', '*.jpeg', '*.png']:
                    self.image_paths.extend(glob.glob(os.path.join(path, ext)))
                
                if self.image_paths:
                    print(f"Found images in: {path}")
                    break
        
        # Limit sample count if specified
        if sample_limit:
            self.image_paths = self.image_paths[:sample_limit]
            
        print(f"Loaded {len(self.image_paths)} images for emotion: {emotion}")
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Apply transform if provided
            if self.transform:
                image = self.transform(image)
            else:
                # Default transform
                image = to_tensor(image)
                image = normalize(image, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                
            return image
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            placeholder = torch.zeros((3, 256, 256))
            return placeholder

#############################################
# UTILITY FUNCTIONS
#############################################

def set_seed(seed):
    """Ensure reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def initialize_weights(model):
    """Initialize weights with Xavier normal distribution"""
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_normal_(model.weight.data, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(model.weight.data, 0.02)
        nn.init.constant_(model.bias.data, 0)

def generate_and_save_samples(netG, fixed_noise, epoch, iteration, config, device):
    """Generate samples and save them"""
    netG.eval()
    with torch.no_grad():
        # Generate fake images
        fake_images = netG(fixed_noise).detach().cpu()
        
        # Number of images to display
        n_samples = min(config["num_samples"], fixed_noise.size(0))
        rows = config["sample_rows"]
        cols = n_samples // rows
        
        # Create sample grid
        sample_grid = vutils.make_grid(
            fake_images[:n_samples], 
            nrow=cols, 
            padding=2, 
            normalize=True
        )
        
        # Save images to file
        sample_dir = os.path.join(config["checkpoint_dir"], 'samples')
        os.makedirs(sample_dir, exist_ok=True)
        
        # Save the file
        sample_path = os.path.join(sample_dir, f'{config["emotion"]}_epoch_{epoch}_iter_{iteration}.png')
        vutils.save_image(sample_grid, sample_path)
        print(f"\nSample generated at epoch {epoch}, iteration {iteration}")
        print(f"Saved to: {sample_path}")
    
    netG.train()

#############################################
# ENHANCED VISUALIZATION FUNCTIONS
#############################################

def create_training_visualizations(G_losses, D_losses, learning_rates, real_scores, fake_scores, config):
    """Create comprehensive training visualizations and metrics"""
    visualization_dir = os.path.join(config["checkpoint_dir"], 'visualizations')
    os.makedirs(visualization_dir, exist_ok=True)
    
    # Set up a larger figure size for multi-plot display
    plt.figure(figsize=(20, 15))
    
    # 1. Generator and Discriminator Loss Plot
    plt.subplot(2, 2, 1)
    plt.title(f"Generator and Discriminator Loss - {config['emotion']}")
    plt.plot(G_losses, label="Generator", color='blue', alpha=0.7)
    plt.plot(D_losses, label="Discriminator", color='red', alpha=0.7)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    # Create smoothed version for trend analysis
    window_size = min(100, len(G_losses) // 10) if len(G_losses) > 100 else 10
    if window_size > 1:
        G_losses_smooth = np.convolve(G_losses, np.ones(window_size)/window_size, mode='valid')
        D_losses_smooth = np.convolve(D_losses, np.ones(window_size)/window_size, mode='valid')
        
        plt.plot(range(window_size-1, len(G_losses)), G_losses_smooth, 
                 label="Generator (Smoothed)", color='darkblue', linewidth=2)
        plt.plot(range(window_size-1, len(D_losses)), D_losses_smooth, 
                 label="Discriminator (Smoothed)", color='darkred', linewidth=2)
        plt.legend()
    
    # 2. Learning Rate Plot
    plt.subplot(2, 2, 2)
    plt.title(f"Learning Rate - {config['emotion']}")
    plt.plot(learning_rates, label="Learning Rate", color='green')
    plt.xlabel("Epochs")
    plt.ylabel("Learning Rate")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    # 3. Real vs Fake Scores
    if real_scores and fake_scores:
        plt.subplot(2, 2, 3)
        plt.title(f"Discriminator Scores - {config['emotion']}")
        plt.plot(real_scores, label="Real Images", color='blue')
        plt.plot(fake_scores, label="Fake Images", color='red')
        plt.xlabel("Iterations")
        plt.ylabel("Average Score")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
    
    # 4. Loss Ratio (G/D) - Helps assess training stability
    if D_losses and G_losses:
        plt.subplot(2, 2, 4)
        plt.title(f"G/D Loss Ratio - {config['emotion']}")
        # Calculate ratios, handle divisions by near-zero
        ratios = []
        for g, d in zip(G_losses, D_losses):
            if abs(d) > 1e-8:  # Avoid division by very small numbers
                ratios.append(g/d)
            else:
                if len(ratios) > 0:
                    ratios.append(ratios[-1])  # Use previous value
                else:
                    ratios.append(1.0)  # Default if first value
                    
        plt.plot(ratios, label="G/D Ratio", color='purple')
        plt.axhline(y=1.0, color='gray', linestyle='--')
        plt.xlabel("Iterations")
        plt.ylabel("Ratio")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(visualization_dir, f"{config['emotion']}_training_metrics.png"), dpi=300)
    plt.close()
    
    # 5. Create statistics summary
    stats_file = os.path.join(visualization_dir, f"{config['emotion']}_training_stats.txt")
    with open(stats_file, 'w') as f:
        f.write(f"Training Statistics for {config['emotion']}\n")
        f.write("="*50 + "\n\n")
        
        f.write("Loss Statistics:\n")
        f.write(f"  Generator - Final: {G_losses[-1]:.4f}, Min: {min(G_losses):.4f}, Max: {max(G_losses):.4f}, Avg: {sum(G_losses)/len(G_losses):.4f}\n")
        f.write(f"  Discriminator - Final: {D_losses[-1]:.4f}, Min: {min(D_losses):.4f}, Max: {max(D_losses):.4f}, Avg: {sum(D_losses)/len(D_losses):.4f}\n\n")
        
        if real_scores and fake_scores:
            f.write("Discriminator Score Statistics:\n")
            f.write(f"  Real Images - Final: {real_scores[-1]:.4f}, Min: {min(real_scores):.4f}, Max: {max(real_scores):.4f}, Avg: {sum(real_scores)/len(real_scores):.4f}\n")
            f.write(f"  Fake Images - Final: {fake_scores[-1]:.4f}, Min: {min(fake_scores):.4f}, Max: {max(fake_scores):.4f}, Avg: {sum(fake_scores)/len(fake_scores):.4f}\n\n")
        
        f.write("Training Parameters:\n")
        f.write(f"  Epochs: {config['epochs']}\n")
        f.write(f"  Batch Size: {config['batch_size']}\n")
        f.write(f"  Initial Learning Rates - G: {config['lr_g']}, D: {config['lr_d']}\n")
        f.write(f"  Mixed Precision: {config['mixed_precision']}\n")
        
    print(f"Training visualizations saved to {visualization_dir}")
    
def create_image_evolution_grid(config):
    """Create a grid showing the evolution of generated images over time"""
    samples_dir = os.path.join(config["checkpoint_dir"], 'samples')
    output_dir = os.path.join(config["checkpoint_dir"], 'visualizations')
    os.makedirs(output_dir, exist_ok=True)
    
    # Find image samples from different epochs
    sample_files = glob.glob(os.path.join(samples_dir, f"{config['emotion']}_epoch_*_iter_*.png"))
    
    # If we don't have enough samples, just return
    if len(sample_files) < 5:
        print("Not enough sample images for evolution grid")
        return
    
    # Sort files by epoch and iteration
    def extract_epoch_iter(filename):
        parts = os.path.basename(filename).split('_')
        epoch = int(parts[parts.index('epoch') + 1])
        iter_idx = int(parts[parts.index('iter') + 1].split('.')[0])
        return epoch * 10000 + iter_idx  # This gives us a sortable value

    sample_files.sort(key=extract_epoch_iter)
    
    # Select evenly spaced samples
    num_samples = min(16, len(sample_files))
    indices = np.linspace(0, len(sample_files)-1, num_samples, dtype=int)
    selected_files = [sample_files[i] for i in indices]
    
    # Create a grid of images
    plt.figure(figsize=(20, 20))
    for i, file in enumerate(selected_files):
        img = plt.imread(file)
        ax = plt.subplot(4, 4, i+1)
        
        # Extract epoch and iteration for the title
        parts = os.path.basename(file).split('_')
        epoch = parts[parts.index('epoch') + 1]
        iter_idx = parts[parts.index('iter') + 1].split('.')[0]
        
        plt.imshow(img)
        plt.title(f"Epoch {epoch}, Iter {iter_idx}")
        plt.axis('off')
    
    plt.suptitle(f"Evolution of Generated {config['emotion']} Images", fontsize=24)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    
    # Save the evolution grid
    output_path = os.path.join(output_dir, f"{config['emotion']}_evolution_grid.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"Image evolution grid saved to {output_path}")

def create_final_report(config, G_losses, D_losses, real_scores, fake_scores, learning_rates, training_time):
    """Create a comprehensive final training report"""
    report_dir = os.path.join(config["checkpoint_dir"], 'report')
    os.makedirs(report_dir, exist_ok=True)
    
    # Create training visualizations
    create_training_visualizations(G_losses, D_losses, learning_rates, real_scores, fake_scores, config)
    
    # Create image evolution grid
    create_image_evolution_grid(config)
    
    # Create HTML report
    report_path = os.path.join(report_dir, f"{config['emotion']}_training_report.html")
    
    with open(report_path, 'w') as f:
        f.write(f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>DCFace Training Report - {config['emotion']}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                h1, h2, h3 {{ color: #333; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .stats {{ display: flex; flex-wrap: wrap; }}
                .stat-box {{ 
                    background-color: #f5f5f5; 
                    border-radius: 10px; 
                    padding: 20px; 
                    margin: 10px; 
                    flex: 1; 
                    min-width: 200px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }}
                .metric-title {{ font-weight: bold; margin-bottom: 5px; color: #555; }}
                .metric-value {{ font-size: 24px; font-weight: bold; margin-bottom: 10px; color: #333; }}
                .metric-desc {{ font-size: 14px; color: #777; }}
                .images {{ margin-top: 30px; text-align: center; }}
                img {{ max-width: 100%; border-radius: 5px; box-shadow: 0 3px 10px rgba(0,0,0,0.2); margin-top: 20px; }}
                .footer {{ margin-top: 50px; font-size: 12px; color: #999; text-align: center; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                tr:hover {{ background-color: #f5f5f5; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>DCFace Fine-tuning Report: {config['emotion']}</h1>
                <p>Training completed at {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <h2>Training Statistics</h2>
                <div class="stats">
                    <div class="stat-box">
                        <div class="metric-title">Training Time</div>
                        <div class="metric-value">{training_time:.2f} hours</div>
                        <div class="metric-desc">Total training duration</div>
                    </div>
                    <div class="stat-box">
                        <div class="metric-title">Epochs</div>
                        <div class="metric-value">{config['epochs']}</div>
                        <div class="metric-desc">Total training epochs</div>
                    </div>
                    <div class="stat-box">
                        <div class="metric-title">Batch Size</div>
                        <div class="metric-value">{config['batch_size']}</div>
                        <div class="metric-desc">Images per batch</div>
                    </div>
                    <div class="stat-box">
                        <div class="metric-title">Final G Loss</div>
                        <div class="metric-value">{G_losses[-1]:.4f}</div>
                        <div class="metric-desc">Generator loss at end of training</div>
                    </div>
                    <div class="stat-box">
                        <div class="metric-title">Final D Loss</div>
                        <div class="metric-value">{D_losses[-1]:.4f}</div>
                        <div class="metric-desc">Discriminator loss at end of training</div>
                    </div>
                </div>
                
                <h2>Configuration</h2>
                <table>
                    <tr>
                        <th>Parameter</th>
                        <th>Value</th>
                    </tr>
        """)
        
        # Add all config parameters to the table
        for key, value in config.items():
            f.write(f"<tr><td>{key}</td><td>{value}</td></tr>\n")
            
        f.write(f"""
                </table>
                
                <h2>Training Metrics</h2>
                <div class="images">
                    <img src="../visualizations/{config['emotion']}_training_metrics.png" alt="Training Metrics">
                </div>
                
                <h2>Image Evolution</h2>
                <div class="images">
                    <img src="../visualizations/{config['emotion']}_evolution_grid.png" alt="Image Evolution">
                </div>
                
                <div class="footer">
                    <p>Generated by DCFace training framework on {time.strftime('%Y-%m-%d')}</p>
                </div>
            </div>
        </body>
        </html>
        """)
    
    print(f"Final training report saved to {report_path}")
    return report_path

#############################################
# TRAINING FUNCTION
#############################################

def train_emotion_model(config):
    """Main training function"""
    # Create checkpoint directory
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    os.makedirs(os.path.join(config["checkpoint_dir"], 'samples'), exist_ok=True)
    
    # Set seed
    set_seed(config["seed"])
    
    # Set device
    device = torch.device(f"cuda:{config['gpu']}" if (torch.cuda.is_available() and config['gpu'] >= 0) else "cpu")
    print(f"Using device: {device}")
    
    # Set up dataset and dataloader
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    
    dataset = EmotionDataset(
        root_dir=config["data_root"],
        emotion=config["emotion"],
        transform=transform,
        sample_limit=config["sample_limit"]
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["workers"],
        drop_last=True,
        pin_memory=True,
    )
    
    # Check if dataset is empty
    if len(dataset) == 0:
        print("ERROR: Dataset is empty. Please check your data path.")
        return None
    
    # Set up networks
    netG = Generator(latent_dim=config["latent_dim"], ngf=config["ngf"]).to(device)
    netD = Discriminator(ndf=config["ndf"]).to(device)
    
    # Initialize weights if not loading from checkpoint
    if not config["resume"]:
        netG.apply(initialize_weights)
        netD.apply(initialize_weights)
    
    # Load pretrained DCFace model if specified
    if config["pretrained"]:
        print(f"Loading pretrained DCFace model from {config['pretrained']}")
        try:
            # Try with weights_only=True first (safer)
            try:
                checkpoint = torch.load(config["pretrained"], map_location=device, weights_only=True)
                print("Loaded model with weights_only=True")
            except Exception as e:
                print(f"Weights-only loading failed: {e}")
                
                # Only proceed with unsafe loading if explicitly allowed
                if config["allow_unsafe_load"]:
                    print("WARNING: Falling back to weights_only=False which can execute arbitrary code.")
                    checkpoint = torch.load(config["pretrained"], map_location=device, weights_only=False)
                    print("Loaded model with weights_only=False")
                else:
                    print("Unsafe loading not allowed. Set allow_unsafe_load=True if you trust this model.")
                    raise ValueError("Could not load model safely")
            
            # Check if checkpoint is a dict with 'generator' key
            if isinstance(checkpoint, dict) and 'generator' in checkpoint:
                netG.load_state_dict(checkpoint['generator'], strict=False)
                print("Loaded generator from checkpoint dictionary")
                if 'discriminator' in checkpoint and not config["reset_disc"]:
                    netD.load_state_dict(checkpoint['discriminator'], strict=False)
                    print("Loaded discriminator from checkpoint dictionary")
            else:
                # Try loading directly into generator
                netG.load_state_dict(checkpoint, strict=False)
                print("Loaded generator directly from checkpoint")
            
            print("Pretrained model loaded successfully")
        except Exception as e:
            print(f"Error loading pretrained model: {e}")
            print("Continuing with randomly initialized model")
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if config["resume"]:
        print(f"Resuming from checkpoint: {config['resume']}")
        try:
            checkpoint = torch.load(config["resume"], map_location=device, 
                                  weights_only=not config["allow_unsafe_load"])
            netG.load_state_dict(checkpoint['generator'])
            netD.load_state_dict(checkpoint['discriminator'])
            print(f"Resuming from epoch {checkpoint['epoch']}")
            start_epoch = checkpoint['epoch'] + 1
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting from epoch 0")
    
    # Set up optimizers
    optimizerG = optim.Adam(
        netG.parameters(), 
        lr=config["lr_g"], 
        betas=(config["beta1"], config["beta2"])
    )
    
    optimizerD = optim.Adam(
        netD.parameters(), 
        lr=config["lr_d"], 
        betas=(config["beta1"], config["beta2"])
    )
    
    # Learning rate schedulers
    schedulerG = optim.lr_scheduler.CosineAnnealingLR(
        optimizerG, 
        T_max=config["epochs"], 
        eta_min=config["lr_g"] / 10
    )
    
    schedulerD = optim.lr_scheduler.CosineAnnealingLR(
        optimizerD, 
        T_max=config["epochs"], 
        eta_min=config["lr_d"] / 10
    )
    
    # Resume optimizers and schedulers if resuming training
    if config["resume"] and 'optimizer_g' in checkpoint:
        optimizerG.load_state_dict(checkpoint['optimizer_g'])
        optimizerD.load_state_dict(checkpoint['optimizer_d'])
        schedulerG.load_state_dict(checkpoint['scheduler_g'])
        schedulerD.load_state_dict(checkpoint['scheduler_d'])
    
    # Set up loss function
    criterion = nn.BCEWithLogitsLoss()
    
    # Fixed noise for visualization
    fixed_noise = torch.randn(config["num_samples"], config["latent_dim"], device=device)
    
    # FIXED: Mixed precision setup - compatible with PyTorch 2.5+
    if config["mixed_precision"] and torch.cuda.is_available():
        try:
            # For PyTorch 2.5+, use explicit device_type
            from torch.amp import autocast, GradScaler
            scaler = GradScaler()
            print("Using torch.amp with explicit device_type='cuda'")
            def amp_context():
                return autocast(device_type='cuda')
        except (ImportError, TypeError):
            # Fall back to older approach
            from torch.cuda.amp import autocast, GradScaler
            scaler = GradScaler()
            print("Using torch.cuda.amp (legacy version)")
            def amp_context():
                return autocast()
    else:
        # Create dummy context for non-mixed precision
        scaler = None
        config["mixed_precision"] = False
        print("Mixed precision not available, using full precision")
        def amp_context():
            return nullcontext()
    
    # Training loop
    print(f"Starting training for {config['epochs']} epochs")
    global_step = 0
    
    # Lists to track progress
    G_losses = []
    D_losses = []
    real_scores = []
    fake_scores = []
    learning_rates = []
    start_time = time.time()
    
    # Real and fake labels with noise for label smoothing
    real_label = 1.0
    fake_label = 0.0
    
    # Generate initial samples
    generate_and_save_samples(netG, fixed_noise, 0, 0, config, device)
    
    try:
        for epoch in range(start_epoch, config["epochs"]):
            netG.train()
            netD.train()
            
            # Progress bar
            pbar = tqdm(enumerate(dataloader), total=len(dataloader), 
                       desc=f"Epoch {epoch}/{config['epochs']}")
            
            for i, real_images in pbar:
                iteration = i + epoch * len(dataloader)
                
                # Move data to device
                real_images = real_images.to(device)
                batch_size = real_images.size(0)
                
                # Label smoothing
                real_labels = torch.full((batch_size, 1), real_label, device=device)
                if config["label_smoothing"]:
                    real_labels = real_labels - torch.rand(real_labels.shape, device=device) * 0.1
                    
                fake_labels = torch.full((batch_size, 1), fake_label, device=device)
                if config["label_smoothing"]:
                    fake_labels = fake_labels + torch.rand(fake_labels.shape, device=device) * 0.1
                
                ############################
                # (1) Update D network
                ###########################
                netD.zero_grad()
                
                if config["mixed_precision"]:
                    with amp_context():  # This now works correctly with PyTorch 2.5+
                        # Real images
                        output_real = netD(real_images)
                        d_loss_real = criterion(output_real, real_labels)
                        
                        # Track discriminator scores on real images
                        real_score = torch.sigmoid(output_real.detach()).mean().item()
                        real_scores.append(real_score)
                        
                        # Fake images
                        noise = torch.randn(batch_size, config["latent_dim"], device=device)
                        fake_images = netG(noise)
                        output_fake = netD(fake_images.detach())
                        d_loss_fake = criterion(output_fake, fake_labels)
                        
                        # Track discriminator scores on fake images
                        fake_score = torch.sigmoid(output_fake.detach()).mean().item()
                        fake_scores.append(fake_score)
                        
                        # Combined loss
                        d_loss = d_loss_real + d_loss_fake
                    
                    scaler.scale(d_loss).backward()
                    
                    # Clip gradients for stability
                    if config["clip_grad"]:
                        scaler.unscale_(optimizerD)
                        nn.utils.clip_grad_norm_(netD.parameters(), config["max_grad_norm"])
                    
                    scaler.step(optimizerD)
                else:
                    # Standard precision training
                    output_real = netD(real_images)
                    d_loss_real = criterion(output_real, real_labels)
                    
                    # Track discriminator scores on real images
                    real_score = torch.sigmoid(output_real.detach()).mean().item()
                    real_scores.append(real_score)
                    
                    # Fake images
                    noise = torch.randn(batch_size, config["latent_dim"], device=device)
                    fake_images = netG(noise)
                    output_fake = netD(fake_images.detach())
                    d_loss_fake = criterion(output_fake, fake_labels)
                    
                    # Track discriminator scores on fake images
                    fake_score = torch.sigmoid(output_fake.detach()).mean().item()
                    fake_scores.append(fake_score)
                    
                    # Combined loss
                    d_loss = d_loss_real + d_loss_fake
                    d_loss.backward()
                    
                    # Clip gradients for stability
                    if config["clip_grad"]:
                        nn.utils.clip_grad_norm_(netD.parameters(), config["max_grad_norm"])
                    
                    optimizerD.step()
                
                ############################
                # (2) Update G network
                ###########################
                if i % config["n_critic"] == 0:  # Update G every n_critic iterations
                    netG.zero_grad()
                    
                    if config["mixed_precision"]:
                        with amp_context():  # This now works correctly with PyTorch 2.5+
                            # Since we updated D, perform another forward pass of fake images
                            noise = torch.randn(batch_size, config["latent_dim"], device=device)
                            fake_images = netG(noise)
                            output = netD(fake_images)
                            
                            # Calculate G's loss based on D's output
                            g_loss = criterion(output, real_labels)
                        
                        scaler.scale(g_loss).backward()
                        
                        # Clip gradients
                        if config["clip_grad"]:
                            scaler.unscale_(optimizerG)
                            nn.utils.clip_grad_norm_(netG.parameters(), config["max_grad_norm"])
                        
                        scaler.step(optimizerG)
                        scaler.update()
                    else:
                        # Standard precision training
                        noise = torch.randn(batch_size, config["latent_dim"], device=device)
                        fake_images = netG(noise)
                        output = netD(fake_images)
                        
                        # Calculate G's loss based on D's output
                        g_loss = criterion(output, real_labels)
                        g_loss.backward()
                        
                        # Clip gradients
                        if config["clip_grad"]:
                            nn.utils.clip_grad_norm_(netG.parameters(), config["max_grad_norm"])
                        
                        optimizerG.step()
                
                # Track losses
                G_losses.append(g_loss.item())
                D_losses.append(d_loss.item())
                
                # Update progress bar
                pbar.set_postfix({
                    'D_loss': f"{d_loss.item():.4f}", 
                    'G_loss': f"{g_loss.item():.4f}",
                    'LR_G': f"{optimizerG.param_groups[0]['lr']:.6f}"
                })
                
                # Generate and display samples at regular intervals
                if global_step % config["vis_interval"] == 0:
                    generate_and_save_samples(
                        netG, fixed_noise, epoch, global_step, config, device
                    )
                
                global_step += 1
            
            # End of epoch - generate sample images
            generate_and_save_samples(
                netG, fixed_noise, epoch, global_step, config, device
            )
            
            # Update learning rates
            schedulerG.step()
            schedulerD.step()
            learning_rates.append(optimizerG.param_groups[0]['lr'])
            
            # Save checkpoint
            if (epoch + 1) % config["save_interval"] == 0 or epoch == config["epochs"] - 1:
                checkpoint_path = os.path.join(
                    config["checkpoint_dir"], 
                    f"dcface_{config['emotion']}_epoch_{epoch}.pt"
                )
                
                torch.save({
                    'epoch': epoch,
                    'generator': netG.state_dict(),
                    'discriminator': netD.state_dict(),
                    'optimizer_g': optimizerG.state_dict(),
                    'optimizer_d': optimizerD.state_dict(),
                    'scheduler_g': schedulerG.state_dict(),
                    'scheduler_d': schedulerD.state_dict(),
                    'g_loss': g_loss.item(),
                    'd_loss': d_loss.item(),
                }, checkpoint_path)
                
                print(f"Checkpoint saved to {checkpoint_path}")
            
            # Plot progress at regular intervals
            if (epoch + 1) % config["plot_interval"] == 0:
                try:
                    # Create intermediate visualizations
                    create_training_visualizations(
                        G_losses, D_losses, learning_rates, real_scores, fake_scores, config
                    )
                except Exception as e:
                    print(f"Warning: Could not create intermediate visualizations: {e}")
        
        # Calculate total training time
        training_time = (time.time() - start_time) / 3600  # Convert to hours
        
        # Save final model
        final_path = os.path.join(config["checkpoint_dir"], f"dcface_{config['emotion']}_final.pt")
        torch.save({
            'generator': netG.state_dict(),
            'discriminator': netD.state_dict(),
        }, final_path)
        
        print(f"Training complete. Final model saved to {final_path}")
        
        # Create final training report
        report_path = create_final_report(
            config, 
            G_losses, 
            D_losses, 
            real_scores, 
            fake_scores, 
            learning_rates,
            training_time
        )
        print(f"Training report available at: {report_path}")
        
        return final_path
    
    except KeyboardInterrupt:
        print("Training interrupted. Saving current model...")
        interrupted_path = os.path.join(config["checkpoint_dir"], f"dcface_{config['emotion']}_interrupted.pt")
        torch.save({
            'epoch': epoch,
            'generator': netG.state_dict(),
            'discriminator': netD.state_dict(),
            'optimizer_g': optimizerG.state_dict(),
            'optimizer_d': optimizerD.state_dict(),
        }, interrupted_path)
        print(f"Interrupted model saved to {interrupted_path}")
        
        # Create intermediate report
        training_time = (time.time() - start_time) / 3600
        report_path = create_final_report(
            config, 
            G_losses, 
            D_losses, 
            real_scores, 
            fake_scores, 
            learning_rates,
            training_time
        )
        print(f"Intermediate training report available at: {report_path}")
        
        return interrupted_path
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        
        return None

#############################################
# MAIN EXECUTION
#############################################

if __name__ == "__main__":
    # Display current configuration
    print("\n=== DCFace Fine-tuning Configuration ===")
    for key, value in CONFIG.items():
        print(f"{key}: {value}")
    print("=========================================\n")
    
    # Create specific checkpoint directory for this emotion
    CONFIG["checkpoint_dir"] = os.path.join(CONFIG["checkpoint_dir"], CONFIG["emotion"])
    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)
    
    # Start training
    print(f"Starting fine-tuning for emotion: {CONFIG['emotion']}")
    
    # Run training
    model_path = train_emotion_model(CONFIG)
    
    if model_path:
        print("\nTraining completed successfully!")
        print(f"Trained model saved to: {model_path}")
    else:
        print("\nTraining did not complete successfully.")