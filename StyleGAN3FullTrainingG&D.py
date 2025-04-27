import os
os.environ['STYLEGAN3_NO_CUSTOM_OPS'] = '1' 
os.environ['CUDA_HOME'] = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4'
os.environ['CUDA_PATH'] = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4'
import sys
import time
import argparse
from datetime import datetime, timedelta
sys.path.append(r'C:\Users\user\stylegan3')
import pickle
import torch
import numpy as np
import dnnlib
import legacy
from PIL import Image
from torch_utils import misc


def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tune StyleGAN3 with custom settings')
    
    parser.add_argument('--pretrained_model', type=str, default=r"C:\Users\user\stylegan3\training_results_angry\network-snapshot-000025.pkl",
                        help='Path to the pretrained model pickle file')
    parser.add_argument('--output_dir', type=str, default=r"C:\Users\user\stylegan3\training_results_angry",
                        help='Output directory for training results')
    parser.add_argument('--dataset_path', type=str, default=r"C:\Users\user\stylegan3\preprocessed_angry",
                        help='Path to custom dataset directory')
    
    # Training settings
    parser.add_argument('--batch_size', type=int, default=4, 
                        help='Batch size for training')
    parser.add_argument('--g_lr', type=float, default=0.003, 
                        help='Learning rate for generator')
    parser.add_argument('--d_lr', type=float, default=0.005, 
                        help='Learning rate for discriminator')
    parser.add_argument('--total_kimg', type=int, default=35, 
                        help='Total training length in thousands of images')
    parser.add_argument('--snapshot_kimg', type=int, default=5, 
                        help='Snapshot interval in thousands of images')
    parser.add_argument('--max_images', type=int, default=5000, 
                        help='Maximum number of training images to use')
    
    # Training modes and model settings
    parser.add_argument('--train_mode', type=str, default='GAN', choices=['GAN', 'generator_only'],
                        help='Training mode: GAN (G+D) or generator_only')
    parser.add_argument('--r1_gamma', type=float, default=5.0, 
                        help='R1 regularization weight for discriminator')
    parser.add_argument('--style_mixing_prob', type=float, default=0.8, 
                        help='Probability of style mixing during training')
    parser.add_argument('--dataset_weight', type=float, default=0.7, 
                        help='Weight blending between original model and new dataset (0-1)')
    
    # Layer freezing settings
    parser.add_argument('--g_unfreeze_blocks', type=str, default='b32,b64,b128,mapping',
                        help='Comma-separated list of generator blocks to unfreeze, e.g., "b32,b64,b128,mapping"')
    parser.add_argument('--d_unfreeze_blocks', type=str, default='b32,b64,b128',
                        help='Comma-separated list of discriminator blocks to unfreeze, e.g., "b32,b64,b128"')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'progress_images'), exist_ok=True)
    with open(os.path.join(args.output_dir, 'hyperparameters.txt'), 'w') as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")

    # Generate progress images
    def generate_progress_images(G_ema, output_dir, num_images=4, seed=42):
        os.makedirs(os.path.join(output_dir, 'progress_images'), exist_ok=True)
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        np.random.seed(seed)
        z = torch.from_numpy(np.random.randn(num_images, G_ema.z_dim)).to(torch.float32).cuda()
        c = None
        if hasattr(G_ema, 'c_dim') and G_ema.c_dim > 0:
            c = torch.zeros(num_images, G_ema.c_dim).cuda()
        # Generate images
        with torch.no_grad():
            imgs = G_ema(z, c)
            imgs = (imgs * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            imgs = imgs.permute(0, 2, 3, 1).cpu().numpy()
        
        # Save images
        for i, img in enumerate(imgs):
            Image.fromarray(img, 'RGB').save(
                os.path.join(output_dir, 'progress_images', f'progress_{timestamp}_{i:02d}.png'))
        
        # Create a grid of images
        grid_size = int(np.sqrt(num_images))
        if grid_size**2 < num_images:
            grid_size += 1
        
        grid_w = grid_size * imgs[0].shape[1]
        grid_h = ((num_images + grid_size - 1) // grid_size) * imgs[0].shape[0]
        grid_img = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
        
        for i, img in enumerate(imgs):
            y = (i // grid_size) * imgs[0].shape[0]
            x = (i % grid_size) * imgs[0].shape[1]
            if y < grid_h and x < grid_w:  # Ensure we're within grid bounds
                h, w = min(imgs[0].shape[0], grid_h-y), min(imgs[0].shape[1], grid_w-x)
                grid_img[y:y+h, x:x+w] = img[:h, :w]
        
        grid_path = os.path.join(output_dir, 'progress_images', f'progress_grid_{timestamp}.png')
        Image.fromarray(grid_img, 'RGB').save(grid_path)
        
        print(f"Generated progress images at {timestamp}")
        return grid_path

    # Function to train StyleGAN3 with both generator and discriminator
    def train_gan(G, D, G_ema, dataset_path, output_dir, args, start_kimg=0):
        print(f'Setting up GAN training with both G and D...')
        device = torch.device('cuda')
        
        # Setup optimizers
        G_opt = torch.optim.Adam(filter(lambda p: p.requires_grad, G.parameters()), lr=args.g_lr, betas=(0.0, 0.99))
        D_opt = torch.optim.Adam(filter(lambda p: p.requires_grad, D.parameters()), lr=args.d_lr, betas=(0.0, 0.99))
        
        # Load dataset images file paths
        print(f'Loading dataset images from {dataset_path}...')
        dataset_files = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if len(dataset_files) == 0:
            raise ValueError(f"No image files found in {dataset_path}")
        
        # Load a subset of images to save memory
        max_images = min(args.max_images, len(dataset_files))
        print(f"Using {max_images} images out of {len(dataset_files)} total images")
        
        # Shuffle and select images
        np.random.shuffle(dataset_files)
        dataset_files = dataset_files[:max_images]
        
        # Create progress file
        progress_file = os.path.join(output_dir, 'training_progress.txt')
        with open(progress_file, 'a') as f:
            f.write(f"\nGAN training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Starting from {start_kimg} kimg\n")
            f.write(f"Total training: {args.total_kimg} kimg with batch size {args.batch_size}\n")
            f.write(f"Dataset: {len(dataset_files)} images, Dataset weight: {args.dataset_weight}\n")
            f.write(f"G learning rate: {args.g_lr}, D learning rate: {args.d_lr}\n")
            f.write(f"R1 regularization: {args.r1_gamma}, Style mixing prob: {args.style_mixing_prob}\n")
        
        # Main Training loop
        print(f"Starting GAN training loop from {start_kimg} kimg...")
        cur_nimg = start_kimg * 1000
        cur_tick = start_kimg // args.snapshot_kimg
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = 0
        
        # EMA update for G_ema
        def update_ema(G, G_ema, beta=0.999):
            with torch.no_grad():
                for p_ema, p in zip(G_ema.parameters(), G.parameters()):
                    if p.requires_grad:
                        p_ema.copy_(p_ema * beta + p * (1 - beta))
        
        # Set up class label tensors if needed
        c_dim = 0
        if hasattr(G, 'c_dim'):
            c_dim = G.c_dim
        
        # Main loop
        batch_idx = 0
        last_image_gen_time = time.time()
        image_gen_interval = 300  # 5 minutes
        
        # Metrics for tracking
        g_losses = []
        d_losses = []
        r1_penalties = []
        
        try:
            while cur_nimg < args.total_kimg * 1000:
                # Create a batch
                batch_files = [dataset_files[(batch_idx * args.batch_size + i) % len(dataset_files)] 
                              for i in range(args.batch_size)]
                batch_idx += 1
                
                # Load and preprocess images
                real_images = []
                for img_file in batch_files:
                    try:
                        img = Image.open(img_file).convert('RGB')
                        # Resize to match the model's output size (1024x1024 in this case)
                        if img.size != (1024, 1024):
                            img = img.resize((1024, 1024), Image.LANCZOS)
                        # Convert to tensor [-1, 1] range
                        img_tensor = torch.from_numpy(np.array(img).transpose(2, 0, 1)).to(torch.float32) / 127.5 - 1
                        real_images.append(img_tensor)
                    except Exception as e:
                        print(f"Error loading image {img_file}: {e}")
                        continue
                
                # Skip this batch if we couldn't load enough images
                if len(real_images) < args.batch_size:
                    print(f"Warning: Only loaded {len(real_images)} images for this batch, skipping...")
                    continue
                
                real_images = torch.stack(real_images).to(device)
                
                # Prepare class labels (even if we don't use them)
                c = None
                if c_dim > 0:
                    c = torch.zeros(args.batch_size, c_dim, device=device)
                
                # Memory management: make sure we have enough GPU memory
                try:
                    # ======== Train Discriminator ========
                    D_opt.zero_grad()
                    
                    # Generate fake images
                    z = torch.randn(args.batch_size, G.z_dim, device=device)
                    fake_images = G(z, c)
                    
                    # Get real/fake scores
                    fake_scores = D(fake_images.detach(), c)
                    real_scores = D(real_images, c)
                    
                    # Non-saturating GAN loss
                    d_loss = torch.nn.functional.softplus(fake_scores).mean()
                    d_loss += torch.nn.functional.softplus(-real_scores).mean()
                    
                    # Optional: R1 regularization (gradient penalty on real images)
                    if args.r1_gamma > 0:
                        real_images.requires_grad = True
                        real_scores = D(real_images, c)
                        r1_grads = torch.autograd.grad(outputs=real_scores.sum(), 
                                                     inputs=real_images, 
                                                     create_graph=True)[0]
                        r1_penalty = r1_grads.square().sum([1,2,3]).mean()
                        r1_penalties.append(r1_penalty.item())
                        
                        # Apply scaled R1 penalty every 16 steps
                        if batch_idx % 16 == 0:
                            d_loss += args.r1_gamma * 0.5 * r1_penalty
                        
                        real_images.requires_grad = False
                    
                    # Backprop and optimize discriminator
                    d_loss.backward()
                    D_opt.step()
                    d_losses.append(d_loss.item())
                    
                    # ======== Train Generator ========
                    G_opt.zero_grad()
                    
                    # Generate fake images again (with new gradients)
                    if args.style_mixing_prob > 0 and np.random.rand() < args.style_mixing_prob:
                        # Style mixing regularization
                        z2 = torch.randn(args.batch_size, G.z_dim, device=device)
                        ws = G.mapping(z, c)
                        ws2 = G.mapping(z2, c)
                        
                        # Choose crossover point
                        crossover = np.random.randint(1, ws.shape[1])
                        ws[:, crossover:] = ws2[:, crossover:]
                        
                        # Generate mixed images
                        fake_images = G.synthesis(ws)
                    else:
                        # Regular generation
                        fake_images = G(z, c)
                    
                    # Get new fake scores from D
                    fake_scores = D(fake_images, c)
                    
                    # Non-saturating GAN loss for generator
                    g_loss = torch.nn.functional.softplus(-fake_scores).mean()
                    
                    # Backprop and optimize generator
                    g_loss.backward()
                    G_opt.step()
                    g_losses.append(g_loss.item())
                    
                    # Update EMA model
                    update_ema(G, G_ema)
                    
                except RuntimeError as e:
                    if 'out of memory' in str(e).lower():
                        print(f"CUDA out of memory error. Clearing cache and reducing batch...")
                        torch.cuda.empty_cache()
                        # If we hit OOM, try with half the batch size next time
                        args.batch_size = max(1, args.batch_size // 2)
                        print(f"Reduced batch size to {args.batch_size}")
                        continue
                    else:
                        raise e
                
                # Accounting
                cur_nimg += args.batch_size
                
                # Print status
                if cur_nimg - tick_start_nimg >= args.snapshot_kimg * 1000 or cur_nimg >= args.total_kimg * 1000:
                    # End of tick
                    tick_end_time = time.time()
                    tick_time = tick_end_time - tick_start_time - maintenance_time
                    
                    # Calculate average losses
                    avg_g_loss = np.mean(g_losses) if g_losses else 0
                    avg_d_loss = np.mean(d_losses) if d_losses else 0
                    avg_r1 = np.mean(r1_penalties) if r1_penalties else 0
                    g_losses, d_losses, r1_penalties = [], [], []  # Reset
                    
                    # Print progress
                    print(f"\rTick {cur_tick}/{args.total_kimg//args.snapshot_kimg}: "
                          f"kimg {cur_nimg/1000:.1f}/{args.total_kimg} "
                          f"time {tick_time:.1f}s G_loss {avg_g_loss:.4f} D_loss {avg_d_loss:.4f}")
                    
                    # Save snapshot
                    maintenance_start = time.time()
                    snapshot_path = os.path.join(output_dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl')
                    try:
                        with open(snapshot_path, 'wb') as f:
                            pickle.dump({'G': G, 'D': D, 'G_ema': G_ema}, f)
                        print(f"Saved snapshot to {snapshot_path}")
                    except Exception as save_error:
                        print(f"Error saving snapshot: {save_error}")
                    
                    # Update progress file
                    with open(progress_file, 'a') as f:
                        f.write(f"\nProgress update at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write(f"kimg {cur_nimg/1000:.1f}/{args.total_kimg}\n")
                        f.write(f"G_loss: {avg_g_loss:.4f}, D_loss: {avg_d_loss:.4f}")
                        if args.r1_gamma > 0:
                            f.write(f", R1_penalty: {avg_r1:.4f}\n")
                        else:
                            f.write("\n")
                        f.write(f"Snapshot saved to {os.path.basename(snapshot_path)}\n")
                    
                    # Generate progress images
                    try:
                        grid_path = generate_progress_images(G_ema, output_dir, num_images=4, seed=int(time.time()) % 1000)
                        with open(progress_file, 'a') as f:
                            f.write(f"Generated progress images: {os.path.basename(grid_path)}\n")
                    except Exception as e:
                        print(f"Error generating progress images: {str(e)}")
                    
                    # Reset for next tick
                    maintenance_time = time.time() - maintenance_start
                    tick_start_nimg = cur_nimg
                    tick_start_time = time.time()
                    cur_tick += 1
                
                # Generate progress images periodically
                current_time = time.time()
                if current_time - last_image_gen_time > image_gen_interval:
                    try:
                        grid_path = generate_progress_images(G_ema, output_dir, num_images=4, seed=int(time.time()) % 1000)
                        with open(progress_file, 'a') as f:
                            f.write(f"Generated progress images: {os.path.basename(grid_path)}\n")
                        last_image_gen_time = current_time
                    except Exception as e:
                        print(f"Error generating progress images: {str(e)}")
                
                # Free up memory occasionally
                if cur_nimg % 10000 == 0:  # Every 10 kimg
                    torch.cuda.empty_cache()
                    
        except KeyboardInterrupt:
            print("Training interrupted by user.")
        except Exception as e:
            print(f"Training error: {str(e)}")
            import traceback
            traceback.print_exc()
            
        return cur_nimg
    
    # Generator-only training function
    def generator_only_training(G, G_ema, dataset_path, output_dir, args, start_kimg=0):
        print(f'Setting up models for generator-only training...')
        device = torch.device('cuda')
        
        # Setup optimizer for generator only
        G_opt = torch.optim.Adam(filter(lambda p: p.requires_grad, G.parameters()), lr=args.g_lr, betas=(0.0, 0.99))
        
        # Load dataset images file paths
        print(f'Loading dataset images from {dataset_path}...')
        dataset_files = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if len(dataset_files) == 0:
            raise ValueError(f"No image files found in {dataset_path}")
        
        # Load a subset of images to save memory
        max_images = min(args.max_images, len(dataset_files))
        print(f"Using {max_images} images out of {len(dataset_files)} total images")
        
        # Shuffle and select images
        np.random.shuffle(dataset_files)
        dataset_files = dataset_files[:max_images]
        
        # Create progress file
        progress_file = os.path.join(output_dir, 'training_progress.txt')
        with open(progress_file, 'a') as f:
            f.write(f"\nGenerator-only training resumed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Resuming from {start_kimg} kimg\n")
            f.write(f"Total training: {args.total_kimg} kimg with batch size {args.batch_size}\n")
            f.write(f"Dataset: {len(dataset_files)} images, Dataset weight: {args.dataset_weight}\n")
        
        # Set up class label tensors if needed
        c_dim = 0
        if hasattr(G, 'c_dim'):
            c_dim = G.c_dim
        
        # Main training loop
        print(f"Starting generator training loop from {start_kimg} kimg...")
        cur_nimg = start_kimg * 1000
        cur_tick = start_kimg // args.snapshot_kimg
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        
        # Feature matching loss function
        def compute_feature_loss(real_images, generated_images):
            # MSE loss with dataset weight for blending
            loss = torch.nn.functional.mse_loss(generated_images, real_images)
            # Apply dataset weight (1.0 = fully use new dataset, 0.0 = preserve original)
            return loss * args.dataset_weight
        
        # EMA update for G_ema
        def update_ema(G, G_ema, beta=0.999):
            with torch.no_grad():
                for p_ema, p in zip(G_ema.parameters(), G.parameters()):
                    if p.requires_grad:
                        p_ema.copy_(p_ema * beta + p * (1 - beta))
        
        # Main loop
        batch_idx = 0
        last_image_gen_time = time.time()
        image_gen_interval = 300  # 5 minutes
        losses = []
        
        try:
            while cur_nimg < args.total_kimg * 1000:
                # Create a batch
                batch_files = [dataset_files[(batch_idx * args.batch_size + i) % len(dataset_files)] 
                              for i in range(args.batch_size)]
                batch_idx += 1
                
                # Load and preprocess images
                real_images = []
                for img_file in batch_files:
                    try:
                        img = Image.open(img_file).convert('RGB')
                        # Resize to match the model's output size (1024x1024 in this case)
                        if img.size != (1024, 1024):
                            img = img.resize((1024, 1024), Image.LANCZOS)
                        # Convert to tensor [-1, 1] range
                        img_tensor = torch.from_numpy(np.array(img).transpose(2, 0, 1)).to(torch.float32) / 127.5 - 1
                        real_images.append(img_tensor)
                    except Exception as e:
                        print(f"Error loading image {img_file}: {e}")
                        continue
                
                # Skip this batch if we couldn't load enough images
                if len(real_images) < args.batch_size:
                    print(f"Warning: Only loaded {len(real_images)} images for this batch, skipping...")
                    continue
                
                real_images = torch.stack(real_images).to(device)
                
                # Prepare class labels (if needed)
                c = None
                if c_dim > 0:
                    c = torch.zeros(args.batch_size, c_dim, device=device)
                
                # Memory management: make sure we have enough GPU memory
                try:
                    # Generate latents matching the real images
                    z = torch.randn(args.batch_size, G.z_dim, device=device)
                    
                    # Train generator with feature matching loss
                    G_opt.zero_grad()
                    
                    # Style mixing regularization (if enabled)
                    if args.style_mixing_prob > 0 and np.random.rand() < args.style_mixing_prob:
                        z2 = torch.randn(args.batch_size, G.z_dim, device=device)
                        ws = G.mapping(z, c)
                        ws2 = G.mapping(z2, c)
                        
                        # Choose crossover point
                        crossover = np.random.randint(1, ws.shape[1])
                        ws[:, crossover:] = ws2[:, crossover:]
                        
                        # Generate mixed images
                        gen_images = G.synthesis(ws)
                    else:
                        gen_images = G(z, c)
                    
                    loss = compute_feature_loss(real_images, gen_images)
                    loss.backward()
                    G_opt.step()
                    losses.append(loss.item())
                    
                    # Update EMA
                    update_ema(G, G_ema)
                    
                except RuntimeError as e:
                    if 'out of memory' in str(e).lower():
                        print(f"CUDA out of memory error. Clearing cache and reducing batch...")
                        torch.cuda.empty_cache()
                        # If we hit OOM, try with half the batch size next time
                        args.batch_size = max(1, args.batch_size // 2)
                        print(f"Reduced batch size to {args.batch_size}")
                        continue
                    else:
                        raise e
                
                # Accounting
                cur_nimg += args.batch_size
                
                # Print status
                if cur_nimg - tick_start_nimg >= args.snapshot_kimg * 1000 or cur_nimg >= args.total_kimg * 1000:
                    # End of tick
                    tick_end_time = time.time()
                    tick_time = tick_end_time - tick_start_time
                    avg_loss = np.mean(losses) if losses else 0
                    losses = []  # Reset
                    
                    # Print progress
                    print(f"\rTick {cur_tick}/{args.total_kimg//args.snapshot_kimg}: "
                          f"kimg {cur_nimg/1000:.1f}/{args.total_kimg} "
                          f"time {tick_time:.1f}s loss {avg_loss:.4f}")
                    
                    # Save snapshot
                    snapshot_path = os.path.join(output_dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl')
                    try:
                        with open(snapshot_path, 'wb') as f:
                            pickle.dump({'G': G, 'G_ema': G_ema}, f)
                        print(f"Saved snapshot to {snapshot_path}")
                    except Exception as save_error:
                        print(f"Error saving snapshot: {save_error}")
                    
                    # Update progress file
                    with open(progress_file, 'a') as f:
                        f.write(f"\nProgress update at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write(f"kimg {cur_nimg/1000:.1f}/{args.total_kimg}\n")
                        f.write(f"Loss: {avg_loss:.4f}\n")
                        f.write(f"Snapshot saved to {os.path.basename(snapshot_path)}\n")
                    
                    # Generate progress images
                    try:
                        grid_path = generate_progress_images(G_ema, output_dir, num_images=4, seed=int(time.time()) % 1000)
                        with open(progress_file, 'a') as f:
                            f.write(f"Generated progress images: {os.path.basename(grid_path)}\n")
                    except Exception as e:
                        print(f"Error generating progress images: {str(e)}")
                    
                    # Reset for next tick
                    tick_start_nimg = cur_nimg
                    tick_start_time = time.time()
                    cur_tick += 1
                
                # Generate progress images periodically
                current_time = time.time()
                if current_time - last_image_gen_time > image_gen_interval:
                    try:
                        grid_path = generate_progress_images(G_ema, output_dir, num_images=4, seed=int(time.time()) % 1000)
                        with open(progress_file, 'a') as f:
                            f.write(f"Generated progress images: {os.path.basename(grid_path)}\n")
                        last_image_gen_time = current_time
                    except Exception as e:
                        print(f"Error generating progress images: {str(e)}")
                
                # Free up memory occasionally
                if cur_nimg % 10000 == 0:  # Every 10 kimg
                    torch.cuda.empty_cache()
                    
        except KeyboardInterrupt:
            print("Training interrupted by user.")
        except Exception as e:
            print(f"Training error: {str(e)}")
            import traceback
            traceback.print_exc()
            
        return cur_nimg

    # Load checkpoint model instead of pretrained model
    print(f'Loading networks from "{args.pretrained_model}"...')
    with open(args.pretrained_model, 'rb') as f:
        checkpoint = pickle.load(f)

    # Check if we have both G and D in the checkpoint
    has_discriminator = 'D' in checkpoint
    
    # Load models
    G = checkpoint['G'].cuda()
    G_ema = checkpoint['G_ema'].cuda()
    
    if has_discriminator and args.train_mode == 'GAN':
        D = checkpoint['D'].cuda()
        print("Loaded discriminator model successfully")
    elif args.train_mode == 'GAN':
        print("No discriminator found in checkpoint. Creating a new discriminator...")
        D_model_path = r"C:\Users\user\stylegan3\models\stylegan3-r-ffhq-1024x1024.pkl"
        if os.path.exists(D_model_path) and D_model_path != args.pretrained_model:
            with open(D_model_path, 'rb') as f:
                D_checkpoint = pickle.load(f)
                if 'D' in D_checkpoint:
                    D = D_checkpoint['D'].cuda()
                    print(f"Loaded discriminator from {D_model_path}")
                else:
                    raise ValueError("Could not find discriminator in the backup model")
        else:
            raise ValueError("Cannot train in GAN mode without a discriminator model")
    else:
        D = None  # Not needed for generator-only training
        print("No discriminator loaded (generator-only training)")
    
    # Unfreeze specified layers in generator
    # First, freeze all parameters by default
    for param in G.parameters():
        param.requires_grad = False
    
    # Parse the unfreeze blocks parameter and unfreeze those specific blocks
    blocks_to_unfreeze = args.g_unfreeze_blocks.split(',')
    
    # Track unfrozen parameters
    unfrozen_params = 0
    total_params = 0
    
    for name, param in G.named_parameters():
        total_params += param.numel()
        
        # Unfreeze mapping network if specified
        if 'mapping' in blocks_to_unfreeze and 'mapping' in name:
            param.requires_grad = True
            unfrozen_params += param.numel()
        
        # Unfreeze specific synthesis blocks
        for block in blocks_to_unfreeze:
            if block != 'mapping' and block in name:
                param.requires_grad = True
                unfrozen_params += param.numel()
    
    print(f"Generator: Unfrozen {unfrozen_params:,} parameters ({unfrozen_params/total_params:.1%} of total)")
    
    # If using discriminator, unfreeze specified layers
    if D is not None and args.train_mode == 'GAN':
        # First, freeze all discriminator parameters
        for param in D.parameters():
            param.requires_grad = False
        
        # Parse D unfreeze blocks and unfreeze them
        d_blocks_to_unfreeze = args.d_unfreeze_blocks.split(',')
        
        # Track unfrozen parameters
        d_unfrozen_params = 0
        d_total_params = 0
        
        for name, param in D.named_parameters():
            d_total_params += param.numel()
            
            # Unfreeze specific discriminator blocks
            for block in d_blocks_to_unfreeze:
                if block in name:
                    param.requires_grad = True
                    d_unfrozen_params += param.numel()
        
        print(f"Discriminator: Unfrozen {d_unfrozen_params:,} parameters ({d_unfrozen_params/d_total_params:.1%} of total)")
    
    # Get model class names
    G_class_name = G.__class__.__module__ + '.' + G.__class__.__name__ 
    print(f"Generator class: {G_class_name}")
    
    if D is not None:
        D_class_name = D.__class__.__module__ + '.' + D.__class__.__name__
        print(f"Discriminator class: {D_class_name}")

    # Generate initial samples
    print("Generating initial samples...")
    initial_samples_path = generate_progress_images(G_ema, args.output_dir, num_images=4, seed=42)
    print(f"Initial samples saved to: {initial_samples_path}")

    # Extract the kimg from the checkpoint filename
    model_filename = os.path.basename(args.pretrained_model)
    if 'snapshot' in model_filename:
        # If it's a training snapshot, extract the kimg from the filename
        try:
            start_kimg = int(model_filename.split('-')[-1].split('.')[0])
        except ValueError:
            print("Could not parse kimg from filename, starting from 0")
            start_kimg = 0
    else:
        # If it's a pretrained model, start from 0
        print("Using pretrained model, starting from 0 kimg")
        start_kimg = 0
    
    print(f"Starting from {start_kimg} kimg")
    
    # Start timing
    start_time = time.time()
    
    # Choose training mode based on args
    if args.train_mode == 'GAN' and D is not None:
        print("\nStarting GAN training with generator and discriminator...")
        final_nimg = train_gan(
            G=G, 
            D=D,
            G_ema=G_ema,
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
            args=args,
            start_kimg=start_kimg
        )
    else:
        print("\nStarting generator-only training...")
        final_nimg = generator_only_training(
            G=G, 
            G_ema=G_ema,
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
            args=args,
            start_kimg=start_kimg
        )
    
    # Calculate training time
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"\nTraining finished.")
    print(f"Trained for {final_nimg//1000} kimg")
    print(f"Total training time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    
    # Generate final samples
    print("Generating final samples...")
    final_samples_path = generate_progress_images(G_ema, args.output_dir, num_images=9, seed=999)
    print(f"Final samples saved to: {final_samples_path}")
    
    # Final status to progress file
    progress_file = os.path.join(args.output_dir, 'training_progress.txt')
    with open(progress_file, 'a') as f:
        f.write(f"\nTraining completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total training time: {int(hours)}h {int(minutes)}m {int(seconds)}s\n")