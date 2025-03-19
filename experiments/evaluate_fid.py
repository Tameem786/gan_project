import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
from pytorch_fid import fid_score
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.baseline_gan_celeba import Generator as BaselineGenerator
from models.dcgan import Generator as DCGANGenerator

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
latent_dim = 100
img_size = 64  # Use 64 for CelebA, 28 for MNIST
channels = 3   # Use 3 for CelebA, 1 for MNIST
num_images = 10000  # Number of images to generate for FID

# Load dataset (example for CelebA)
transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = torchvision.datasets.ImageFolder(root='./data/celeba', transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

# Function to generate images
def generate_images(generator, num_images, latent_dim, device, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    generator.eval()
    with torch.no_grad():
        for i in range(0, num_images, 64):
            batch_size = min(64, num_images - i)
            noise = torch.randn(batch_size, latent_dim, device=device)
            fake_images = generator(noise)
            fake_images = fake_images * 0.5 + 0.5  # Denormalize to [0, 1]
            for j in range(batch_size):
                save_image(fake_images[j], f"{output_dir}/img_{i+j}.png")

# Function to save real images
def save_real_images(dataloader, num_images, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    count = 0
    for real_images, _ in dataloader:
        for j in range(real_images.size(0)):
            if count >= num_images:
                break
            real_images[j] = real_images[j] * 0.5 + 0.5  # Denormalize to [0, 1]
            save_image(real_images[j], f"{output_dir}/real_{count}.png")
            count += 1
        if count >= num_images:
            break

# Load models
baseline_g = BaselineGenerator(latent_dim, img_size, channels).to(device)
dcgan_g = DCGANGenerator(latent_dim, img_size, channels).to(device)

# Load pretrained weights (replace with your trained model paths)
baseline_g.load_state_dict(torch.load('results/baseline_gan_celeba/generator.pth'))
dcgan_g.load_state_dict(torch.load('results/dcgan_celeba/generator.pth'))

# Generate images
generate_images(baseline_g, num_images, latent_dim, device, 'results/fid/baseline')
generate_images(dcgan_g, num_images, latent_dim, device, 'results/fid/dcgan')
save_real_images(dataloader, num_images, 'results/fid/real')

# Compute FID scores
fid_baseline = fid_score.calculate_fid_given_paths(['results/fid/real', 'results/fid/baseline'], batch_size=50, device=device, dims=2048)
fid_dcgan = fid_score.calculate_fid_given_paths(['results/fid/real', 'results/fid/dcgan'], batch_size=50, device=device, dims=2048)

print(f"FID Scores:\nBaseline GAN: {fid_baseline:.2f}\nDCGAN: {fid_dcgan:.2f}")