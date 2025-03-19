import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
import os

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.dcgan import Generator, Discriminator

# Set random seed for reproducibility
torch.manual_seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters
latent_dim = 100
img_size = 28
channels = 1
batch_size = 64
num_epochs = 50
lr = 0.0002
beta1 = 0.5

# Create directory for saving images
if not os.path.exists('results/dcgan_mnist'):
    os.makedirs('results/dcgan_mnist')

# Load and preprocess MNIST dataset
transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
mnist_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
dataloader = torch.utils.data.DataLoader(mnist_dataset, batch_size=batch_size, shuffle=True)

# Initialize models
generator = Generator(latent_dim, img_size, channels).to(device)
discriminator = Discriminator(img_size, channels).to(device)

# Loss and optimizers
criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

real_label = 1.0
fake_label = 0.0


# Function to denormalize images for visualization
def denormalize(tensor):
    return tensor * 0.5 + 0.5


# Training loop
for epoch in range(num_epochs):
    for i, (real_images, _) in enumerate(dataloader):
        batch_size = real_images.size(0)
        real_images = real_images.to(device)
        real_labels = torch.full((batch_size, 1), real_label, device=device)
        fake_labels = torch.full((batch_size, 1), fake_label, device=device)

        # Train Discriminator
        optimizer_D.zero_grad()
        real_output = discriminator(real_images)
        d_loss_real = criterion(real_output, real_labels)
        noise = torch.randn(batch_size, latent_dim, device=device)
        fake_images = generator(noise)
        fake_output = discriminator(fake_images.detach())
        d_loss_fake = criterion(fake_output, fake_labels)
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_D.step()

        # Train Generator
        optimizer_G.zero_grad()
        noise = torch.randn(batch_size, latent_dim, device=device)
        fake_images = generator(noise)
        fake_output = discriminator(fake_images)
        g_loss = criterion(fake_output, real_labels)
        g_loss.backward()
        optimizer_G.step()

        if i % 300 == 0:
            print(f"Epoch [{epoch}/{num_epochs}] Batch [{i}/{len(dataloader)}] "
                  f"D Loss: {d_loss.item():.4f} G Loss: {g_loss.item():.4f}")

    # Save generated images
    if epoch % 5 == 0 or epoch == num_epochs - 1:
        with torch.no_grad():
            noise = torch.randn(16, latent_dim, device=device)
            fake_images = generator(noise)
            fake_images = denormalize(fake_images)
            save_image(fake_images, f"results/dcgan_mnist/epoch_{epoch}.png", nrow=4, normalize=True)

torch.save(generator.state_dict(), f"results/dcgan_mnist/generator.pth")
torch.save(discriminator.state_dict(), f"results/dcgan_mnist/discriminator.pth")