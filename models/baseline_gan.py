import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim, img_size=28, channels=1):
        super(Generator, self).__init__()
        self.img_size = img_size
        self.channels = channels
        self.latent_dim = latent_dim
        self.img_dim = img_size * img_size * channels

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, self.img_dim),
            nn.Tanh()  # Output in [-1, 1]
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), self.channels, self.img_size, self.img_size)
        return img


class Discriminator(nn.Module):
    def __init__(self, img_size=28, channels=1):
        super(Discriminator, self).__init__()
        self.img_size = img_size
        self.channels = channels
        self.img_dim = img_size * img_size * channels

        self.model = nn.Sequential(
            nn.Linear(self.img_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Output probability in [0, 1]
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity