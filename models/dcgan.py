import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim, img_size, channels):
        super(Generator, self).__init__()
        self.img_size = img_size
        self.channels = channels
        self.latent_dim = latent_dim
        self.init_size = img_size // 4  # Initial size after projection

        # Project noise into a feature map
        self.fc = nn.Linear(latent_dim, 128 * self.init_size * self.init_size)

        # Convolutional layers for upsampling
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),  # Upsample to img_size//2
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),  # Upsample to img_size
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()  # Output in [-1, 1]
        )

    def forward(self, z):
        out = self.fc(z)
        out = out.view(out.size(0), 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self, img_size, channels):
        super(Discriminator, self).__init__()
        self.img_size = img_size
        self.channels = channels

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [
                nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25)
            ]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # Dynamically calculate the output size after convolutions
        def calculate_output_size(input_size, layers, kernel_size=3, stride=2, padding=1):
            size = input_size
            for _ in range(layers):
                size = (size + 2 * padding - kernel_size) // stride + 1
            return size

        ds_size_h = calculate_output_size(img_size, 4)  # Height
        ds_size_w = calculate_output_size(img_size, 4)  # Width (assuming square input)
        num_features = 128 * ds_size_h * ds_size_w  # 128 is the number of channels
        print(
            f"Discriminator: img_size={img_size}, ds_size_h={ds_size_h}, ds_size_w={ds_size_w}, num_features={num_features}")
        self.fc = nn.Sequential(
            nn.Linear(num_features, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        out = self.model(img)
        # print("Output shape before flattening:", out.shape)
        out = out.view(out.size(0), -1)
        validity = self.fc(out)
        return validity