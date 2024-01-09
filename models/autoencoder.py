import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn

from models import ResMLP

class Encoder(nn.Module):
    def __init__(self, input_size, latent_size):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_size, 680, kernel_size=3, stride=2, padding=1)  # Output: 680x48x72
        self.conv2 = nn.Conv2d(input_size*2, 1024, kernel_size=3, stride=2, padding=1)  # Output: 1024x24x36
        self.conv3 = nn.Conv2d(1024, 2048, kernel_size=3, stride=2, padding=1)  # Output: 2048x12x18
        self.relu = nn.ReLU(inplace=True)
        
        # Batch Normalization layers
        self.bn1 = nn.BatchNorm2d(input_size*2)
        self.bn2 = nn.BatchNorm2d(1024)
        self.bn3 = nn.BatchNorm2d(2048)

        # Flatten and fully connected layer
        self.fc = nn.Linear(2048 * 12 * 18, latent_size*96*144)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        
        # Flatten and pass through fully connected layer
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class Decoder(nn.Module):
    def __init__(self, input_size, latent_size):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        # Fully connected layer
        self.fc = nn.Linear(latent_size*96*144, 2048 * 12 * 18)

        # Deconvolutional layers
        self.conv_transpose1 = nn.ConvTranspose2d(2048, 1024, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv_transpose2 = nn.ConvTranspose2d(1024, 680, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv_transpose3 = nn.ConvTranspose2d(input_size*2, input_size, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu = nn.ReLU(inplace=True)

        # Batch Normalization layers
        self.bn1 = nn.BatchNorm2d(1024)
        self.bn2 = nn.BatchNorm2d(input_size*2)
        self.bn3 = nn.BatchNorm2d(input_size)

    def forward(self, x):
        # Map from latent space to spatial dimensions
        x = self.fc(x)
        x = x.view(-1, 2048, 12, 18)  # Reshape to match the output of the encoder's last convolutional layer

        # Apply deconvolutional layers
        x = self.relu(self.bn1(self.conv_transpose1(x)))
        x = self.relu(self.bn2(self.conv_transpose2(x)))
        x = self.relu(self.bn3(self.conv_transpose3(x)))
        return x

# class Autoencoder(nn.Module):
#     def __init__(self):
#         super(Autoencoder, self).__init__()
#         self.encoder = Encoder()
#         self.decoder = Decoder()

#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.decoder(x)
#         return x


class AutoencoderResMLP(nn.Module):
    def __init__(self, input_size, output_size, m, activation, num_blocks, latent_size):
        super(AutoencoderResMLP, self).__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.encoder = Encoder(input_size, latent_size)
        self.decoder = Decoder(input_size, latent_size)
        self.resmlp = ResMLP(input_size, output_size, m, activation, num_blocks)

    def forward(self, x):
        curr_x = x[:, -122:]
        latent = self.encoder(x)
        x_rec = self.decoder(latent)
        latent = latent.view(-1, self.latent_size, 96,144)
        x = torch.cat((curr_x, latent), dim=1) # x_concat (batch, 122+latent_size, 96, 144)
        # transpose and make x_concat (batch*96*144, channels)
        x = x.transpose(1,2).transpose(2,3).reshape(-1, 122+self.latent_size)
        x = self.resmlp(x)
        return x, x_rec
