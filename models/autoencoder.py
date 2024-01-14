import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

sys.path.append('../utils')
from data_shape import to_inference_shape_torch
from models import ResMLP

class Encoder(nn.Module):
    def __init__(self, input_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 512, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(512)  # Batch normalization for the first layer
        self.conv2 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(1024)  # Batch normalization for the second layer
        self.conv3 = nn.Conv2d(1024, 2048, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(2048)  # Batch normalization for the third layer
        # compute the flattened size
        self.flattened_size = 2048 * 12 * 18
        self.fc = nn.Linear(self.flattened_size, 256)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(-1, self.flattened_size)
        x = F.relu(self.fc(x))
        return x

# write a correspoinding decoder
class Decoder(nn.Module):
    def __init__(self, output_dim):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(256, 2048 * 12 * 18)

        self.deconv1 = nn.ConvTranspose2d(2048, 1024, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(1024)  # Batch normalization for the first layer
        self.deconv2 = nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(512)  # Batch normalization for the second layer
        self.deconv3 = nn.ConvTranspose2d(512, output_dim, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        x = F.relu(self.fc(x))
        x = x.view(-1, 2048, 12, 18)
        x = F.relu(self.bn1(self.deconv1(x)))
        x = F.relu(self.bn2(self.deconv2(x)))
        x = torch.sigmoid(self.deconv3(x))  # Using sigmoid for the final layer
        return x

class Encoder3D(nn.Module):
    def __init__(self):
        super(Encoder3D, self).__init__()
        self.conv1 = nn.Conv3d(5, 16, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm3d(16)  # Batch normalization for the first layer
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm3d(32)  # Batch normalization for the second layer
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm3d(64)  # Batch normalization for the third layer

        self.flattened_size = 64 * 4 * 12 * 18
        self.fc = nn.Linear(self.flattened_size, 256)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(-1, self.flattened_size)
        x = F.relu(self.fc(x))
        return x

class Decoder3D(nn.Module):
    def __init__(self):
        super(Decoder3D, self).__init__()
        self.fc = nn.Linear(256, 64 * 4 * 12 * 18)

        self.deconv1 = nn.ConvTranspose3d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm3d(32)  # Batch normalization for the first layer
        self.deconv2 = nn.ConvTranspose3d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm3d(16)  # Batch normalization for the second layer
        self.deconv3 = nn.ConvTranspose3d(16, 5, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        x = F.relu(self.fc(x))
        x = x.view(-1, 64, 4, 12, 18)
        x = F.relu(self.bn1(self.deconv1(x)))
        x = F.relu(self.bn2(self.deconv2(x)))
        x = torch.sigmoid(self.deconv3(x))  # Using sigmoid for the final layer
        return x


class Autoencoder(nn.Module):
    def __init__(self, input_size):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_size)
        self.decoder = Decoder(input_size)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class AutoencoderResMLP(nn.Module):
    def __init__(self, input_size, output_size, m, activation, num_blocks, latent_dim=4):
        super(AutoencoderResMLP, self).__init__()
        self.encoder = Encoder(input_size)
        self.decoder = Decoder(input_size)
        self.resmlp = ResMLP(122+latent_dim, output_size, m, activation, num_blocks)
        self.fc = nn.Linear(256, 4*96*144)

    def forward(self, x):
        latent = self.encoder(x)
        x_resmlp = x[:, -122:, :, :]
        x_resmlp_ex = F.relu(self.fc(latent))
        x_resmlp_ex = x_resmlp_ex.view(-1, 4, 96, 144)
        x_resmlp = torch.cat((x_resmlp, x_resmlp_ex), dim=1)
        # print(x_resmlp.shape)
        x_resmlp = to_inference_shape_torch(x_resmlp)
        x_resmlp = self.resmlp(x_resmlp)
        x = self.decoder(latent)
        return x_resmlp, x

if __name__ == '__main__':
    input_size = 340
    autoencoder = Autoencoder(input_size)
    print(autoencoder)

    # Example input
    input_data = torch.randn(1, 340, 96, 144)  # Batch size of 1
    output = autoencoder(input_data)
    print(output.shape)  # Should be the same as input_data's shape

    for name, module in autoencoder.named_modules():
        num_params = sum(p.numel() for p in module.parameters(recurse=False))
        if num_params > 0:
            layer_size_gb = (num_params * 4) / (1024**3)  # Calculating size in GB
            print(f"{name}: {type(module).__name__}, Parameters: {num_params}, Size: {layer_size_gb:.6f} GB")


    autoencoder = AutoencoderResMLP(input_size, 30, 512, 'relu', 7)
    print(autoencoder)

    # Example input
    input_data = torch.randn(1, 340, 96, 144)  # Batch size of 1
    output1, output2 = autoencoder(input_data)
    print(output1.shape)  # Should be the same as input_data's shape
    print(output2.shape)  # Should be the same as input_data's shape

    for name, module in autoencoder.named_modules():
        num_params = sum(p.numel() for p in module.parameters(recurse=False))
        if num_params > 0:
            layer_size_gb = (num_params * 4) / (1024**3)  # Calculating size in GB
            print(f"{name}: {type(module).__name__}, Parameters: {num_params}, Size: {layer_size_gb:.6f} GB")
