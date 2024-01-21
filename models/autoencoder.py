import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

sys.path.append('../utils')
from data_shape import to_inference_shape_torch, to_inference_shape
from models import ResMLP

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 512, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(512)  # Batch normalization for the first layer
        self.conv2 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(1024)  # Batch normalization for the second layer
        self.conv3 = nn.Conv2d(1024, 2048, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(2048)  # Batch normalization for the third layer
        # compute the flattened size
        self.flattened_size = 2048 * 12 * 18
        self.fc = nn.Linear(self.flattened_size, latent_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        #print("pre flatten:", x.shape)
        x = x.reshape(-1, self.flattened_size)
        x = F.relu(self.fc(x))
        return x

# write a correspoinding decoder
class Decoder(nn.Module):
    def __init__(self, output_dim, latent_dim):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 2048 * 12 * 18)

        self.deconv1 = nn.ConvTranspose2d(2048, 1024, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(1024)  # Batch normalization for the first layer
        self.deconv2 = nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(512)  # Batch normalization for the second layer
        self.deconv3 = nn.ConvTranspose2d(512, output_dim, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        x = F.relu(self.fc(x))
        x = x.reshape(-1, 2048, 12, 18)
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
        self.encoder = Encoder(input_size, 256)
        self.decoder = Decoder(input_size, 256)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class AutoencoderResMLP(nn.Module):
    def __init__(self, input_size, output_size, m, activation, num_blocks, latent_dim=256, region_mask=None, resmlp=True):
        super(AutoencoderResMLP, self).__init__()
        self.encoder = Encoder(input_size, latent_dim)
        self.decoder = Decoder(input_size, latent_dim)
        self.resmlp_flag = resmlp
        if self.resmlp_flag:
            self.resmlp = ResMLP(122+4, output_size, m, activation, num_blocks)
            self.fc = nn.Linear(latent_dim, 4*96*144) # 4x96x144 x 4x96x144 (4xregion_mask)
        if region_mask is not None:
            # check if cuda is available
            self.region_mask = torch.tensor(region_mask, dtype=torch.bool).squeeze()
            #if torch.cuda.is_available():
            #    self.region_mask = self.region_mask.cuda()
        else:
            self.region_mask = None
        self.latent_dim = latent_dim
        print(f"Autoencoder, resmlp: {resmlp}, latent_dim {latent_dim}")

    def forward(self, x): # (Q,T,ps,dqls, dtls, qtend,stend, radiation_related, cloud, lwup)t-1, (Q,T,ps,dqls,dtls)
        # 3D conv 30 perssure
        latent = self.encoder(x) # 256
        x = self.decoder(latent) # reconstruct all inputs
        #print("x shape:", x.shape)
        if self.resmlp_flag:
            x_resmlp = x[:, -122:, :, :] # Q, T, ps, dqls, dtls of current step
            x_resmlp_ex = F.relu(self.fc(latent)) # 4x96x144
            x_resmlp_ex = x_resmlp_ex.view(-1, 4, 96, 144)
            x_resmlp = torch.cat((x_resmlp, x_resmlp_ex), dim=1) # concat 4 extra variable
            x_resmlp = to_inference_shape_torch(x_resmlp)
        #print("x_resmlp shape:", x_resmlp.shape)
            if self.region_mask is not None:
                x_resmlp = x_resmlp[:, self.region_mask] # 96x144 boolean value
                # 3440 grid True
                # 96x144 -> 13824 input for resmlp
                # 3440 -> input for resmlp
            x_resmlp = self.resmlp(x_resmlp) # predict qtend
            return x_resmlp, x
        return None, x

if __name__ == '__main__':
    import numpy as np
    input_size = 340
    # autoencoder = Autoencoder(input_size)
    # print(autoencoder)

    # # Example input
    # input_data = torch.randn(1, 340, 96, 144)  # Batch size of 1
    # output = autoencoder(input_data)
    # print(output.shape)  # Should be the same as input_data's shape

    # for name, module in autoencoder.named_modules():
    #     num_params = sum(p.numel() for p in module.parameters(recurse=False))
    #     if num_params > 0:
    #         layer_size_gb = (num_params * 4) / (1024**3)  # Calculating size in GB
    #         print(f"{name}: {type(module).__name__}, Parameters: {num_params}, Size: {layer_size_gb:.6f} GB")


    autoencoder = AutoencoderResMLP(input_size, 30, 512, 'relu', 7, latent_dim=1024)
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

    region_mask = np.load("../consts/landmask.npy")[None, None]
    region_mask = to_inference_shape(region_mask)
    print(region_mask.shape)

    autoencoder = AutoencoderResMLP(input_size, 30, 512, 'relu', 7, region_mask=region_mask)
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
