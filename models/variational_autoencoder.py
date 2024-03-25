import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import numpy as np

sys.path.append('../utils')
from data_shape import to_inference_shape_torch, to_inference_shape
from models import ResMLP

class ElementWiseMultiplyAddBias(nn.Module):
    def __init__(self, num_features):
        super(ElementWiseMultiplyAddBias, self).__init__()
        # Initialize the weights and biases as learnable parameters
        # num_features should match the number of features in the input
        self.weights = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        # Perform element-wise multiplication
        x = x * self.weights
        # Add bias
        x = x + self.bias
        return x

class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        input_size = config["input_size"]
        channel_sizes = config["channel_sizes"]
        kernel_size = config["kernel_size"]
        strides = config["stride"]
        paddings = config["conv_padding"]
        # output_paddings = config["output_padding"]
        self.fc_sizes = config["fc_sizes"][:]
        self.num_em_layers = config["num_em_layers"]
        self.num_fc_layers = len(config["fc_sizes"])

        conv_layers = [
            nn.Conv2d(input_size[0], channel_sizes[0], kernel_size=kernel_size, stride=strides[0], padding=paddings[0]),
            nn.ReLU(True)
            # GDN(channel_sizes[0])
        ]
        for i in range(len(channel_sizes)-1):
            conv_layers.extend([
                nn.Conv2d(channel_sizes[i], channel_sizes[i+1], kernel_size=kernel_size, stride=strides[i+1], padding=paddings[i+1]),
                nn.ReLU(True),
                # GDN(channel_sizes[i+1])
            ])
        self.conv = nn.Sequential(*conv_layers)

        self.flatten = nn.Flatten(start_dim=1)

        h = input_size[1]
        w = input_size[2]
        for i in range(len(channel_sizes)):
            h = np.floor((h - kernel_size + 2 * paddings[i]) / strides[i]) + 1
            w = np.floor((w - kernel_size + 2 * paddings[i]) / strides[i]) + 1
        # print("h:", h, ",w:", w)
        self.flattened_size = int(channel_sizes[-1] * h * w)

        if self.num_em_layers > 0:
            em_layers1 = []
            for i in range(self.num_em_layers):
                em_layers1.append(ElementWiseMultiplyAddBias(self.flattened_size))
                em_layers1.append(nn.ReLU(True))
                # em_layers1.append(GDN(self.flatten_size))
            # not applying activation to latent space
            self.em1 = nn.Sequential(*em_layers1[:-1])
            em_layers2 = []
            for i in range(self.num_em_layers):
                em_layers2.append(ElementWiseMultiplyAddBias(self.flattened_size))
                em_layers2.append(nn.ReLU(True))
                # em_layers2.append(GDN(self.flattened_size))
            # not applying activation to latent space
            self.em2 = nn.Sequential(*em_layers2[:-1])

        if len(self.fc_sizes) > 0:
            fc_layers1 = []
            self.fc_sizes.insert(0, self.flattened_size)
            for i in range(len(self.fc_sizes)-1):
                fc_layers1.append(nn.Linear(self.fc_sizes[i], self.fc_sizes[i+1]))
                fc_layers1.append(nn.ReLU(True))
                # fc_layers1.append(GDN(self.fc_sizes[i+1]))
            # not applying activation to latent space
            self.fc1 = nn.Sequential(*fc_layers1[:-1])
            fc_layers2 = []
            for i in range(len(self.fc_sizes)-1):
                fc_layers2.append(nn.Linear(self.fc_sizes[i], self.fc_sizes[i+1]))
                fc_layers2.append(nn.ReLU(True))
                # fc_layers2.append(GDN(self.fc_sizes[i+1]))
            # not applying activation to latent space
            self.fc2 = nn.Sequential(*fc_layers2[:-1])

    def forward(self, x):
        # print("x shape:", x.shape)
        x = self.conv(x)
        # print("after conv: ", x.shape)
        x = self.flatten(x)
        # print("flatten: ", x.shape)
        mu = x
        std = x
        if self.num_em_layers > 0:
            mu = self.em1(mu)
            std = self.em2(std)
        if self.num_fc_layers > 0:
            mu = self.fc1(mu)
            std = self.fc2(std)
        return mu, std

# write a correspoinding decoder
class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        input_size = config["input_size"]
        channel_sizes = config["channel_sizes"]
        kernel_size = config["kernel_size"]
        strides = config["stride"]
        paddings = config["deconv_padding"]
        output_paddings = config["output_padding"]
        # print("output padding:", output_paddings)
        self.fc_sizes = config["fc_sizes"][::-1]
        self.num_em_layers = config["num_em_layers"]
        self.num_fc_layers = len(config["fc_sizes"])

        h = input_size[1]
        w = input_size[2]
        for i in range(len(config["conv_padding"])):
            h = np.floor((h - kernel_size + 2 * config["conv_padding"][i]) / config["stride"][i]) + 1
            w = np.floor((w - kernel_size + 2 * config["conv_padding"][i]) / config["stride"][i]) + 1
        self.flattened_size = int(channel_sizes[0] * h * w)

        if len(self.fc_sizes) > 0:
            fc_layers = []
            self.fc_sizes.append(self.flattened_size)
            for i in range(len(self.fc_sizes)-1):
                fc_layers.append(nn.Linear(self.fc_sizes[i], self.fc_sizes[i+1]))
                fc_layers.append(nn.ReLU(True))
                # fc_layers.append(GDN(self.fc_sizes[i+1], inverse=True))
            # self.fc = nn.Sequential(*fc_layers[:-1])
            self.fc = nn.Sequential(*fc_layers)

        if self.num_em_layers > 0:
            em_layers = []
            for i in range(self.num_em_layers):
                em_layers.append(ElementWiseMultiplyAddBias(self.flattened_size))
                em_layers.append(nn.ReLU(True))
                # em_layers.append(GDN(self.flattened_size, inverse=True))
            # self.em = nn.Sequential(*em_layers[:-1])
            self.em = nn.Sequential(*em_layers)

        self.unflatten = nn.Unflatten(
            dim=1,
            unflattened_size=(channel_sizes[0], int(h), int(w))
        )

        deconv_layers = []
        for i in range(len(channel_sizes)-1):
            deconv_layers.extend([
                nn.ConvTranspose2d(channel_sizes[i], channel_sizes[i+1], kernel_size, stride=strides[i], padding=paddings[i], output_padding=output_paddings[i]),
                nn.ReLU(True),
                # GDN(channel_sizes[i+1], inverse=True),
            ])
        deconv_layers.extend([
            nn.ConvTranspose2d(channel_sizes[-1], input_size[0], kernel_size, stride=strides[-1], padding=paddings[-1], output_padding=output_paddings[-1]),
        ])
        self.deconv = nn.Sequential(*deconv_layers)

    def forward(self, x):
        if self.num_fc_layers > 0:
            x = self.fc(x)
        if self.num_em_layers > 0:
            x = self.em(x)
        x = self.unflatten(x)
        x = self.deconv(x)
        # x = torch.sigmoid(x)  # Using sigmoid for the final layer
        return x

class VAE(nn.Module):
    def __init__(self, encoder_config, decoder_config):
        super(VAE, self).__init__()
        self.encoder = Encoder(encoder_config)
        self.decoder = Decoder(decoder_config)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x = self.decoder(z)
        return mu, logvar, x

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


# class Autoencoder(nn.Module):
#     def __init__(self, input_size):
#         super(Autoencoder, self).__init__()
#         self.encoder = Encoder(input_size, 256)
#         self.decoder = Decoder(input_size, 256)

#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.decoder(x)
#         return x

class AutoencoderResMLP(nn.Module):
    def __init__(self, config, input_size, output_size, m, activation, \
                 num_blocks, sub_region_mask=None, resmlp=True):
        super(AutoencoderResMLP, self).__init__()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.resmlp_flag = resmlp
        self.latent_size = config["latent_size"]
        self.latent_window = config["input_size"][1:]
        self.input_size = input_size
        if self.resmlp_flag and self.latent_size > 0:
            self.resmlp = ResMLP(self.input_size+self.latent_size, output_size,
                                  m, activation, num_blocks)
        if sub_region_mask is not None:
            # check if cuda is available
            self.sub_region_mask = torch.tensor(sub_region_mask,
                                                dtype=torch.bool).squeeze()
            #if torch.cuda.is_available():
            #    self.region_mask = self.region_mask.cuda()
        else:
            self.sub_region_mask = None
        print(f"Autoencoder, resmlp: {resmlp}, latent_dim {self.latent_size}")

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def forward(self, x): # (Q,T,ps,dqls, dtls, qtend,stend, radiation_related, cloud, lwup)t-1, (Q,T,ps,dqls,dtls)
        # 3D conv 30 perssure
        # latent = self.encoder(x) # 256
        mu, logvar = self.encoder(x)
        latent = self.reparameterize(mu, logvar)
        x_rec = self.decoder(latent) # reconstruct all inputs
        #print("x shape:", x.shape)
        if self.resmlp_flag and self.latent_size > 0:
            #x_resmlp = x[:, -self.input_size:, :, :] # Q, T, ps, dqls, dtls of current step
            x_resmlp = x # Q, T, ps, dqls, dtls of current step
            # x_resmlp_ex = F.relu(self.fc(latent)) # 4x96x144
            x_resmlp_ex = latent
            #print(x_resmlp_ex.shape)
            # x_resmlp_ex = F.relu(self.fc(latent)) # 4x96x144
            x_resmlp_ex = x_resmlp_ex.view(
                x_resmlp_ex.shape[0],
                self.latent_size, self.latent_window[0], self.latent_window[1])
            x_resmlp = torch.cat((x_resmlp, x_resmlp_ex), dim=1) # concat 4 extra variable
            if self.sub_region_mask is not None:
                x_resmlp = x_resmlp[:, :, self.sub_region_mask] # 96x144 boolean value
            # print("x_resmlp shape:", x_resmlp.shape)
            x_resmlp = x_resmlp.permute(0, 2, 1).reshape(-1, x_resmlp.shape[1])
            x_resmlp = self.resmlp(x_resmlp) # predict qtend
            return x_resmlp, x_rec, mu, logvar
        return None, x_rec, mu, logvar

if __name__ == '__main__':
    import numpy as np
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "dataloader"))
    from preprocess import get_min_max_coords
    input_size = 340
    config = {
        "input_size": [340,29,55],
        # "input_size": [1,28,28],
        "channel_sizes": [512, 256, 128],
        # "channel_sizes": [16, 32, 64],
        "latent_size": 128,
        "kernel_size": 3,
        "stride": [1, 1, 1],
        # "padding": [1, 1, 0],
        "padding": [1, 1, 1],
        # "output_padding": [1, 1, 0],
        "output_padding": [0, 0, 0],
        "fc_sizes": [],
        "num_em_layers": 2,
    }
    autoencoder = Autoencoder(config)
    print(autoencoder)
    for name, module in autoencoder.named_modules():
        num_params = sum(p.numel() for p in module.parameters(recurse=False))
        if num_params > 0:
            layer_size_gb = (num_params * 4) / (1024**3)  # Calculating size in GB
            print(f"{name}: {type(module).__name__}, Parameters: {num_params}, Size: {layer_size_gb:.6f} GB")

    # Example input
    input_data = torch.randn(1, *config["input_size"])  # Batch size of 1
    output = autoencoder(input_data)
    print(output.shape)  # Should be the same as input_data's shape

    region_mask = np.load("../consts/pacific_region_mask.npy")
    min_x, max_x, min_y, max_y = get_min_max_coords(region_mask, 2)
    sub_region_mask = region_mask[min_x:max_x, min_y:max_y]
    autoencoderresmlp = AutoencoderResMLP(
        config,
        input_size=122,
        output_size=30,
        m=512,
        activation='relu',
        num_blocks=7,
        sub_region_mask=sub_region_mask
        )
    print(autoencoderresmlp)

    for name, module in autoencoderresmlp.named_modules():
        num_params = sum(p.numel() for p in module.parameters(recurse=False))
        if num_params > 0:
            layer_size_gb = (num_params * 4) / (1024**3)  # Calculating size in GB
            print(f"{name}: {type(module).__name__}, Parameters: {num_params}, Size: {layer_size_gb:.6f} GB")

    input_data = torch.randn(1, *config["input_size"])  # Batch size of 1
    pred, rec = autoencoderresmlp(input_data)
    print(pred.shape)  # Should be the same as input_data's shape
    print(rec.shape)  # Should be the same as input_data's shape

    # autoencoder = AutoencoderResMLP(input_size, 30, 512, 'relu', 7, latent_dim=96*144*4)
    # print(autoencoder)

    # # Example input
    # input_data = torch.randn(1, 340, 96, 144)  # Batch size of 1
    # output1, output2 = autoencoder(input_data)
    # print(output1.shape)  # Should be the same as input_data's shape
    # print(output2.shape)  # Should be the same as input_data's shape

    # for name, module in autoencoder.named_modules():
    #     num_params = sum(p.numel() for p in module.parameters(recurse=False))
    #     if num_params > 0:
    #         layer_size_gb = (num_params * 4) / (1024**3)  # Calculating size in GB
    #         print(f"{name}: {type(module).__name__}, Parameters: {num_params}, Size: {layer_size_gb:.6f} GB")

    # region_mask = np.load("../consts/landmask.npy")[None, None]
    # region_mask = to_inference_shape(region_mask)
    # print(region_mask.shape)

    # autoencoder = AutoencoderResMLP(input_size, 30, 512, 'relu', 7, region_mask=region_mask)
    # print(autoencoder)

    # # Example input
    # input_data = torch.randn(1, 340, 96, 144)  # Batch size of 1
    # output1, output2 = autoencoder(input_data)
    # print(output1.shape)  # Should be the same as input_data's shape
    # print(output2.shape)  # Should be the same as input_data's shape

    # for name, module in autoencoder.named_modules():
    #     num_params = sum(p.numel() for p in module.parameters(recurse=False))
    #     if num_params > 0:
    #         layer_size_gb = (num_params * 4) / (1024**3)  # Calculating size in GB
    #         print(f"{name}: {type(module).__name__}, Parameters: {num_params}, Size: {layer_size_gb:.6f} GB")
