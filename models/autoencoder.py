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
        #self.conv1 = nn.Conv2d(input_size, input_size*2, kernel_size=3, stride=2, padding=1)  # Output: 680x48x72
        #self.conv2 = nn.Conv2d(input_size*2, 1024, kernel_size=3, stride=2, padding=1)  # Output: 1024x24x36
        #self.conv3 = nn.Conv2d(1024, 2048, kernel_size=3, stride=2, padding=1)  # Output: 2048x12x18
        self.e0 = nn.Conv2d(input_dim, 128, kernel_size=1, padding=1, stride=1) # output: 122x96x184
        self.e11 = nn.Conv2d(input_dim, 256, kernel_size=3, padding=1, stride=1) # output: 122x96x184
        self.e12 = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1) # output: 256x96x184
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 256x48x92

        # input: 256x48x92
        self.e21 = nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=1) # output: 512x48x92
        self.e22 = nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1) # output: 512x48x92
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 512x24x46

        # input: 512x24x46
        self.e31 = nn.Conv2d(512, 1024, kernel_size=3, padding=1, stride=1) # output: 1024x24x46
        self.e32 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1, stride=1) # output: 1024x24x46
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 1024x12x23

        # input: 1024x12x23
        self.e41 = nn.Conv2d(1024, 2048, kernel_size=3, padding=1, stride=1) # output: 2048x12x23
        self.e42 = nn.Conv2d(2048, 2048, kernel_size=3, padding=1, stride=1) # output: 2048x12x23
        self.relu = nn.ReLU(inplace=True)

        # Batch Normalization layers
        #self.bn1 = nn.BatchNorm2d(input_size*2)
        #self.bn2 = nn.BatchNorm2d(1024)
        #self.bn3 = nn.BatchNorm2d(2048)
        self.bn0 = nn.BatchNorm2d(64)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)

        # Flatten and fully connected layer
        #self.fc = nn.Linear(2048 * 12 * 18, latent_size*96*144)
        self.fc = nn.Linear(64 * 12 * 18, latent_size*96*144)

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
        #self.fc = nn.Linear(latent_size*96*144, 2048 * 12 * 18)
        self.fc = nn.Linear(latent_size*96*144, input_size * 12 * 18)

        # Deconvolutional layers
        #self.conv_transpose1 = nn.ConvTranspose2d(2048, 1024, kernel_size=3, stride=2, padding=1, output_padding=1)
        #self.conv_transpose2 = nn.ConvTranspose2d(1024, 680, kernel_size=3, stride=2, padding=1, output_padding=1)
        #self.conv_transpose3 = nn.ConvTranspose2d(input_size*2, input_size, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv_transpose1 = nn.ConvTranspose2d(input_size, input_size, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv_transpose2 = nn.ConvTranspose2d(input_size, input_size, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv_transpose3 = nn.ConvTranspose2d(input_size, input_size, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu = nn.ReLU(inplace=True)

        # Batch Normalization layers
        #self.bn1 = nn.BatchNorm2d(1024)
        #self.bn2 = nn.BatchNorm2d(input_size*2)
        #self.bn3 = nn.BatchNorm2d(input_size)
        self.bn1 = nn.BatchNorm2d(input_size)
        self.bn2 = nn.BatchNorm2d(input_size)
        self.bn3 = nn.BatchNorm2d(input_size)

    def forward(self, x):
        # Map from latent space to spatial dimensions
        x = self.fc(x)
        #x = x.view(-1, 2048, 12, 18)  # Reshape to match the output of the encoder's last convolutional layer
        x = x.view(-1, 340, 12, 18)  # Reshape to match the output of the encoder's last convolutional layer

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
        self.resmlp = ResMLP(122+latent_size, output_size, m, activation, num_blocks)

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

if __name__ == "__main__":
    from torch.profiler import profile, record_function, ProfilerActivity
    batch_size = 1
    input_size = 340
    model = AutoencoderResMLP(input_size,30,512,"relu", 7,4)
    print("Model loaed")
    # Assuming you're using a GPU
    model.float().cuda()
    input_data = torch.randn(batch_size, input_size, 96, 144).float().cuda()
    target_data = torch.randn(96*144, 30).float().cuda()
    criterion = nn.MSELoss()

    # Memory for model parameters
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"Parameter memory: {param_memory / (1024**2):.2f} MB")

    # Forward pass
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as prof:
        with record_function("model_inference"):
            #model(input_data)
            output1, output2 = model(input_data)
    print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
    print(output1.shape)
    print(output2.shape)
    torch.cuda.synchronize()  # Wait for operations to finish
    forward_memory = torch.cuda.memory_allocated()
    print(f"Forward pass memory: {forward_memory / (1024**2):.2f} MB")

    # Backward pass
    loss1 = criterion(output1, target_data)
    loss2 = criterion(output2, input_data)
    loss = loss1+loss2
    loss.backward()
    torch.cuda.synchronize()
    peak_memory = torch.cuda.max_memory_allocated()

    # Print memory usage
    print(f"Peak memory during forward and backward pass: {peak_memory / (1024**2):.2f} MB")
