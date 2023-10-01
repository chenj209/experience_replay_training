import torch.nn as nn
import torch

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        
        # Define the hidden layers
        hidden_layers = []
        in_channels = input_dim  # Initial input channels
        for _ in range(6):
            hidden_layers.append(nn.Linear(in_channels, 512, kernel_size=3, stride=1, padding=1))
            hidden_layers.append(nn.ReLU(inplace=True))
            in_channels = 512  # Next layers will have 512 input channels
        
        # Define the final layer to get the desired output shape
        final_layer = nn.Linear(512, output_dim, kernel_size=3, stride=1, padding=1)
        
        # Combine all layers
        self.layers = nn.Sequential(*hidden_layers, final_layer)

    def forward(self, x):
        return self.layers(x)

class FCN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FCN, self).__init__()
        
        # Define the hidden layers
        hidden_layers = []
        in_channels = input_dim  # Initial input channels
        for _ in range(6):
            hidden_layers.append(nn.Conv2d(in_channels, 512, kernel_size=3, stride=1, padding=1))
            hidden_layers.append(nn.ReLU(inplace=True))
            in_channels = 512  # Next layers will have 512 input channels
        
        # Define the final layer to get the desired output shape
        final_layer = nn.Conv2d(512, output_dim, kernel_size=3, stride=1, padding=1)
        
        # Combine all layers
        self.layers = nn.Sequential(*hidden_layers, final_layer)

    def forward(self, x):
        x = self.layers(x)
        x = x[:, :, :, 20:-20]
        return x

class Unet(nn.Module):
    """
    UNet implementation
    """
    def __init__(self, input_dim, output_dim):
        super(Unet, self).__init__()
        # Encoder
        # In the encoder, convolutional layers with the Conv2d function are used to extract features from the input image. 
        # Each block in the encoder consists of two convolutional layers followed by a max-pooling layer, with the exception of the last block which does not include a max-pooling layer.
        # -------
        # input: 122x96x184
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

        # Decoder
        # input 2048x12x23
        self.upconv1 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2) # output: 1024x24x46
        self.d11 = nn.Conv2d(2048, 1024, kernel_size=3, padding=1) # output: 1024x24x46
        self.d12 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1) # output: 1024x24x46

        self.upconv2 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2) # output: 512x48x92
        self.d21 = nn.Conv2d(1024, 512, kernel_size=3, padding=1) # output: 512x48x92
        self.d22 = nn.Conv2d(512, 512, kernel_size=3, padding=1) # output: 512x48x92

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2) # output: 256x96x184
        self.d31 = nn.Conv2d(512, 256, kernel_size=3, padding=1) # output: 256x96x184
        self.d32 = nn.Conv2d(256, 256, kernel_size=3, padding=1) # output: 256x96x184

        # Output layer
        self.outconv = nn.Conv2d(256, output_dim, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        e11 = self.e11(x)
        e11 = nn.ReLU(inplace=True)(e11)
        e12 = self.e12(e11)
        e12 = nn.ReLU(inplace=True)(e12)
        pool1 = self.pool1(e12)

        e21 = self.e21(pool1)
        e21 = nn.ReLU(inplace=True)(e21)
        e22 = self.e22(e21)
        e22 = nn.ReLU(inplace=True)(e22)
        pool2 = self.pool2(e22)

        e31 = self.e31(pool2)
        e31 = nn.ReLU(inplace=True)(e31)
        e32 = self.e32(e31)
        e32 = nn.ReLU(inplace=True)(e32)
        pool3 = self.pool3(e32)

        e41 = self.e41(pool3)
        e41 = nn.ReLU(inplace=True)(e41)
        e42 = self.e42(e41)
        e41 = nn.ReLU(inplace=True)(e42)

        # Decoder
        upconv1 = self.upconv1(e42)
        upconv1 = nn.ReLU(inplace=True)(upconv1)
        concat1 = torch.cat([upconv1, e32], dim=1)
        d11 = self.d11(concat1)
        d11 = nn.ReLU(inplace=True)(d11)
        d12 = self.d12(d11)
        d12 = nn.ReLU(inplace=True)(d12)

        upconv2 = self.upconv2(d12)
        upconv2 = nn.ReLU(inplace=True)(upconv2)
        concat2 = torch.cat([upconv2, e22], dim=1)
        d21 = self.d21(concat2)
        d21 = nn.ReLU(inplace=True)(d21)
        d22 = self.d22(d21)
        d22 = nn.ReLU(inplace=True)(d22)

        upconv3 = self.upconv3(d22)
        upconv3 = nn.ReLU(inplace=True)(upconv3)
        concat3 = torch.cat([upconv3, e12], dim=1)
        d31 = self.d31(concat3)
        d31 = nn.ReLU(inplace=True)(d31)
        d32 = self.d32(d31)
        d32 = nn.ReLU(inplace=True)(d32)

        # Output layer
        outconv = self.outconv(d32)
        x = outconv[:, :, :, 20:-20]
        return x


                


if __name__ == "__main__":
    from torchviz import make_dot
    import torch
    # Create an instance of the FCN model
    fcn_model = FCN()
    print(fcn_model)
    unet_model = Unet()
    print(unet_model)

    inputs = torch.rand(1,122,96,184)
    outputs = unet_model(inputs)
    dot = make_dot(outputs)
    dot.view()

