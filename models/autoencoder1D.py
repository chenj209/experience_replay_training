import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

sys.path.append('../utils')
from data_shape import to_inference_shape_torch, to_inference_shape
from models import ResMLP

class ElementWiseMatMul(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ElementWiseMatMul, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        # Defining the weights
        self.weight1 = nn.Parameter(torch.randn(input_dim, output_dim, 30, 96, 144))
        self.bias1 = nn.Parameter(torch.randn(output_dim, 30, 96, 144))

        self.weight2 = nn.Parameter(torch.randn(30, 1, 96, 144))
        self.bias2 = nn.Parameter(torch.randn(output_dim, 96, 144))

    def forward(self, x):
        # Accommodate the batch dimension in einsum
        x = torch.einsum('biklm,inklm->bnklm', x, self.weight1)
        x = x + self.bias1
        x = F.relu(x)
        # Second operation with adjusted einsum
        # 2x8x30x96x144 * 30x1x96x144
        x = torch.einsum('bnklm,ktlm->bnlm', x, self.weight2)
        x = x + self.bias2
        x = F.relu(x)
        return x

class SurfaceMapper(nn.Module):
    def __init__(self, input_dim):
        super(SurfaceMapper, self).__init__()
        self.input_dim = input_dim
        self.fc = nn.Linear(input_dim, 120)  # Maps 4 surface variables to 4x30 dimension

    def forward(self, x):
        x = self.fc(x)
        return x.view(-1, self.input_dim, 30)

class AutoencoderResMLP(nn.Module):
    def __init__(self, input_size, output_size, m, activation, num_blocks, latent_dim=256, region_mask=None, resmlp=True):
        super(AutoencoderResMLP, self).__init__()
        self.surfacemapper = SurfaceMapper(input_size)
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

    def forward(self, x): # (Q,T,dqls, dtls, qtend,stend, cloud,.., ps, radiation_related, lwup)t-1, (Q,T,ps,dqls,dtls)
        # 3D conv 30 perssure
        latent = self.encoder(x) # 256
        x = self.decoder(latent) # reconstruct all inputs
        #print("x shape:", x.shape)
        if self.resmlp_flag:
            x_resmlp = x[:, -122:, :, :] # Q, T, ps, dqls, dtls of current step
            # x_resmlp_ex = F.relu(self.fc(latent)) # 4x96x144
            x_resmlp_ex = latent
            print(x_resmlp_ex.shape)
            # x_resmlp_ex = F.relu(self.fc(latent)) # 4x96x144
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

# Example of using the model
model = ElementWiseMatMul(12, 8)
input_tensor = torch.randn(2, 12, 30, 96, 144)  # Adjusted for batch size 2
output = model(input_tensor)
print(output.shape)
