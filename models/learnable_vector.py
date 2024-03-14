import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import numpy as np

sys.path.append('../utils')
from data_shape import to_inference_shape_torch, to_inference_shape
from models import ResMLP

class LearnableLocationResMLP(nn.Module):
    def __init__(self, config, input_size, output_size, m, activation, \
                 num_blocks, sub_region_mask=None, resmlp=True):
        super(LearnableLocationResMLP, self).__init__()
        self.learnable_location = nn.Parameter(
            torch.zeros(1,config["latent_size"], config["input_size"][1], config["input_size"][2]))
        self.resmlp_flag = resmlp
        self.latent_size = config["latent_size"]
        self.latent_window = config["input_size"][1:]
        self.input_size = input_size
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
        print(f"Learnable vector, resmlp: {resmlp}, latent_dim {self.latent_size}")

    def forward(self, x): # (Q,T,ps,dqls, dtls, qtend,stend, radiation_related, cloud, lwup)t-1, (Q,T,ps,dqls,dtls)
        # 3D conv 30 perssure
        #print("x shape:", x.shape)
        #print("learnable_location shape:", self.learnable_location.shape)
        latent = self.learnable_location.repeat_interleave(x.shape[0],dim=0) # 256
        #print("latent shape:", latent.shape)
        x_resmlp = x # Q, T, ps, dqls, dtls of current step
        x_resmlp = torch.cat((x_resmlp, latent), dim=1) # concat 4 extra variable
        if self.sub_region_mask is not None:
            x_resmlp = x_resmlp[:, :, self.sub_region_mask] # 96x144 boolean value
        # print("x_resmlp shape:", x_resmlp.shape)
        x_resmlp = x_resmlp.permute(0, 2, 1).reshape(-1, x_resmlp.shape[1])
        x_resmlp = self.resmlp(x_resmlp) # predict qtend
        return x_resmlp

if __name__ == '__main__':
    import numpy as np
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "dataloader"))
    from preprocess import get_min_max_coords
    input_size = 340
    config = {
        "input_size": [340,29,55],
        "latent_size": 8,
    }
    region_mask = np.load("../consts/pacific_region_mask.npy")
    min_x, max_x, min_y, max_y = get_min_max_coords(region_mask, 2)
    sub_region_mask = region_mask[min_x:max_x, min_y:max_y]
    locresmlp = LearnableLocationResMLP(
        config,
        input_size=340,
        output_size=30,
        m=512,
        activation='relu',
        num_blocks=7,
        sub_region_mask=sub_region_mask
        )
    print(locresmlp)

    for name, module in locresmlp.named_modules():
        num_params = sum(p.numel() for p in module.parameters(recurse=False))
        if num_params > 0:
            layer_size_gb = (num_params * 4) / (1024**3)  # Calculating size in GB
            print(f"{name}: {type(module).__name__}, Parameters: {num_params}, Size: {layer_size_gb:.6f} GB")

    input_data = torch.randn(1, *config["input_size"])  # Batch size of 1
    print(f"Input data shape: {input_data.shape}")
    pred = locresmlp(input_data)
    print(pred.shape)  # Should be the same as input_data's shape
