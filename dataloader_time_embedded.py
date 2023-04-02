from torch.utils import data
import os
import numpy as np
import time
import glob
import re

# norm_vec = np.load('norm_vec.npz')
# channel_max_x = norm_vec['channel_max_x']
# channel_min_x = norm_vec['channel_min_x']
# channel_max_y = norm_vec['channel_max_y']
# channel_min_y = norm_vec['channel_min_y']

# def normalization(data_x, data_y):
#     data_x_norm = (data_x - channel_min_x) / (channel_max_x - channel_min_x) * 2 - 1
#     data_y_norm = (data_y - channel_min_y) / (channel_max_y - channel_min_y) * 2 - 1
#     return data_x_norm, data_y_norm

def normalization(data_x, data_y):
    x = data_x
    y = data_y
#     print('!!!!!!!!!',x.shape,y.shape)

    x[:, 0:30,:,:]  = (x[:, 0:30,:,:] - 0) /(0.0238) * 2 - 1
    x[:,30:60,:,:]  = (x[:,30:60,:,:] - 159) / (323 - 159) * 2 - 1
    x[:,60:90,:,: ] = (x[:,60:90,:,:] + 2.13e-6) / (2.13e-6*2) * 2 - 1
    x[:,90:120,:,:] = (x[:,90:120,:,:] + 3.89e-3) / (3.89e-3*2) * 2 - 1
    x[:,120,:,:]    = (x[:,120,:,:] - 0)/ (1412 - 0)
    x[:,121,:,:]    = (x[:,121,:,:] - 59928) / (105782 - 59928)

    # output data (target)
    y[:, 0:30,:,:] = (y[:, 0:30,:,:] + 3.11e-6) / (3.11e-6*2) * 2 - 1
    y[:,30:60,:,:] = (y[:,30:60,:,:] + 3.63) / (3.63*2) * 2 - 1
    y[:,60:61,:,:]    =  y[:,60:61,:,:] / (2.12e-6) * 2 - 1

    y[:,61:62,:,:]    = (y[:,61:62,:,:] - 0) / (1412 - 0)
    y[:,62:63,:,:]    = (y[:,62:63,:,:] - 0) / (1412 - 0)
    y[:,63:64,:,:]    = (y[:,63:64,:,:] - 0) / (1412 - 0)
    y[:,64:65,:,:]    = (y[:,64:65,:,:] - 0) / (1412 - 0)
    y[:,65:66,:,:]    = (y[:,65:66,:,:] - 0) / (1412 - 0)


    data_x_norm, data_y_norm = x,y

    return data_x_norm, data_y_norm

def filename_to_idx(filename):
    data_pattern = ".*(\d{5})\.npz"
    m = re.search(data_pattern, filename)
    if m is None:
        return -1
    else:
        return int(m.group(1))


class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, file_names, is_train, noise_std = 0):
        ### load the data ###
        x = []
        y = []
        for idx, file_name in enumerate(file_names):
            _file = np.load(file_name)
            tx = _file["data_x"]
            ty = _file["data_y"]
            ############# normalization ###############
            tx, ty = normalization(tx, ty)
            tx = np.transpose(tx, (0, 2, 3, 1))
            ty = np.transpose(ty, (0, 2, 3, 1))
            print(idx, len(file_names), 'x-shape & y-shape:', tx.shape, ty.shape) # (1, 96, 144, 32) (1, 96, 144, 5)
            tx = np.reshape(tx, (-1, tx.shape[-1]))
            ty = np.reshape(ty, (-1, ty.shape[-1]))
            x.append(tx)
            y.append(ty)
        self.x = np.concatenate(x, axis=0)
        self.y = np.concatenate(y, axis=0)
#         print('size of self.x!!!!!!!!!!!!!',np.shape(self.x))
        self.size = self.x.shape[0]
        print(self.x.shape, self.y.shape, self.size)
        self.noise_std = noise_std
        self.is_train = is_train

    def __len__(self):
        'Denotes the total number of samples'
        return self.size

    def __getitem__(self, index):
        'Generates one sample of data'
        x = self.x[index:index+1]
        y = self.y[index:index+1]

        x = x[0]
        y = y[0]

        if self.is_train and self.noise_std>0:
            # print(self.noise_std)
            noise_x = np.random.randn(x.shape[0]) * self.noise_std
            noise_y = np.random.randn(y.shape[0]) * self.noise_std
            x = x + noise_x
            y = y + noise_y

        return x, y

if __name__ == '__main__':
    training_set = Dataset(datadir='/data/nncam_data/image_set', is_train=True, train62=False, noise_std=0)
    trainloader = data.DataLoader(training_set, shuffle=True, batch_size=1, num_workers=4)
    for idx, batch in enumerate(trainloader):
        x, y = batch
        if idx == 0:
            np.save('checkcode_y', y.numpy())
        print(idx, x.size(), y.size())
