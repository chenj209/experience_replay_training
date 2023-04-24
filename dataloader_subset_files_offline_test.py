from torch.utils import data
import os
import numpy as np
import time
import glob
import pickle
import re
from nncam_data_explore.src.utility import filename_to_idx

# norm_vec = np.load('norm_vec.npz')
# channel_max_x = norm_vec['channel_max_x']
# channel_min_x = norm_vec['channel_min_x']
# channel_max_y = norm_vec['channel_max_y']
# channel_min_y = norm_vec['channel_min_y']

# def normalization(data_x, data_y):
#     data_x_norm = (data_x - channel_min_x) / (channel_max_x - channel_min_x) * 2 - 1
#     data_y_norm = (data_y - channel_min_y) / (channel_max_y - channel_min_y) * 2 - 1
#     return data_x_norm, data_y_norm

TRAINING_DATAPATH = '/data/nncam_data/image_set/'

def compute_normalization_by_level_const(data_path):
    """
    data_x shape: (1,122,96,144)
    data_y shape: (1,66,96,144)

    To normalize each level to 0-1 range, we use formula: 
    data_norm = (data - data_min) / (data_max - data_min)
    
    This functions stores data_min and data_max for each level
    so that after the inference we can invert the normalization
    """
    data_x_min_max = None
    data_y_min_max = None
    for fn in os.listdir(data_path):
        data = np.load(data_path + fn)
        data_x = data['data_x']
        data_y = data['data_y']
        if data_x_min_max is None:
            # init data_x, data_y min_max to store the results
            data_x = data['data_x']
            data_x_min_max = np.zeros((data_x.shape[1], 2))
            data_x_min_max[:,0] = np.Inf
            data_x_min_max[:,1] = -np.Inf
            data_y_min_max = np.zeros((data_y.shape[1], 2))
            data_y_min_max[:,0] = np.Inf
            data_y_min_max[:,1] = -np.Inf
        for i in range(data_x.shape[1]):
            data_x_min_max[i,0] = np.min([np.min(data_x[:,i,:,:]), data_x_min_max[i,0]])
            data_x_min_max[i,1] = np.max([np.max(data_x[:,i,:,:]), data_x_min_max[i,1]])
        for i in range(data_y.shape[1]):
            data_y_min_max[i,0] = np.min([np.min(data_y[:,i,:,:]), data_y_min_max[i,0]])
            data_y_min_max[i,1] = np.max([np.max(data_y[:,i,:,:]), data_y_min_max[i,1]])

    output_name = f"{data_path.replace('/', '-')}-level_min_max".strip("-")
    with open(f"{output_name}-data-x.npy", "wb") as f:
        np.save(f, data_x_min_max)
    with open(f"{output_name}-data-y.npy", "wb") as f:
        np.save(f, data_y_min_max)

def normalization_by_level(data_x, data_y, data_path):
    output_name = f"{data_path.replace('/', '-')}-level_min_max".strip("-")
    computed_min_max_x = np.load(f"{output_name}-data-x.npy")
    computed_min_max_y = np.load(f"{output_name}-data-y.npy")
    x = data_x
    y = data_y
    for i in range(x.shape[1]):
        x_min, x_max = computed_min_max_x[i, :]
        x[:,i,:,:] = (x[:,i,:,:] - x_min) / (x_max - x_min)
    for i in range(y.shape[1]):
        y_min, y_max = computed_min_max_y[i, :]
        y[:,i,:,:] = (y[:,i,:,:] - y_min) / (y_max - y_min)
        
    data_x_norm, data_y_norm = x, y
    return data_x_norm, data_y_norm


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

class TimeDataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, file_names, is_train, noise_std = 0, debug=False):
        ### load the data ###
        self.debug = debug
        x = []
        y = []
        y_raw = []
        y_label = []
        file_names.sort(key=filename_to_idx)
        for prev_fidx, file_name in enumerate(file_names[1:]):
            prev_data = np.load(file_names[prev_fidx])
            curr_data = np.load(file_name)
            prev_tidx = filename_to_idx(file_names[prev_fidx])
            curr_tidx = filename_to_idx(file_name)
            if int(prev_tidx) != int(curr_tidx)-1:
                print(f"{prev_tidx} != {curr_tidx} + 1, {file_names[prev_fidx]} is not previous timestep of {file_name}")
                continue
            tx = curr_data["data_x"]
            ty = curr_data["data_y"]
            tx_prev = prev_data["data_x"]
            ty_prev = prev_data["data_y"]

            if self.debug:
                ty_raw = np.transpose(ty.copy(), (0, 2, 3, 1))
                ty_raw = np.reshape(ty_raw, (-1, ty_raw.shape[-1]))
                y_raw.append(ty_raw)
                y_label.extend([int(re.search("(\d+)\.npz", file_name).group(1))]*ty_raw.shape[0])

            ############# normalization ###############
            tx, ty = normalization(tx, ty)
            tx_prev, ty_prev = normalization(tx_prev, ty_prev)
            ty_prev = np.delete(ty_prev, 60, axis=1)
            print("1", tx.shape, ty.shape, tx_prev.shape, ty_prev.shape)
            tx = np.transpose(tx, (0, 2, 3, 1))
            ty = np.transpose(ty, (0, 2, 3, 1))
            tx_prev = np.transpose(tx_prev, (0, 2, 3, 1))
            ty_prev = np.transpose(ty_prev, (0, 2, 3, 1))
            tx = np.reshape(tx, (-1, tx.shape[-1]))
            ty = np.reshape(ty, (-1, ty.shape[-1]))
            tx_prev = np.reshape(tx_prev, (-1, tx_prev.shape[-1]))
            ty_prev = np.reshape(ty_prev, (-1, ty_prev.shape[-1]))
            tx_concat = np.concatenate([tx_prev, tx, ty_prev], axis=1)
            x.append(tx_concat)
            y.append(ty)
            print(prev_fidx+1, len(file_names), 'x-shape & y-shape:', tx_concat.shape, ty.shape) # (1, 96, 144, 32) (1, 96, 144, 5)
        self.x = np.concatenate(x, axis=0)
        self.y = np.concatenate(y, axis=0)

        if self.debug:
            self.y_raw = np.concatenate(y_raw, axis=0)
            self.y_label = np.array(y_label)
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
        # for debug
        if self.debug:
            y_raw = self.y_raw[index:index+1]
            y_raw = y_raw[0]
            y_label = self.y_label[index:index+1]

        if self.is_train and self.noise_std>0:
            # print(self.noise_std)
            noise_x = np.random.randn(x.shape[0]) * self.noise_std
            noise_y = np.random.randn(y.shape[0]) * self.noise_std
            x = x + noise_x
            y = y + noise_y

        if self.debug:
            return x, y, y_raw, y_label

        return x, y



class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(
        self, 
        file_names, 
        is_train, 
        norm_type, 
        noise_std = 0, 
        debug=False,
        reshaped=False,
        equator=False
        ):
        """
        norm_type(str):
            1. 'level_norm': normalize input and output to 0-1 by level
            2. 'input_level_norm': normalize input to 0-1 by level, normalize output to 0-1
            2. '01norm': normalize input and output to 0-1
        """
        self.debug = debug
        ### load the data ###
        x = []
        y = []
        y_raw = []
        y_label = []
        for idx, file_name in enumerate(file_names):
            _file = np.load(file_name)
            tx = _file["data_x"]
            ty = _file["data_y"]
            if reshaped:
                tx = inverse_reshape_x(tx)
                ty = inverse_reshape_y(ty)
            if equator:
              assert(tx.shape[2] == 96)
              assert(ty.shape[2] == 96)
              tx = tx[:,:,40:57,:]
              ty = ty[:,:,40:57,:]
            ############# normalization ###############

            # for debug
            if self.debug:
                ty_raw = np.transpose(ty.copy(), (0, 2, 3, 1))
                ty_raw = np.reshape(ty_raw, (-1, ty_raw.shape[-1]))
                y_raw.append(ty_raw)
                y_label.extend([int(re.search("(\d+)\.npz", file_name).group(1))]*ty_raw.shape[0])

            if norm_type == 'level_norm':
                tx_norm, ty_norm = normalization_by_level(tx.copy(), ty.copy(), TRAINING_DATAPATH)
            elif norm_type == 'input_level_norm':
                tx_norm, _ = normalization_by_level(tx.copy(), ty.copy(), TRAINING_DATAPATH)
                _, ty_norm = normalization(tx.copy(), ty.copy())
            elif norm_type == '01norm':
                tx_norm, ty_norm = normalization(tx.copy(), ty.copy())
            else:
                print(f"Skipping norm, norm_type {norm_type} not implemented, choose in 'level_norm', 'input_level_norm' and '01norm'")
            tx_norm = np.transpose(tx_norm, (0, 2, 3, 1))
            ty_norm = np.transpose(ty_norm, (0, 2, 3, 1))


            print(idx, len(file_names), 'x-shape & y-shape:', tx.shape, ty.shape, end='\r') # (1, 96, 144, 32) (1, 96, 144, 5)
            tx_norm = np.reshape(tx_norm, (-1, tx_norm.shape[-1]))
            ty_norm = np.reshape(ty_norm, (-1, ty_norm.shape[-1]))


            x.append(tx_norm)
            y.append(ty_norm)
        self.x = np.concatenate(x, axis=0)
        self.y = np.concatenate(y, axis=0)
        # for debug
        if self.debug:
            self.y_raw = np.concatenate(y_raw, axis=0)
            self.y_label = np.array(y_label)
#         print('size of self.x!!!!!!!!!!!!!',np.shape(self.x))
        self.size = self.x.shape[0]
        if self.debug:
            print(self.x.shape, self.y.shape, self.y_raw.shape, self.size, self.y_label.shape)
        else:
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
        
        # for debug
        if self.debug:
            y_raw = self.y_raw[index:index+1]
            y_raw = y_raw[0]
            y_label = self.y_label[index:index+1]

        if self.is_train and self.noise_std>0:
            # print(self.noise_std)
            noise_x = np.random.randn(x.shape[0]) * self.noise_std
            noise_y = np.random.randn(y.shape[0]) * self.noise_std
            x = x + noise_x
            y = y + noise_y

        if self.debug:
            #return x, y, y_raw, y_label
            return x, y_raw
        return x, y

def inverse_reshape_x(x_reshaped):
  x_ori = np.reshape(x_reshaped.copy(), (-1,96,144,x_reshaped.shape[-1]))
  x_ori = np.transpose(x_ori, (0,3,1,2))
  return x_ori

def inverse_reshape_y(y_reshaped):
  """
  reshape y into the original [document, data_type, longitude, latitude] = [N, 66, 96, 144]
  
  the reshape operation to inverse it the following:
    y = np.transpose(y, (0, 2, 3, 1)) # [N,66,96,144] -> [N,96,144,66]
    y = np.reshape(y, (-1, y.shape[-1])) # [N,96,144,66] -> [N*96*144,66]

  """
  y_ori = np.reshape(y_reshaped.copy(), (-1,96,144,y_reshaped.shape[-1]))
  y_ori = np.transpose(y_ori, (0,3,1,2))
  return y_ori


if __name__ == '__main__':
#    training_set = Dataset(datadir='/data/nncam_data/image_set', is_train=True, train62=False, noise_std=0)
#    trainloader = data.DataLoader(training_set, shuffle=True, batch_size=1, num_workers=4)
#    for idx, batch in enumerate(trainloader):
#        x, y = batch
#        if idx == 0:
#            np.save('checkcode_y', y.numpy())
#        print(idx, x.size(), y.size())

  # test if inverse reshape works
  test_y = np.random.normal(0, 1, size=(10,66,96,144))
  print(test_y)
  y_reshaped = np.transpose(test_y.copy(), (0, 2, 3, 1)) # [N,66,96,144] -> [N,96,144,66]
  y_reshaped = np.reshape(y_reshaped, (-1, y_reshaped.shape[-1])) # [N,96,144,66] -> [N*96*144,66]
  y_ori = inverse_reshape_y(y_reshaped)
  print(np.all(y_ori == test_y))
  y_ori = inverse_reshape_y(y_reshaped[:,:20])
  print(np.all(y_ori == test_y[:,:20,:,:]))
