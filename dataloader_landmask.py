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
def get_ps_from_x(x, hyam, hybm):
    ps = x[:, 121, :, :] # 1x96x144 surface pressure
    ps_level = ps[:, np.newaxis, :, :] * np.array(hybm)[np.newaxis, :, np.newaxis, np.newaxis]
    ps_level += np.array(hyam)[np.newaxis, :, np.newaxis, np.newaxis] * 100000
    return ps_level # 30x96x144 pmid

def normalize_x_pmid(data_x):
    x = data_x.copy()

    x[:, 0:30,:,:]  = (x[:, 0:30,:,:] - 0) /(0.0238) * 2 - 1
    x[:,30:60,:,:]  = (x[:,30:60,:,:] - 159) / (323 - 159) * 2 - 1
    x[:,60:90,:,: ] = (x[:,60:90,:,:] + 2.13e-6) / (2.13e-6*2) * 2 - 1
    x[:,90:120,:,:] = (x[:,90:120,:,:] + 3.89e-3) / (3.89e-3*2) * 2 - 1
    x[:,120,:,:]    = (x[:,120,:,:] - 0)/ (1412 - 0)
    x[:,121:,:,:]    = (x[:,121:,:,:] - 0) / (105782 - 0)

    data_x_norm = x

    return data_x_norm

def normalize_x(data_x):
    x = data_x.copy()

    x[:, 0:30,:,:]  = (x[:, 0:30,:,:] - 0) /(0.0238) * 2 - 1
    x[:,30:60,:,:]  = (x[:,30:60,:,:] - 159) / (323 - 159) * 2 - 1
    x[:,60:90,:,: ] = (x[:,60:90,:,:] + 2.13e-6) / (2.13e-6*2) * 2 - 1
    x[:,90:120,:,:] = (x[:,90:120,:,:] + 3.89e-3) / (3.89e-3*2) * 2 - 1
    x[:,120,:,:]    = (x[:,120,:,:] - 0)/ (1412 - 0)
    x[:,121,:,:]    = (x[:,121,:,:] - 59928) / (105782 - 59928)

    data_x_norm = x

    return data_x_norm

def normalize_y(data_y):
    y = data_y.copy()
    # output data (target)
    y[:, 0:30,:,:] = (y[:, 0:30,:,:] + 3.11e-6) / (3.11e-6*2) * 2 - 1
    y[:,30:60,:,:] = (y[:,30:60,:,:] + 3.63) / (3.63*2) * 2 - 1
    y[:,60:61,:,:]    =  y[:,60:61,:,:] / (2.12e-6) * 2 - 1

    y[:,61:62,:,:]    = (y[:,61:62,:,:] - 0) / (1412 - 0)
    y[:,62:63,:,:]    = (y[:,62:63,:,:] - 0) / (1412 - 0)
    y[:,63:64,:,:]    = (y[:,63:64,:,:] - 0) / (1412 - 0)
    y[:,64:65,:,:]    = (y[:,64:65,:,:] - 0) / (1412 - 0)
    y[:,65:66,:,:]    = (y[:,65:66,:,:] - 0) / (1412 - 0)


    data_y_norm = y

    return data_y_norm

def inverse_61_65(y):
    y[:,0] = (y[:,0]) * (1412 - 0)
    y[:,1] = (y[:,1]) * (1412 - 0)
    y[:,2] = (y[:,2]) * (1412 - 0)
    y[:,3] = (y[:,3]) * (1412 - 0)
    y[:,4] = (y[:,4]) * (1412 - 0)
    return y

def inverse_61_64(y):
    y[:,0] = (y[:,0]+1)/2*(332+53)-53       # flns
    y[:,1] = (y[:,1]+1)/2*(419-83)+83       # flnt
    y[:,2] = (y[:,2]+1)/2*(1063+2.13)-2.13  # fsns
    y[:,3] = (y[:,3]+1)/2*(1299)+0          # fsnt
    return y

def get_inverse():
    inverse = {}
    inverse[ '0_29'] = lambda y: (y+1)/2*(3.11e-6*2)-3.11e-6
    inverse['30_59'] = lambda y: (y+1)/2*(3.63*2)-3.63
    inverse['60']    = lambda y: (y+1)/2*(2.12e-6)
    inverse['61_64'] = lambda y: inverse_61_64(y)
    inverse['61_65'] = lambda y: inverse_61_65(y)
    
    return inverse


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

def idx_to_filename(idx):
  return str(idx).rjust(5,'0') + '.npz'

def filename_to_idx(filename):
    data_pattern = ".*(\d{5})\.npz"
    m = re.search(data_pattern, filename)
    if m is None:
        return -1
    else:
        return int(m.group(1))

import numpy as np

class DatasetDiskLandMask(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, file_names, is_train, noise_std = 0, output_normalized=True, silent=True):
        ### load the data ###
        all_files = file_names
        for i in range(17507,17530):
            for file_name in all_files:
                if str(i) in file_name:
                    all_files.remove(file_name)

        #print('after file num:', len(all_files))

        for i in ['00001', '08690', '17522', '26210']:
            for file_name in all_files:
                if i in file_name:
                    all_files.remove(file_name)

        #print('hahahahahah after file num:', len(all_files))
        self.file_names = all_files
        self.silent = silent
        file_names.sort(key=filename_to_idx)
        self.noise_std = noise_std
        self.is_train = is_train
        self.size = len(self.file_names)
        self.output_normalized = output_normalized
        pconsts = np.load(os.path.dirname(os.path.abspath(__file__))+"/phys_consts.npz")
        self.hyam = pconsts["hyam"]
        self.hybm = pconsts["hybm"]
        self.landmask = np.load("landmask.npy")

    def __len__(self):
        'Denotes the total number of samples'
        return self.size

    def __getitem__(self, index):
        'Generates one sample of data'
        x = []
        y = []
        x_raw = []
        _file = self.file_names[index]
        if os.path.exists(_file):
        #for idx, file_name in enumerate(file_names):
            _file = np.load(_file)
            tx_raw = _file["data_x"]
            ty = _file["data_y"]
            ############# normalization ###############
            #tx, ty = normalization(tx, ty)
            #position_encoding = positional_encoding(4,96,144)
            tx = normalize_x(tx_raw)
            tx = np.concatenate((tx, self.landmask[None, None]), axis=1)
            if self.output_normalized:
                ty = normalize_y(ty)
            tx = np.transpose(tx, (0, 2, 3, 1))
            tx_raw = np.transpose(tx_raw, (0, 2, 3, 1))
            ty = np.transpose(ty, (0, 2, 3, 1))
            tx = np.reshape(tx, (-1, tx.shape[-1]))
            tx_raw = np.reshape(tx_raw, (-1, tx_raw.shape[-1]))
            ty = np.reshape(ty, (-1, ty.shape[-1]))
            x.append(tx)
            y.append(ty)
            x_raw.append(tx_raw)
            if not self.silent:
                print(index, len(file_names), 'x-shape & y-shape:', tx.shape, ty.shape) # (1, 96, 144, 32) (1, 96, 144, 5)
        x = np.concatenate(x, axis=0).squeeze()
        x_raw = np.concatenate(x_raw, axis=0).squeeze()
        y = np.concatenate(y, axis=0).squeeze()

        if self.is_train and self.noise_std>0:
            # print(self.noise_std)
            noise_x = np.random.randn(x.shape[0]) * self.noise_std
            noise_y = np.random.randn(y.shape[0]) * self.noise_std
            x = x + noise_x
            y = y + noise_y

        return x, y, x_raw

class DatasetDisk(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, file_names, is_train, noise_std = 0, output_normalized=True, silent=True):
        ### load the data ###
        all_files = file_names
        for i in range(17507,17530):
            for file_name in all_files:
                if str(i) in file_name:
                    all_files.remove(file_name)

        #print('after file num:', len(all_files))

        for i in ['00001', '08690', '17522', '26210']:
            for file_name in all_files:
                if i in file_name:
                    all_files.remove(file_name)

        #print('hahahahahah after file num:', len(all_files))
        self.file_names = all_files
        self.silent = silent
        file_names.sort(key=filename_to_idx)
        self.noise_std = noise_std
        self.is_train = is_train
        self.size = len(self.file_names)
        self.output_normalized = output_normalized


    def __len__(self):
        'Denotes the total number of samples'
        return self.size

    def __getitem__(self, index):
        'Generates one sample of data'
        x = []
        y = []
        _file = self.file_names[index]
        if os.path.exists(_file):
        #for idx, file_name in enumerate(file_names):
            _file = np.load(_file)
            tx = _file["data_x"]
            ty = _file["data_y"]
            ############# normalization ###############
            #tx, ty = normalization(tx, ty)
            tx = normalize_x(tx)
            if self.output_normalized:
                ty = normalize_y(ty)
            tx = np.transpose(tx, (0, 2, 3, 1))
            ty = np.transpose(ty, (0, 2, 3, 1))
            tx = np.reshape(tx, (-1, tx.shape[-1]))
            ty = np.reshape(ty, (-1, ty.shape[-1]))
            x.append(tx)
            y.append(ty)
            if not self.silent:
                print(index, len(file_names), 'x-shape & y-shape:', tx.shape, ty.shape) # (1, 96, 144, 32) (1, 96, 144, 5)
        x = np.concatenate(x, axis=0).squeeze()
        y = np.concatenate(y, axis=0).squeeze()

        if self.is_train and self.noise_std>0:
            # print(self.noise_std)
            noise_x = np.random.randn(x.shape[0]) * self.noise_std
            noise_y = np.random.randn(y.shape[0]) * self.noise_std
            x = x + noise_x
            y = y + noise_y

        return x, y
class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, file_names, is_train, noise_std = 0):
        ### load the data ###
        x = []
        y = []
        file_names.sort(key=filename_to_idx)
        for idx, file_name in enumerate(file_names):
            _file = np.load(file_name)
            tx = _file["data_x"]
            ty = _file["data_y"]
            ############# normalization ###############
            tx, ty = normalization(tx, ty)
            tx = np.transpose(tx, (0, 2, 3, 1))
            ty = np.transpose(ty, (0, 2, 3, 1))
            tx = np.reshape(tx, (-1, tx.shape[-1]))
            ty = np.reshape(ty, (-1, ty.shape[-1]))
            x.append(tx)
            y.append(ty)
            if not self.silent:
                print(idx, len(file_names), 'x-shape & y-shape:', tx.shape, ty.shape) # (1, 96, 144, 32) (1, 96, 144, 5)
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
    import os
    data_dir = "/home/users/data/nncam_data/image_set/"
    file_names = os.listdir(data_dir)
    file_names = [data_dir + fn for fn in file_names][:10]
    training_set = TimeDataset(file_names, is_train=True, noise_std=0)
    trainloader = data.DataLoader(training_set, shuffle=False, batch_size=1, num_workers=4)
    for idx, batch in enumerate(trainloader):
        x, y = batch
        np.save('checkcode_x_time'+str(idx), x.numpy())
        np.save('checkcode_y_time'+str(idx), y.numpy())
        if idx == 1:
            break
        print(idx, x.size(), y.size())

    training_set = Dataset(file_names, is_train=True, noise_std=0)
    trainloader = data.DataLoader(training_set, shuffle=False, batch_size=1, num_workers=4)
    for idx, batch in enumerate(trainloader):
        x, y = batch
        np.save('checkcode_x'+str(idx), x.numpy())
        np.save('checkcode_y'+str(idx), y.numpy())
        if idx == 2:
            break
        print(idx, x.size(), y.size())
