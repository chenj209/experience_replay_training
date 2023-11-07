from torch.utils import data
import os
import numpy as np
import time
import glob
import re
from rh import get_pmid_from_x, cal_rh
from normalization import normalize_x, normalize_y

def idx_to_filename(idx):
  return str(idx).rjust(5,'0') + '.npz'

def filename_to_idx(filename):
    data_pattern = ".*(\d{5})\.npz"
    m = re.search(data_pattern, filename)
    if m is None:
        return -1
    else:
        return int(m.group(1))

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
        pconsts = np.load(os.path.dirname(os.path.abspath(__file__))+"/phys_consts.npz")
        self.hyam = pconsts["hyam"]
        self.hybm = pconsts["hybm"]

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
            tx = _file["data_x"]
            ty = _file["data_y"]
            ############# normalization ###############
            #tx, ty = normalization(tx, ty)
            tx = normalize_x(tx_raw) 
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
            # if not self.silent:
            #     print(index, len(self.file_names), 'x-shape & y-shape:', tx.shape, ty.shape) # (1, 96, 144, 32) (1, 96, 144, 5)
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

# if __name__ == '__main__':
#     import os
#     data_dir = "/home/users/data/nncam_data/image_set/"
#     file_names = os.listdir(data_dir)
#     file_names = [data_dir + fn for fn in file_names][:10]
#     training_set = TimeDataset(file_names, is_train=True, noise_std=0)
#     trainloader = data.DataLoader(training_set, shuffle=False, batch_size=1, num_workers=4)
#     for idx, batch in enumerate(trainloader):
#         x, y = batch
#         np.save('checkcode_x_time'+str(idx), x.numpy())
#         np.save('checkcode_y_time'+str(idx), y.numpy())
#         if idx == 1:
#             break
#         print(idx, x.size(), y.size())

#     training_set = Dataset(file_names, is_train=True, noise_std=0)
#     trainloader = data.DataLoader(training_set, shuffle=False, batch_size=1, num_workers=4)
#     for idx, batch in enumerate(trainloader):
#         x, y = batch
#         np.save('checkcode_x'+str(idx), x.numpy())
#         np.save('checkcode_y'+str(idx), y.numpy())
#         if idx == 2:
#             break
#         print(idx, x.size(), y.size())
