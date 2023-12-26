from torch.utils import data
import sys
import os
import numpy as np
import time
import glob
import re
#sys.path.append(
#from rh import get_pmid_from_x, cal_rh
sys.path.append(os.path.join(sys.path[0], "..", "utils"))
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

import re
def col_name_cmp(var_name, col_name):
    pattern = f"^{var_name}(_lev\d+)?$"
    # print(pattern)
    return re.match(pattern, col_name) is not None

def get_index_from_colnames(col_names, var_name):
    start_idx = -1
    end_idx = -1
    for i, col_name in enumerate(col_names):
        if col_name_cmp(var_name, col_name) and start_idx == -1:
            start_idx = i
        if col_name_cmp(var_name, col_name):
            end_idx = i
        if end_idx != -1 and not col_name_cmp(var_name, col_name):
            return start_idx, end_idx+1
    return start_idx, end_idx+1

class DatasetDisk(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(
        self,
        file_names,
        col_names, # list of all column names in the dataset
        col_names_x, # list of column names for input
        col_names_y, # list of column names for output
        is_train,
        noise_std = 0,
        output_normalized=True,
        silent=True,
        filename=False):
        ### load the data ###
        all_files = file_names
        if is_train:
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
        self.file_name = filename
        self.size = len(self.file_names)
        self.output_normalized = output_normalized
        pconsts = np.load(os.path.join(sys.path[0], "..", "consts", "phys_consts.npz"))
        self.hyam = pconsts["hyam"]
        self.hybm = pconsts["hybm"]
        self.col_names = col_names
        self.col_names_x = col_names_x
        self.col_names_y = col_names_y
        input_indices = []
        for cn in col_names_x:
            start_idx, end_idx = get_index_from_colnames(col_names, cn)
            print(cn, start_idx, end_idx)
            input_indices.extend(list(range(start_idx, end_idx)))
        output_indices = []
        for cn in col_names_y:
            print(cn, start_idx, end_idx)
            start_idx, end_idx = get_index_from_colnames(col_names, cn)
            output_indices.extend(list(range(start_idx, end_idx)))
        self.input_indices = input_indices
        self.output_indices = output_indices

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
            data = np.load(_file)
            print(_file, data.shape)
            tx_raw = data[self.input_indices,:,:][None]
            tx = data[self.input_indices,:,:][None]
            ty = data[self.output_indices,:,:][None]
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

        if self.file_name:
            return x, y, x_raw, self.file_names[index]

        return x, y, x_raw

if __name__ == '__main__':
    import os
    import glob
    data_dir = "/home/users/data/nncam_data/image_set/"
    if not os.path.isdir(data_dir):
        data_dir = "/data/nncam_data/image_set/"
    if not os.path.isdir(data_dir):
        # data_dir = "./data/"
        data_dir = "/Users/jiandachen/Projects/NNCAM_packages/precip_pattern/test_data/"
    if not os.path.isdir(data_dir):
        # data_dir = "./data/"
        data_dir = "/pscratch/sd/c/chenjd21/spcam_new_data/"
    file_names = glob.glob(data_dir + "*.npy")[35041:]
    file_names.sort()
    file_names = file_names[:10]
    # file_names = [data_dir + fn for fn in file_names]
    print(file_names[:10])
    col_names = np.loadtxt("/pscratch/sd/c/chenjd21/spcam_new_data/col_names.txt", dtype=str)
    col_names_x = ["QL", "T_nn_in", "dqvls_nn_in", "dTls_nn_in", "SOLIN", "SPPS"]
    col_names_y = ["qtend_check", "stend_check", "SOLL", "SOLLD", "SOLS", "SOLSD", "FSDS"]
    training_set = DatasetDisk(file_names, col_names, col_names_x, col_names_y, is_train=True, noise_std=0, filename=True, output_normalized=False)
    trainloader = data.DataLoader(training_set, shuffle=False, batch_size=1, num_workers=1)
    for idx, batch in enumerate(trainloader):
        x, y, x_raw, filenames = batch
        print(idx, x.size(), y.size(), x_raw.size(), filenames)
        # np.save('checkcode_x_new'+str(idx), x.numpy())
        np.save('checkcode_x_new'+str(idx), x_raw.numpy())
        np.save('checkcode_y_new'+str(idx), y.numpy())
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
