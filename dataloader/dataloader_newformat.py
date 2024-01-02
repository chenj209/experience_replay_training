from torch.utils import data
import sys
import os
import numpy as np
import time
import glob
import re
from torch.utils.data.dataloader import default_collate
#sys.path.append(
#from rh import get_pmid_from_x, cal_rh
sys.path.append(os.path.join(sys.path[0], "..", "utils"))
from normalization import normalize_x, normalize_y, normalization
from data_shape import to_inference_shape, inverse_to_inference_shape

def idx_to_filename(idx):
  return str(idx).rjust(5,'0') + '.npy'

def filename_to_idx(filename):
    data_pattern = ".*(\d{5})\.npy"
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
        input_normalized=True,
        output_normalized=True,
        silent=True,
        filename=False,
        multistep=0,
        sample_rate=1,
        image=False):
        ### load the data ###
        all_files = file_names[:]
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
        self.silent = silent
        self.multistep = multistep
        self.all_files = all_files[:]
        self.all_files.sort(key=filename_to_idx)
        self.file_names = []
        if self.multistep > 0:
            for file_name in self.all_files[::sample_rate]:
                cur_idx = filename_to_idx(file_name)
                missing_flag = False
                for p in range(1, self.multistep+1):
                    prev_idx = cur_idx - p
                    tokens = file_name.split("/")
                    prev_file_name = "/".join(tokens[:-1]+[idx_to_filename(prev_idx)])
                    if prev_file_name not in self.all_files:
                        print(f"Missing {prev_file_name} for {file_name}")
                        missing_flag = True
                        break
                if not missing_flag:
                    self.file_names.append(file_name)
        else:
            self.file_names = self.all_files[::sample_rate]
        self.noise_std = noise_std
        self.is_train = is_train
        self.file_name = filename
        self.size = len(self.file_names)
        self.output_normalized = output_normalized
        self.input_normalized = input_normalized
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
        self.image = image # if True, the shape would be (CxHxW)

    def __len__(self):
        'Denotes the total number of samples'
        return self.size
    
    @staticmethod
    def get_xy_from_file(file_name, input_indices, output_indices, input_normalized=True, output_normalized=True, image=False):
        data = np.load(file_name)
        tx_raw = data[input_indices,:,:][None,]
        tx = data[input_indices,:,:][None,]
        ty = data[output_indices,:,:][None,]

        ############# normalization ###############
        #tx, ty = normalization(tx, ty)
        if input_normalized:
            tx = normalize_x(tx)
        if output_normalized:
            ty = normalize_y(ty)
        if not image:
            tx = to_inference_shape(tx)
            ty = to_inference_shape(ty)
            tx_raw = to_inference_shape(tx_raw)
        # tx = np.transpose(tx, (0, 2, 3, 1))
        # ty = np.transpose(ty, (0, 2, 3, 1))
        # tx_raw = np.transpose(tx_raw, (0, 2, 3, 1))
        # tx = np.reshape(tx, (-1, tx.shape[-1]))
        # ty = np.reshape(ty, (-1, ty.shape[-1]))
        # tx_raw = np.reshape(tx_raw, (-1, tx_raw.shape[-1]))
        return tx.squeeze(), ty.squeeze(), tx_raw.squeeze()

    def __getitem__(self, index):
        'Generates one sample of data'
        x = []
        y = []
        x_raw = []
        target_file = self.file_names[index]
        file_names = [target_file]
        if os.path.exists(target_file):
        #for idx, file_name in enumerate(file_names):
            tx, y, tx_raw = self.get_xy_from_file(target_file, self.input_indices, self.output_indices, self.input_normalized, self.output_normalized, self.image)

            tokens = target_file.split("/")
            # curr_tidx = filename_to_idx(target_file)
            target_fileidx = filename_to_idx(tokens[-1])
            prev_inputs = []
            prev_raws = []
            for p in range(1,self.multistep+1):
                prev_file = "/".join(tokens[:-1]+[idx_to_filename(target_fileidx-p)])
                tx_prev, ty_prev, tx_raw = self.get_xy_from_file(prev_file, self.input_indices, self.output_indices, input_normalized=self.input_normalized, output_normalized=True, image=self.image)
                prev_inputs.extend([tx_prev, ty_prev])
                prev_raws.append(tx_raw)
                file_names.append(prev_file)
            print(tx.shape, y.shape, tx_raw.shape)

            channel_axis = 1
            if self.image:
                channel_axis = 0
            x = np.concatenate([*prev_inputs, tx], axis=channel_axis)
            x_raw = np.concatenate([*prev_raws, tx_raw], axis=channel_axis)

            if self.is_train and self.noise_std>0:
                # print(self.noise_std)
                noise_x = np.random.randn(x.shape[0]) * self.noise_std
                noise_y = np.random.randn(y.shape[0]) * self.noise_std
                x = x + noise_x
                y = y + noise_y

            if self.file_name:
                return x, y, x_raw, file_names[::-1]

            return x, y, x_raw
        return None

def filter_collate(batch):
    batch = list(filter (lambda x:x is not None, batch))
    return default_collate(batch)


if __name__ == '__main__':
    import os
    import glob
    data_dir = "/home/users/data/nncam_data/image_set/"
    if not os.path.isdir(data_dir):
        data_dir = "/data/nncam_data/image_set/"
    if not os.path.isdir(data_dir):
        # data_dir = "./data/"
        data_dir = "../analysis/test_data/"
    if not os.path.isdir(data_dir):
        # data_dir = "./data/"
        data_dir = "/pscratch/sd/c/chenjd21/spcam_new_data/"
    file_names = glob.glob(data_dir + "*.npy")
    file_names.sort()
    file_names = file_names[:10]
    # file_names = [data_dir + fn for fn in file_names]
    print("file_names:", file_names[:10])
    col_names = np.loadtxt(data_dir + "/col_names.txt", dtype=str)
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
    training_set = DatasetDisk(file_names, col_names, col_names_x, col_names_y, is_train=True, noise_std=0, filename=True, output_normalized=False, multistep=1)
    trainloader = data.DataLoader(training_set, shuffle=False, batch_size=1, num_workers=1, collate_fn=filter_collate)
    for idx, batch in enumerate(trainloader):
        x, y, x_raw, filenames = batch
        print(idx, x.size(), y.size(), x_raw.size(), filenames)
        # np.save('checkcode_x_new'+str(idx), x.numpy())
        np.save('checkcode_x_new_ts1'+str(idx), x_raw.numpy())
        np.save('checkcode_y_new_ts1'+str(idx), y.numpy())
    col_names_x = col_names[:]
    training_set = DatasetDisk(file_names, col_names, col_names_x, col_names_y, is_train=True, noise_std=0, filename=True, output_normalized=False, multistep=1, image=True)
    trainloader = data.DataLoader(training_set, shuffle=False, batch_size=1, num_workers=1, collate_fn=filter_collate)
    for idx, batch in enumerate(trainloader):
        x, y, x_raw, filenames = batch
        print(idx, x.size(), y.size(), x_raw.size(), filenames)
        # np.save('checkcode_x_new'+str(idx), x.numpy())
        np.save('checkcode_x_new_image'+str(idx), x_raw.numpy())
        np.save('checkcode_y_new_image'+str(idx), y.numpy())
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
