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
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))
from normalization import normalize_data_var_names, inverse_data_var_names
from data_shape import to_inference_shape, inverse_to_inference_shape
from dataloader_utils import get_index_from_colnames, filename_to_idx, idx_to_filename, \
    gen_multistep_col_indices
from tqdm.autonotebook import tqdm

def region_slice2d(region_mask2d):
    mask, pad = region_mask2d
    min_x = np.min(np.where(mask)[0])
    max_x = np.max(np.where(mask)[0])
    min_y = np.min(np.where(mask)[1])
    max_y = np.max(np.where(mask)[1])
    return min_x-pad, max_x+pad, min_y-pad, max_y+pad

class PairDatasetDisk(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(
        self,
        file_names,
        curr_input_indices1,
        prev_input_indices1,
        output_indices1,
        curr_input_indices2,
        prev_input_indices2,
        output_indices2,
        is_train,
        noise_std = 0,
        transform1=None,
        transform2=None,
        silent=True,
        multistep=0,
        sample_rate=1,
        include_filename=False,
        region_mask1d=None,
        region_mask2d=None):
        ### load the data ###
        all_files = file_names[:]
        # self.data_std = data_std
        # self.data_mean = data_mean
        # if is_train:
        for i in range(17507,17530):
            for file_name in all_files:
                if str(i) in file_name:
                    all_files.remove(file_name)

        #print('after file num:', len(all_files))

        #for i in ['00001', '08690', '17522', '26210', '09242']:
        for i in ['00001', '08690', '17522', '26210']:
            for file_name in all_files:
                if i in file_name:
                    all_files.remove(file_name)
        for i in ['35042', '43730', '52562']:
            for file_name in all_files:
                if i in file_name:
                    all_files.remove(file_name)
        for i in range(55015,55056):
            for file_name in all_files:
                if str(i) in file_name:
                    all_files.remove(file_name)


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
        print(f"is_train: {is_train}, dataset size: {len(self.file_names)}")
        self.noise_std = noise_std
        self.is_train = is_train
        self.include_filename = include_filename
        self.size = len(self.file_names)
        pconsts = np.load(os.path.join(os.path.dirname(__file__), "..", "consts", "phys_consts.npz"))
        self.hyam = pconsts["hyam"]
        self.hybm = pconsts["hybm"]
        self.input_indices1 = curr_input_indices1
        self.input_indices2 = curr_input_indices2
        self.prev_input_indices1 = prev_input_indices1
        self.prev_input_indices2 = prev_input_indices2
        self.output_indices1 = output_indices1
        self.output_indices2 = output_indices2
        self.transform1 = transform1
        self.transform2 = transform2
        self.region_mask1d = region_mask1d
        self.region_mask2d = None
        if region_mask2d:
            self.region_mask2d = region_slice2d(region_mask2d)

    def __len__(self):
        'Denotes the total number of samples'
        return self.size

    def inverse_y(self):
        inverse = {}
        for yname in self.col_names_y:
            print(f"gen {yname} inverse")
            inverse[yname] = lambda y: inverse_data_var_names(y, [yname], \
                self.col_names, self.data_mean, self.data_std)
        return inverse


    def load_slice(self, filename, slice):
        data = np.load(filename, mmap_mode="r")
        if self.region_mask1d is not None:
            return np.array([data[i, self.region_mask1d.astype(bool)] for i in slice])
        if self.region_mask2d is not None:
            min_x, max_x, min_y, max_y = self.region_mask2d
            return data[slice, min_x:max_x, min_y:max_y]
        return data[slice]

    def load_data1(self, index):
        target_file = self.file_names[index]
        file_names = [target_file]
        if not os.path.exists(target_file):
            raise ValueError(f"File {target_file} does not exist")
        tx = self.load_slice(target_file, self.input_indices1)
        y = self.load_slice(target_file, self.output_indices1)

        tokens = target_file.split("/")
        target_fileidx = filename_to_idx(tokens[-1])
        prev_inputs = []
        for p in range(1,self.multistep+1):
            prev_file = "/".join(tokens[:-1]+[idx_to_filename(target_fileidx-p)])
            tx_prev = self.load_slice(prev_file, self.prev_input_indices1)
            prev_inputs.append(tx_prev)
            # prev_raws.append(tx_raw_prev)
            file_names.append(prev_file)

        x = np.concatenate([*prev_inputs, tx], axis=0)

        if self.is_train and self.noise_std>0:
            # print(self.noise_std)
            noise_x = np.random.randn(x.shape[0]) * self.noise_std
            noise_y = np.random.randn(y.shape[0]) * self.noise_std
            x = x + noise_x
            y = y + noise_y

        return x, y, file_names[::-1]

    def load_data2(self, index):
        target_file = self.file_names[index]
        file_names = [target_file]
        if not os.path.exists(target_file):
            raise ValueError(f"File {target_file} does not exist")
        tx = self.load_slice(target_file, self.input_indices2)
        y = self.load_slice(target_file, self.output_indices2)

        tokens = target_file.split("/")
        target_fileidx = filename_to_idx(tokens[-1])
        prev_inputs = []
        for p in range(1,self.multistep+1):
            prev_file = "/".join(tokens[:-1]+[idx_to_filename(target_fileidx-p)])
            tx_prev = self.load_slice(prev_file, self.prev_input_indices2)
            prev_inputs.append(tx_prev)
            # prev_raws.append(tx_raw_prev)
            file_names.append(prev_file)

        x = np.concatenate([*prev_inputs, tx], axis=0)

        if self.is_train and self.noise_std>0:
            # print(self.noise_std)
            noise_x = np.random.randn(x.shape[0]) * self.noise_std
            noise_y = np.random.randn(y.shape[0]) * self.noise_std
            x = x + noise_x
            y = y + noise_y

        return x, y, file_names[::-1]


    def __getitem__(self, index):
        'Generates one sample of data'
        x1, y1, file_names1 = self.load_data1(index)
        x2, y2, file_names2 = self.load_data2(index)
        assert(file_names1 == file_names2)
        file_names = file_names1
        sample1 = [x1, y1]
        sample2 = [x2, y2]
        if self.transform1 and self.transform2:
            sample1 = self.transform1(sample1)
            sample2 = self.transform2(sample2)
        if sample1 is None or sample2 is None:
            return None
        sample = [*sample1, *sample2]
        if self.include_filename:
            sample.append(file_names)
        return sample



class DatasetDisk(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(
        self,
        file_names,
        curr_input_indices,
        prev_input_indices,
        output_indices,
        is_train,
        noise_std = 0,
        transform=None,
        silent=True,
        multistep=0,
        sample_rate=1,
        include_filename=False,
        region_mask1d=None,
        region_mask2d=None):
        ### load the data ###
        all_files = file_names[:]
        # self.data_std = data_std
        # self.data_mean = data_mean
        # if is_train:
        for i in range(17507,17530):
            for file_name in all_files:
                if str(i) in file_name:
                    all_files.remove(file_name)

        #print('after file num:', len(all_files))

        #for i in ['00001', '08690', '17522', '26210', '09242']:
        for i in ['00001', '08690', '17522', '26210']:
            for file_name in all_files:
                if i in file_name:
                    all_files.remove(file_name)
        for i in ['35042', '43730', '52562']:
            for file_name in all_files:
                if i in file_name:
                    all_files.remove(file_name)
        for i in range(55015,55056):
            for file_name in all_files:
                if str(i) in file_name:
                    all_files.remove(file_name)


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
        print(f"is_train: {is_train}, dataset size: {len(self.file_names)}")
        self.noise_std = noise_std
        self.is_train = is_train
        self.include_filename = include_filename
        self.size = len(self.file_names)
        pconsts = np.load(os.path.join(os.path.dirname(__file__), "..", "consts", "phys_consts.npz"))
        self.hyam = pconsts["hyam"]
        self.hybm = pconsts["hybm"]
        self.input_indices = curr_input_indices
        self.prev_input_indices = prev_input_indices
        self.output_indices = output_indices
        self.transform = transform
        self.region_mask1d = region_mask1d
        self.region_mask2d = None
        if region_mask2d:
            self.region_mask2d = region_slice2d(region_mask2d)

    def __len__(self):
        'Denotes the total number of samples'
        return self.size

    def inverse_y(self):
        inverse = {}
        for yname in self.col_names_y:
            print(f"gen {yname} inverse")
            inverse[yname] = lambda y: inverse_data_var_names(y, [yname], \
                self.col_names, self.data_mean, self.data_std)
        return inverse


    def load_slice(self, filename, slice):
        data = np.load(filename, mmap_mode="r")
        if self.region_mask1d is not None:
            return np.array([data[i, self.region_mask1d.astype(bool)] for i in slice])
        if self.region_mask2d is not None:
            min_x, max_x, min_y, max_y = self.region_mask2d
            return data[slice, min_x:max_x, min_y:max_y]
        return data[slice]

    def load_data(self, index):
        target_file = self.file_names[index]
        file_names = [target_file]
        if not os.path.exists(target_file):
            raise ValueError(f"File {target_file} does not exist")
        tx = self.load_slice(target_file, self.input_indices)
        y = self.load_slice(target_file, self.output_indices)

        tokens = target_file.split("/")
        target_fileidx = filename_to_idx(tokens[-1])
        prev_inputs = []
        for p in range(1,self.multistep+1):
            prev_file = "/".join(tokens[:-1]+[idx_to_filename(target_fileidx-p)])
            tx_prev = self.load_slice(prev_file, self.prev_input_indices)
            prev_inputs.append(tx_prev)
            # prev_raws.append(tx_raw_prev)
            file_names.append(prev_file)

        x = np.concatenate([*prev_inputs, tx], axis=0)

        if self.is_train and self.noise_std>0:
            # print(self.noise_std)
            noise_x = np.random.randn(x.shape[0]) * self.noise_std
            noise_y = np.random.randn(y.shape[0]) * self.noise_std
            x = x + noise_x
            y = y + noise_y

        return x, y, file_names[::-1]


    def __getitem__(self, index):
        'Generates one sample of data'
        x, y, file_names = self.load_data(index)
        sample = [x, y]
        if self.include_filename:
            sample.append(file_names)
        if self.transform:
            sample = self.transform(sample)
        return sample



def filter_collate(batch):
    batch = list(filter (lambda x:x is not None, batch))
    if not batch:
        return None
    return default_collate(batch)


if __name__ == '__main__':
    import os
    import glob
    import argparse
    from preprocess import FlattenSpatialTransform, StandardizeTransform
    # import torch transforms
    import torchvision.transforms as transforms
    parser = argparse.ArgumentParser()
    parser.add_argument("--ex_input", type=str, nargs="*", default=[])
    args = parser.parse_args()
    print(args)
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
    # file_names = file_names[:10]
    # file_names = [data_dir + fn for fn in file_names]
    print("file_names:", file_names[:10])
    col_names = np.loadtxt(data_dir + "/col_names.txt", dtype=str)
    col_names_x = ["QL", "T_nn_in", "dqvls_nn_in", "dTls_nn_in", "SOLIN", "SPPS"]
    col_names_y = ["qtend_check"]
    # varaibles that are used as input in the previous time step
    prev_ex_vars = ["qtend_check", "stend_check", "SOLL", "SOLLD", "SOLS", "SOLSD", "FSDS"]

    data_means = dict(np.load(data_dir + "/data_means.npz"))
    data_stds = dict(np.load(data_dir + "/data_stds.npz"))

    input_indices, prev_input_indices, output_indices = gen_multistep_col_indices(
        col_names, prev_ex_vars, col_names_x, col_names_y, 1)
    print("input_indices:", col_names[input_indices])
    print("prev_input_indices:", col_names[prev_input_indices])
    print("output_indices:", col_names[output_indices])

    transform = transforms.Compose([
        StandardizeTransform(
            data_means,
            data_stds,
            col_names_x,
            col_names_y,
            col_names,
            normalize_input=True,
            normalize_output=True
            ),
        # FlattenSpatialTransform()
        ])

    region_mask = np.load(os.path.join(os.path.dirname(__file__), "..", "consts", "pacific_region_mask.npy"))

    training_set = DatasetDisk(
        file_names,
        input_indices,
        prev_input_indices,
        output_indices,
        multistep=0,
        sample_rate=12,
        is_train=True,
        transform=transform,
        include_filename=True,
        region_mask1d=region_mask
        )
    trainloader = data.DataLoader(training_set, shuffle=False, batch_size=1, num_workers=1)
    dqvls_norm = []
    dqvls = []
    # start_idx, end_idx = get_index_from_colnames(col_names, "dqvls_nn_in")
    for idx, batch in enumerate(trainloader):
        x, y, filenames = batch
        print(idx, x.size(), y.size(), filenames)
        # dqvls_norm.append(x[:,:,60:90].numpy())
        # dqvls.append(x_raw[:,60:90].numpy())
    # np.save("dqvls_norm.npy", np.concatenate(dqvls_norm, axis=0))
    # np.save("dqvls_raw.npy", np.concatenate(dqvls, axis=0))
        # np.save('checkcode_x_new'+str(idx), x.numpy())
        #np.save('checkcodes/checkcode_x'+str(idx), x_raw.numpy())
        #np.save('checkcodes/checkcode_y'+str(idx), y.numpy())
    # training_set = DatasetDisk(file_names, col_names, col_names_x, col_names_y, data_stds, data_means, is_train=True, noise_std=0, filename=True, output_normalized=False, multistep=1, prev_ex_vars=prev_ex_vars)
    # trainloader = data.DataLoader(training_set, shuffle=False, batch_size=1, num_workers=1, collate_fn=filter_collate)
    # for idx, batch in enumerate(trainloader):
    #     x, y, x_raw, filenames = batch
    #     print(idx, x.size(), y.size(), x_raw.size(), filenames)
    #     np.save('checkcode_x_new_ts1_'+str(idx), x.numpy())
    #     np.save('checkcode_x_new_ts1_raw_'+str(idx), x_raw.numpy())
    #     np.save('checkcode_y_new_ts1_raw_'+str(idx), y.numpy())
    # training_set = DatasetDisk(file_names, col_names, col_names_x, col_names_y, data_stds, data_means, is_train=True, noise_std=0, filename=True, output_normalized=True, multistep=1, prev_ex_vars=prev_ex_vars)
    # trainloader = data.DataLoader(training_set, shuffle=False, batch_size=1, num_workers=1, collate_fn=filter_collate)
    # for idx, batch in enumerate(trainloader):
    #     x, y, x_raw, filenames = batch
    #     print(idx, x.size(), y.size(), x_raw.size(), filenames)
    #     # np.save('checkcode_x_new_ts'+str(idx), x.numpy())
    #     # np.save('checkcode_x_new_ts1_raw'+str(idx), x_raw.numpy())
    #     np.save('checkcode_y_new_ts1_'+str(idx), y.numpy())
    # var_names = []
    # pattern = f"^(.*?)(?=_lev\d+|$)"
    # for col_name in col_names:
    #     match = re.match(pattern, col_name)
    #     if match and match.group(1) not in var_names:
    #         var_names.append(match.group(1))
    #col_names_x = var_names
    # col_names_x = ["QL"]
    # col_names_y = ["FSDS"]
    # training_set = DatasetDisk(file_names, col_names, col_names_x, col_names_y, data_stds, data_means, is_train=True, noise_std=0, filename=True, output_normalized=True, multistep=1, image=True, prev_ex_vars=args.ex_input)
    # trainloader = data.DataLoader(training_set, shuffle=False, batch_size=1, num_workers=1, collate_fn=filter_collate)
    # for idx, batch in enumerate(trainloader):
    #     x, y, x_raw, filenames = batch
    #     print(idx, x.size(), y.size(), x_raw.size(), filenames)
    #     np.save('checkcode_x_new_image'+str(idx), x.numpy())
    #     # np.save('checkcode_x_new_image'+str(idx), x_raw.numpy())
    #     np.save('checkcode_y_new_image'+str(idx), y.numpy())
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
