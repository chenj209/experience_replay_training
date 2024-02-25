import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "utils"))
import os
import glob
import argparse
import numpy as np
import torchvision.transforms as transforms
from torch.utils import data

from preprocess import FlattenSpatialTransform, StandardizeTransform, RegionMaskTransform, \
    RectRegionMaskTransform
from dataloader_utils import gen_multistep_col_indices, get_index_from_colnames
from dataloader_newformat import DatasetDisk
from logger import debug_print
# import torch transforms

DEBUG = True

data_dir = "/home/users/data/nncam_data/image_set/"
if not os.path.isdir(data_dir):
    data_dir = "/data/nncam_data/image_set/"
if not os.path.isdir(data_dir):
    # data_dir = "./data/"
    data_dir = "../analysis/test_data/"
if not os.path.isdir(data_dir):
    # data_dir = "./data/"
    data_dir = "/pscratch/sd/c/chenjd21/spcam_new_data/"

region_mask = np.load(os.path.join(os.path.dirname(__file__), "..", "..", \
                                    "consts", "pacific_region_mask.npy"))
assert region_mask.shape == (96,144)

def test_single_column_multistep0_pacific_region():
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
        RegionMaskTransform(region_mask, include_raw=True),
        StandardizeTransform(
            data_means,
            data_stds,
            col_names_x,
            col_names_y,
            col_names,
            normalize_input=True,
            normalize_output=True
            ),
        FlattenSpatialTransform()
        ])

    training_set = DatasetDisk(
        file_names,
        input_indices,
        prev_input_indices,
        output_indices,
        multistep=0,
        sample_rate=12,
        is_train=True,
        transform=transform,
        include_filename=True
        )
    trainloader = data.DataLoader(training_set, shuffle=False, batch_size=1, num_workers=1)
    norm_data_x = []
    norm_data_y = []
    raw_data_x = []
    # start_idx, end_idx = get_index_from_colnames(col_names, "dqvls_nn_in")
    for idx, batch in enumerate(trainloader):
        x, y, x_raw_region, filenames = batch
        print(idx, x.size(), y.size(), x_raw_region.size(), filenames)
        norm_data_x.append(x.numpy())
        norm_data_y.append(x.numpy())
        raw_data_x.append(x_raw_region.numpy())
    norm_data_x = np.concatenate(norm_data_x, axis=0)
    norm_data_y = np.concatenate(norm_data_y, axis=0)
    raw_data_x = np.concatenate(raw_data_x, axis=0)
    print("norm_data_x shape: ", norm_data_x.shape)
    print("raw_data_x shape: ", raw_data_x.shape)
    cur_idx = 0
    for col in col_names_x:
        start_idx, end_idx = get_index_from_colnames(col_names, col)
        data_range = end_idx - start_idx
        print(col, norm_data_x[:,:,cur_idx:cur_idx+data_range].mean(), \
              norm_data_x[:,:,cur_idx:cur_idx+data_range].std())
        print("raw: ", col, raw_data_x[:,cur_idx:cur_idx+data_range].mean(), \
              raw_data_x[:,cur_idx:cur_idx+data_range].std())
        cur_idx += data_range
    print("qtend_check: ", norm_data_y.mean(), norm_data_y.std())

def test_single_column_multistep0_pacific_region_rect():
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
        RectRegionMaskTransform(region_mask, include_raw=True),
        StandardizeTransform(
            data_means,
            data_stds,
            col_names_x,
            col_names_y,
            col_names,
            normalize_input=True,
            normalize_output=True
            ),
        FlattenSpatialTransform()
        ])

    training_set = DatasetDisk(
        file_names,
        input_indices,
        prev_input_indices,
        output_indices,
        multistep=0,
        sample_rate=12,
        is_train=True,
        transform=transform,
        include_filename=True
        )
    trainloader = data.DataLoader(training_set, shuffle=False, batch_size=1, num_workers=1)
    norm_data_x = []
    norm_data_y = []
    raw_data_x = []
    # start_idx, end_idx = get_index_from_colnames(col_names, "dqvls_nn_in")
    for idx, batch in enumerate(trainloader):
        debug_print(f"batch elements: {len(batch)}", DEBUG)
        x, y, x_raw_region, filenames = batch
        print(idx, x.size(), y.size(), x_raw_region.size(), filenames)
        norm_data_x.append(x.numpy())
        norm_data_y.append(x.numpy())
        raw_data_x.append(x_raw_region.numpy())
    norm_data_x = np.concatenate(norm_data_x, axis=0)
    norm_data_y = np.concatenate(norm_data_y, axis=0)
    raw_data_x = np.concatenate(raw_data_x, axis=0)
    print("norm_data_x shape: ", norm_data_x.shape)
    print("raw_data_x shape: ", raw_data_x.shape)
    cur_idx = 0
    for col in col_names_x:
        start_idx, end_idx = get_index_from_colnames(col_names, col)
        data_range = end_idx - start_idx
        print(col, norm_data_x[:,:,cur_idx:cur_idx+data_range].mean(), \
              norm_data_x[:,:,cur_idx:cur_idx+data_range].std())
        print("raw: ", col, raw_data_x[:,cur_idx:cur_idx+data_range].mean(), \
              raw_data_x[:,cur_idx:cur_idx+data_range].std())
        cur_idx += data_range
    print("qtend_check: ", norm_data_y.mean(), norm_data_y.std())
    

def test_image_multistep0_pacific_region():
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
        RectRegionMaskTransform(region_mask, include_raw=True)
        ])

    training_set = DatasetDisk(
        file_names,
        input_indices,
        prev_input_indices,
        output_indices,
        multistep=0,
        sample_rate=12,
        is_train=True,
        transform=transform,
        include_filename=True
        )
    trainloader = data.DataLoader(training_set, shuffle=False, batch_size=1, num_workers=1)
    norm_data_x = []
    norm_data_y = []
    # start_idx, end_idx = get_index_from_colnames(col_names, "dqvls_nn_in")
    for idx, batch in enumerate(trainloader):
        x, y, x_raw_region, filenames = batch
        print(idx, x.size(), y.size(), x_raw_region.size(), filenames)
        norm_data_x.append(x.numpy())
        norm_data_y.append(x.numpy())
    norm_data_x = np.concatenate(norm_data_x, axis=0)
    norm_data_y = np.concatenate(norm_data_y, axis=0)
    print("Q: ", norm_data_x[:,:30].mean(), norm_data_x[:,:30].std())
    print("T: ", norm_data_x[:,30:60].mean(), norm_data_x[:,30:60].std())
    print("dqvls: ", norm_data_x[:,60:90].mean(), norm_data_x[:,60:90].std())
    print("dTls: ", norm_data_x[:,90:120].mean(), norm_data_x[:,90:120].std())
    print("SOLIN: ", norm_data_x[:,120].mean(), norm_data_x[:,120].std())
    print("SPPS: ", norm_data_x[:,121].mean(), norm_data_x[:,121].std())
    print("qtend_check: ", norm_data_y.mean(), norm_data_y.std())

def test_single_column_multistep1_pacific_region():
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
    # col_names_x = col_names_x + prev_ex_vars + col_names_x

    input_indices, prev_input_indices, output_indices = gen_multistep_col_indices(
        col_names, prev_ex_vars, col_names_x, col_names_y, 1)
    print("input_indices:", col_names[input_indices])
    print("prev_input_indices:", col_names[prev_input_indices])
    print("output_indices:", col_names[output_indices])

    transform = transforms.Compose([
        RegionMaskTransform(region_mask, include_raw=True),
        StandardizeTransform(
            data_means,
            data_stds,
            col_names_x + prev_ex_vars + col_names_x,
            col_names_y,
            col_names,
            normalize_input=True,
            normalize_output=True
            ),
        FlattenSpatialTransform()
        ])

    training_set = DatasetDisk(
        file_names,
        input_indices,
        prev_input_indices,
        output_indices,
        multistep=1,
        sample_rate=12,
        is_train=True,
        transform=transform,
        include_filename=True
        )
    trainloader = data.DataLoader(training_set, shuffle=False, batch_size=1, num_workers=1)
    norm_data_x = []
    norm_data_y = []
    # start_idx, end_idx = get_index_from_colnames(col_names, "dqvls_nn_in")
    for idx, batch in enumerate(trainloader):
        x, y, x_raw_region, filenames = batch
        print(idx, x.size(), y.size(), x_raw_region.size(), filenames)
        norm_data_x.append(x.numpy())
        norm_data_y.append(x.numpy())
    norm_data_x = np.concatenate(norm_data_x, axis=0)
    norm_data_y = np.concatenate(norm_data_y, axis=0)
    cur_idx = 0
    for col in col_names_x + prev_ex_vars + col_names_x:
        start_idx, end_idx = get_index_from_colnames(col_names, col)
        data_range = end_idx - start_idx
        print(col, norm_data_x[:,:,cur_idx:cur_idx+data_range].mean(), \
              norm_data_x[:,:,cur_idx:cur_idx+data_range].std())
        cur_idx += data_range
    print("qtend_check: ", norm_data_y.mean(), norm_data_y.std())

def test_image_multistep1_pacific_region():
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
    # col_names_x = col_names_x + prev_ex_vars + col_names_x

    input_indices, prev_input_indices, output_indices = gen_multistep_col_indices(
        col_names, prev_ex_vars, col_names_x, col_names_y, 1)
    print("input_indices:", col_names[input_indices])
    print("prev_input_indices:", col_names[prev_input_indices])
    print("output_indices:", col_names[output_indices])

    transform = transforms.Compose([
        RectRegionMaskTransform(region_mask, include_raw=True),
        StandardizeTransform(
            data_means,
            data_stds,
            col_names_x + prev_ex_vars + col_names_x,
            col_names_y,
            col_names,
            normalize_input=True,
            normalize_output=True
            ),
        ])

    training_set = DatasetDisk(
        file_names,
        input_indices,
        prev_input_indices,
        output_indices,
        multistep=1,
        sample_rate=12,
        is_train=True,
        transform=transform,
        include_filename=True
        )
    trainloader = data.DataLoader(training_set, shuffle=False, batch_size=1, num_workers=1)
    norm_data_x = []
    norm_data_y = []
    # start_idx, end_idx = get_index_from_colnames(col_names, "dqvls_nn_in")
    for idx, batch in enumerate(trainloader):
        x, y, x_raw_region, filenames = batch
        print(idx, x.size(), y.size(), x_raw_region.size(), filenames)
        norm_data_x.append(x.numpy())
        norm_data_y.append(x.numpy())
    norm_data_x = np.concatenate(norm_data_x, axis=0)
    norm_data_y = np.concatenate(norm_data_y, axis=0)
    cur_idx = 0
    for col in col_names_x + prev_ex_vars + col_names_x:
        start_idx, end_idx = get_index_from_colnames(col_names, col)
        data_range = end_idx - start_idx
        print(col, norm_data_x[:,cur_idx:cur_idx+data_range].mean(), \
              norm_data_x[:,cur_idx:cur_idx+data_range].std())
        cur_idx += data_range
    print("qtend_check: ", norm_data_y.mean(), norm_data_y.std())

if __name__ == "__main__":
    # test_single_column_multistep0_pacific_region()
    test_single_column_multistep0_pacific_region_rect()
    # test_single_column_multistep1_pacific_region()
    # test_image_multistep0_pacific_region()
    test_image_multistep1_pacific_region()

