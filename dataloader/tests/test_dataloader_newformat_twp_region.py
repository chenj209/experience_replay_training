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
from dataloader_utils import gen_multistep_col_indices, get_index_from_colnames, levelwise_variable
from dataloader_newformat import DatasetDisk, filter_collate
from logger import debug_print
from debug_utils import print_mean_std_by_var
# import torch transforms

DEBUG = True

data_dir = "/home/users/data/nncam_data/image_set/"
if not os.path.isdir(data_dir):
    # data_dir = "./data/"
    data_dir = "/pscratch/sd/c/chenjd21/spcam_new_data_32/"
if not os.path.isdir(data_dir):
    data_dir = "/data/nncam_data/image_set/"
if not os.path.isdir(data_dir):
    # data_dir = "./data/"
    data_dir = "../analysis/test_data/"
print("data_dir:", data_dir)

#region_mask = np.load(os.path.join(os.path.dirname(__file__), "..", "..", \
#                                    "consts", "pacific_region_mask.npy"))
region_mask = np.load(os.path.join(os.path.dirname(__file__), "..", "..", \
                                    "consts", "twp_region.npy"))
assert region_mask.shape == (96,144)

def test_single_column_multistep0_pacific_region():
    file_names = glob.glob(data_dir + "*.npy")
    file_names.sort()
    file_names = file_names[:35040]
    # file_names = file_names[:10]
    # file_names = [data_dir + fn for fn in file_names]
    print("file_names:", file_names[:10])
    col_names = np.loadtxt(data_dir + "/col_names.txt", dtype=str)
    col_names_x = ["QL", "T_nn_in", "dqvls_nn_in", "dTls_nn_in", "SOLIN", "SPPS"]
    col_names_y = ["qtend_check"]
    # varaibles that are used as input in the previous time step
    prev_ex_vars = ["qtend_check", "stend_check", "SOLL", "SOLLD", "SOLS", "SOLSD", "FSDS"]

    # data_means = dict(np.load(data_dir + "/data_means.npz"))
    # data_stds = dict(np.load(data_dir + "/data_stds.npz"))

    data_means = dict(np.load("../consts/twp_region_means.npz"))
    # data_means_by_lvl = dict(np.load(data_dir + "/std_mean_by_level_means.npz"))
    # data_means.update(data_means_by_lvl)
    data_stds = dict(np.load("../consts/twp_region_stds.npz"))
    # data_stds_by_lvl = dict(np.load(data_dir + "/std_mean_by_level_stds.npz"))
    # data_stds.update(data_stds_by_lvl)

    # use the same mean and std for variables except for q related
    for k in data_means:
        if "_lev" in k and \
            ("QL" not in k and "qtend" not in k and "dqvls" not in k):
            debug_print(f"Replacing means {k}({data_means[k]}) with\
                        {k.rstrip('_lev')}({data_means[k.split('_lev')[0]]})", DEBUG)
            data_means[k] = data_means[k.split("_lev")[0]]
            debug_print(f"Replacing stds {k}({data_stds[k]}) with\
                        {k.rstrip('_lev')}({data_stds[k.split('_lev')[0]]})", DEBUG)
            data_stds[k] = data_stds[k.split("_lev")[0]]

    input_indices, prev_input_indices, output_indices = gen_multistep_col_indices(
        col_names, prev_ex_vars, col_names_x, col_names_y, 1)
    print("input_indices:", col_names[input_indices])
    print("prev_input_indices:", col_names[prev_input_indices])
    print("output_indices:", col_names[output_indices])

    transform = transforms.Compose([
        # RegionMaskTransform(region_mask, include_raw=True),
        StandardizeTransform(
            data_means,
            data_stds,
            col_names_x,
            col_names_y,
            col_names,
            normalize_input=True,
            normalize_output=True,
            include_raw=True,
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
        include_filename=True,
        region_mask1d=region_mask
        )
    trainloader = data.DataLoader(
        training_set, shuffle=False, batch_size=1,
        num_workers=1, collate_fn=filter_collate)

    norm_data_x = []
    norm_data_y = []
    raw_data_x = []
    raw_data_y = []
    # start_idx, end_idx = get_index_from_colnames(col_names, "dqvls_nn_in")
    for idx, batch in enumerate(trainloader):
        if batch is None:
            continue
        x, y = batch[:2]
        x, y = x.reshape(-1, x.shape[-1]), y.reshape(-1, y.shape[-1])
        filenames = batch[-1]
        x_raw, y_raw = batch[2:4]
        print(idx, x.size(), y.size(), x_raw.size(), filenames)
        # print(idx, x.size(), y.size(), filenames)
        norm_data_x.append(x.numpy())
        norm_data_y.append(x.numpy())
        raw_data_x.append(x_raw.numpy())
        raw_data_y.append(y_raw.numpy())

    norm_data_x = np.concatenate(norm_data_x, axis=0)
    norm_data_y = np.concatenate(norm_data_y, axis=0)
    raw_data_x = np.concatenate(raw_data_x, axis=0)
    raw_data_y = np.concatenate(raw_data_y, axis=0)
    print("norm_data_x shape: ", norm_data_x.shape)
    print("raw_data_x shape: ", raw_data_x.shape)
    print_mean_std_by_var(norm_data_x, col_names_x, col_names)
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
        # RectRegionMaskTransform(region_mask, include_raw=True),
        StandardizeTransform(
            data_means,
            data_stds,
            col_names_x,
            col_names_y,
            col_names,
            normalize_input=True,
            normalize_output=True,
            include_raw=True
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
        include_filename=True,
        region_mask2d=(region_mask, 2)
        )
    trainloader = data.DataLoader(training_set, shuffle=False, batch_size=1,
                                  num_workers=1, collate_fn=filter_collate)
    norm_data_x = []
    norm_data_y = []
    raw_data_x = []
    raw_data_y = []
    # start_idx, end_idx = get_index_from_colnames(col_names, "dqvls_nn_in")
    for idx, batch in enumerate(trainloader):
        if batch is None:
            continue
        debug_print(f"batch elements: {len(batch)}", DEBUG)
        x, y, x_raw, y_raw, filenames = batch
        x, y = x.reshape(-1, x.shape[-1]), y.reshape(-1, y.shape[-1])
        print(idx, x.size(), y.size(), x_raw.size(), filenames)
        norm_data_x.append(x.numpy())
        norm_data_y.append(x.numpy())
        raw_data_x.append(x_raw.numpy())
        raw_data_y.append(y_raw.numpy())
    norm_data_x = np.concatenate(norm_data_x, axis=0)
    norm_data_y = np.concatenate(norm_data_y, axis=0)
    raw_data_x = np.concatenate(raw_data_x, axis=0)
    raw_data_y = np.concatenate(raw_data_y, axis=0)
    print("norm_data_x shape: ", norm_data_x.shape)
    print("raw_data_x shape: ", raw_data_x.shape)
    print("Norm data:")
    print_mean_std_by_var(norm_data_x, col_names_x, col_names)
    print("Raw data:")
    print_mean_std_by_var(raw_data_x, col_names_x, col_names)
    print("qtend_check: ", norm_data_y.mean(), norm_data_y.std())
    print_mean_std_by_var(raw_data_y, col_names_y, col_names)


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
            normalize_output=True,
            include_raw=True,
            region_mask2d=(region_mask,2)
            ),
        # RectRegionMaskTransform(region_mask, include_raw=True)
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
    trainloader = data.DataLoader(training_set, shuffle=False, batch_size=1,
                                  num_workers=1, collate_fn=filter_collate)
    norm_data_x = []
    norm_data_y = []
    # start_idx, end_idx = get_index_from_colnames(col_names, "dqvls_nn_in")
    for idx, batch in enumerate(trainloader):
        if batch is None:
            continue
        x, y = batch
        x_raw = batch[2]
        filenames = batch[-1]
        print(idx, x.size(), y.size(), x_raw.size(), filenames)
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

    # data_means = dict(np.load(data_dir + "/data_means.npz"))
    # data_stds = dict(np.load(data_dir + "/data_stds.npz"))

    data_means = dict(np.load("twp_region_means.npz"))
    data_stds = dict(np.load("twp_region_stds.npz"))

    # for k in data_means:
    #     if "_lev" in k and \
    #         ("QL" not in k and "qtend" not in k and "dqvls" not in k):
    #         debug_print(f"Replacing means {k}({data_means[k]}) with\
    #                     {k.rstrip('_lev')}({data_means[k.split('_lev')[0]]})", DEBUG)
    #         data_means[k] = data_means[k.split("_lev")[0]]
    #         debug_print(f"Replacing stds {k}({data_stds[k]}) with\
    #                     {k.rstrip('_lev')}({data_stds[k.split('_lev')[0]]})", DEBUG)
    #         data_stds[k] = data_stds[k.split("_lev")[0]]

    input_indices, prev_input_indices, output_indices = gen_multistep_col_indices(
        col_names, prev_ex_vars, col_names_x, col_names_y, 1)
    print("input_indices:", col_names[input_indices])
    print("prev_input_indices:", col_names[prev_input_indices])
    print("output_indices:", col_names[output_indices])

    transform = transforms.Compose([
        # RegionMaskTransform(region_mask, include_raw=True),
        StandardizeTransform(
            data_means,
            data_stds,
            col_names_x + prev_ex_vars + col_names_x,
            col_names_y,
            col_names,
            normalize_input=True,
            normalize_output=True,
            include_raw=True
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
        include_filename=True,
        region_mask1d=region_mask
        )
    trainloader = data.DataLoader(training_set, shuffle=False, batch_size=1,
                                  num_workers=1, collate_fn=filter_collate)
    norm_data_x = []
    norm_data_y = []
    # start_idx, end_idx = get_index_from_colnames(col_names, "dqvls_nn_in")
    for idx, batch in enumerate(trainloader):
        if batch is None:
            continue
        x, y = batch[:2]
        x_raw = batch[2]
        x, y = x.reshape(-1, x.shape[-1]), y.reshape(-1, y.shape[-1])
        filenames = batch[-1]
        print(idx, x.size(), y.size(), x_raw.size(), filenames)
        norm_data_x.append(x.numpy())
        norm_data_y.append(x.numpy())
    norm_data_x = np.concatenate(norm_data_x, axis=0)
    norm_data_y = np.concatenate(norm_data_y, axis=0)
    print_mean_std_by_var(norm_data_x, col_names_x + prev_ex_vars + col_names_x,
                          col_names)
    print("qtend_check: ", norm_data_y.mean(), norm_data_y.std())

def test_image_multistep1_pacific_region():
    file_names = glob.glob(data_dir + "*.npy")
    file_names.sort()
    file_names = file_names[:35040]
    # file_names = file_names[:10]
    # file_names = [data_dir + fn for fn in file_names]
    print("file_names:", file_names[:10])
    col_names = np.loadtxt(data_dir + "/col_names.txt", dtype=str)
    col_names_y = ["qtend_check"]
    x_ae = [
        "QL_lev",
        "T_nn_in_lev",
        "dqvls_nn_in_lev",
        "dTls_nn_in_lev",
        "SOLIN",
        "SPPS",
        "LWUP",
        "CAPE",
        "UL_lev",
        "VL_lev"
    ]
    x_ae_prev = [
        "qtend_check_lev",
        "stend_check_lev",
        "SOLL",
        "SOLLD",
        "SOLS",
        "SOLSD",
        "FSDS",
        "CLOUD_lev",
        "SPPRECC",
        "FLNS",
        "FLNT",
        "SPQRL_lev",
        "SPQRS_lev"
    ]
    col_names_x = x_ae
    # varaibles that are used as input in the previous time step
    # prev_ex_vars = ["qtend_check", "stend_check", "SOLL", "SOLLD", "SOLS", "SOLSD", "FSDS"]
    prev_ex_vars = x_ae_prev

    col_names_x = levelwise_variable(col_names_x, 0, 29)
    prev_ex_vars = levelwise_variable(prev_ex_vars, 0, 29)

    data_means = dict(np.load("../consts/twp_region_means.npz"))
    data_stds = dict(np.load("../consts/twp_region_stds.npz"))
    # col_names_x = col_names_x + prev_ex_vars + col_names_x

    input_indices, prev_input_indices, output_indices = gen_multistep_col_indices(
        col_names, prev_ex_vars, col_names_x, col_names_y, 1)
    print("input_indices:", col_names[input_indices])
    print("prev_input_indices:", col_names[prev_input_indices])
    print("output_indices:", col_names[output_indices])

    transform = transforms.Compose([
        # RectRegionMaskTransform(region_mask, include_raw=True),
        StandardizeTransform(
            data_means,
            data_stds,
            col_names_x + prev_ex_vars + col_names_x,
            col_names_y,
            col_names,
            normalize_input=True,
            normalize_output=True,
            include_raw=True
            ),
        ])

    training_set = DatasetDisk(
        file_names,
        input_indices,
        prev_input_indices,
        output_indices,
        multistep=1,
        sample_rate=192,
        is_train=True,
        transform=transform,
        include_filename=True,
        region_mask2d=(region_mask, 2)
        )
    trainloader = data.DataLoader(training_set, shuffle=False, batch_size=1, num_workers=1, collate_fn=filter_collate)
    norm_data_x = []
    norm_data_y = []
    # start_idx, end_idx = get_index_from_colnames(col_names, "dqvls_nn_in")
    for idx, batch in enumerate(trainloader):
        if batch is None:
            continue
        x, y, x_raw, _, filenames = batch
        print(idx, x.size(), y.size(), x_raw.size(), filenames)
        norm_data_x.append(x.numpy())
        norm_data_y.append(x.numpy())
    norm_data_x = np.concatenate(norm_data_x, axis=0)
    norm_data_y = np.concatenate(norm_data_y, axis=0)
    print_mean_std_by_var(norm_data_x, col_names_x + prev_ex_vars + col_names_x,
                          col_names)
    print("qtend_check: ", norm_data_y.mean(), norm_data_y.std())

if __name__ == "__main__":
    # test_single_column_multistep0_pacific_region()
    # test_single_column_multistep1_pacific_region()
    # test_single_column_multistep0_pacific_region_rect()
    # test_image_multistep0_pacific_region()
    test_image_multistep1_pacific_region()

