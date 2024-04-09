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

from preprocess import FlattenSpatialTransform, StandardizeTransform
from dataloader_utils import gen_multistep_col_indices, levelwise_variable, \
    delevelwise_variable
from dataloader_newformat import PairDatasetDisk, filter_collate
from debug_utils import print_mean_std_by_var
from logger import debug_print
# import torch transforms

DEBUG = True
START_LEV = 6
END_LEV = 29
col_names_x = [
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
col_names_x_ae = levelwise_variable(col_names_x, START_LEV, END_LEV)
col_names_x_resmlp = delevelwise_variable(col_names_x)
prev_ex_vars = [
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
prev_ex_vars_ae = levelwise_variable(prev_ex_vars, START_LEV, END_LEV)
prev_ex_vars_resmlp = delevelwise_variable(prev_ex_vars)
region_mask = np.load(os.path.join(os.path.dirname(__file__), "..", "..", \
                                "consts", "pacific_region_mask.npy"))
assert region_mask.shape == (96,144)

data_dir = "/home/users/data/nncam_data/image_set/"
if not os.path.isdir(data_dir):
    data_dir = "/data/nncam_data/image_set/"
if not os.path.isdir(data_dir):
    # data_dir = "./data/"
    data_dir = "../analysis/test_data/"
if not os.path.isdir(data_dir):
    # data_dir = "./data/"
    data_dir = "/pscratch/sd/c/chenjd21/spcam_new_data/"

data_means = dict(np.load(data_dir + "/data_means.npz"))
data_means_by_lvl = dict(np.load(data_dir + "/std_mean_by_level_means.npz"))
data_means.update(data_means_by_lvl)
data_stds = dict(np.load(data_dir + "/data_stds.npz"))
data_stds_by_lvl = dict(np.load(data_dir + "/std_mean_by_level_stds.npz"))
data_stds.update(data_stds_by_lvl)

# use the same mean and std for variables except for q related
for k in data_means:
    if "_lev" in k and \
        ("QL" not in k and "qtend" not in k and "dqvls" not in k):
        # debug_print(f"Replacing means {k}({data_means[k]}) with\
                    #  {k.rstrip('_lev')}({data_means[k.split('_lev')[0]]})", DEBUG)
        data_means[k] = data_means[k.split("_lev")[0]]
        # debug_print(f"Replacing stds {k}({data_stds[k]}) with\
                    #  {k.rstrip('_lev')}({data_stds[k.split('_lev')[0]]})", DEBUG)
        data_stds[k] = data_stds[k.split("_lev")[0]]

def test_single_column_multistep0():
    file_names = glob.glob(data_dir + "*.npy")
    file_names.sort()
    # file_names = file_names[:10]
    # file_names = [data_dir + fn for fn in file_names]
    print("file_names:", file_names[:10])
    col_names = np.loadtxt(data_dir + "/col_names.txt", dtype=str)
    col_names_y = ["qtend_check"]
    # varaibles that are used as input in the previous time step

    input_indices_ae, prev_input_indices_ae, output_indices_ae = gen_multistep_col_indices(
        col_names, prev_ex_vars_ae, col_names_x_ae, col_names_y, 1)
    print("input_indices_ae:", col_names[input_indices_ae])
    print("prev_input_indices_ae:", col_names[prev_input_indices_ae])
    print("output_indices_ae:", col_names[output_indices_ae])

    input_indices_resmlp, prev_input_indices_resmlp, output_indices_resmlp = gen_multistep_col_indices(
        col_names, prev_ex_vars_resmlp, col_names_x_resmlp, col_names_y, 1)
    print("input_indices_resmlp:", col_names[input_indices_resmlp])
    print("prev_input_indices_resmlp:", col_names[prev_input_indices_resmlp])
    print("output_indices_resmlp:", col_names[output_indices_resmlp])

    transform_ae = transforms.Compose([
        StandardizeTransform(
            data_means,
            data_stds,
            col_names_x_ae,
            col_names_y,
            col_names,
            normalize_input=True,
            normalize_output=True
            ),
        FlattenSpatialTransform()
        ])
    transform_resmlp = transforms.Compose([
        StandardizeTransform(
            data_means,
            data_stds,
            col_names_x_resmlp,
            col_names_y,
            col_names,
            normalize_input=True,
            normalize_output=True
            ),
        FlattenSpatialTransform()
        ])

    training_set = PairDatasetDisk(
        file_names,
        curr_input_indices1=input_indices_ae,
        curr_input_indices2=input_indices_resmlp,
        prev_input_indices1=prev_input_indices_ae,
        prev_input_indices2=prev_input_indices_resmlp,
        output_indices1=output_indices_ae,
        output_indices2=output_indices_resmlp,
        multistep=0,
        sample_rate=12,
        is_train=True,
        transform1=transform_ae,
        transform2=transform_resmlp,
        include_filename=True,
        region_mask1d=region_mask
        )
    trainloader = data.DataLoader(training_set, shuffle=False, batch_size=1, 
                                  num_workers=1, collate_fn=filter_collate)
    norm_data_x_ae = []
    norm_data_y_ae = []
    norm_data_x_resmlp = []
    norm_data_y_resmlp = []
    # start_idx, end_idx = get_index_from_colnames(col_names, "dqvls_nn_in")
    for idx, batch in enumerate(trainloader):
        if batch is None:
            continue
        x_ae, y_ae, x_resmlp, y_resmlp, filenames = batch # x shape (batch, n_samples, n_features, ...
        # reshape x into (batch*n_samples, n_features, ...)
        x_ae, y_ae = x_ae.reshape(-1, x_ae.shape[-1]), y_ae.reshape(-1, y_ae.shape[-1])
        print(idx, x_ae.size(), y_ae.size(), filenames)
        x_resmlp, y_resmlp = x_resmlp.reshape(-1, x_resmlp.shape[-1]), \
            y_resmlp.reshape(-1, y_resmlp.shape[-1])
        print(idx, x_resmlp.size(), y_resmlp.size(), filenames)
        norm_data_x_ae.append(x_ae.numpy())
        norm_data_y_ae.append(y_ae.numpy())
        norm_data_x_resmlp.append(x_resmlp.numpy())
        norm_data_y_resmlp.append(y_resmlp.numpy())
    norm_data_x_ae = np.concatenate(norm_data_x_ae, axis=0)
    norm_data_y_ae = np.concatenate(norm_data_y_ae, axis=0)
    norm_data_x_resmlp = np.concatenate(norm_data_x_resmlp, axis=0)
    norm_data_y_resmlp = np.concatenate(norm_data_y_resmlp, axis=0)
    print("norm_data_x_ae shape: ", norm_data_x_ae.shape)
    print("norm_data_x_resmlp shape: ", norm_data_x_resmlp.shape)
    print("norm ae:")
    print_mean_std_by_var(norm_data_x_ae, col_names_x_ae, col_names)
    print_mean_std_by_var(norm_data_x_ae, col_names_x_ae, col_names, delevelwise=True, reduce_lvl=6)
    print("norm resmlp:")
    print_mean_std_by_var(norm_data_x_resmlp, col_names_x_resmlp, col_names)
    print("qtend_check ae: ", norm_data_y_ae.mean(), norm_data_y_ae.std())
    print("qtend_check resmlp: ", norm_data_y_resmlp.mean(), norm_data_y_resmlp.std())
    
def test_single_column_multistep1():
    file_names = glob.glob(data_dir + "*.npy")
    file_names.sort()
    # file_names = file_names[:10]
    # file_names = [data_dir + fn for fn in file_names]
    print("file_names:", file_names[:10])
    col_names = np.loadtxt(data_dir + "/col_names.txt", dtype=str)
    col_names_y = ["qtend_check"]
    # varaibles that are used as input in the previous time step

    # col_names_x = col_names_x + prev_ex_vars + col_names_x

    input_indices_ae, prev_input_indices_ae, output_indices_ae = gen_multistep_col_indices(
        col_names, prev_ex_vars_ae, col_names_x_ae, col_names_y, 1)
    print("input_indices_ae:", col_names[input_indices_ae])
    print("prev_input_indices_ae:", col_names[prev_input_indices_ae])
    print("output_indices_ae:", col_names[output_indices_ae])

    input_indices_resmlp, prev_input_indices_resmlp, output_indices_resmlp = gen_multistep_col_indices(
        col_names, prev_ex_vars_resmlp, col_names_x_resmlp, col_names_y, 1)
    print("input_indices_resmlp:", col_names[input_indices_resmlp])
    print("prev_input_indices_resmlp:", col_names[prev_input_indices_resmlp])
    print("output_indices_resmlp:", col_names[output_indices_resmlp])

    transform_ae = transforms.Compose([
        StandardizeTransform(
            data_means,
            data_stds,
            col_names_x_ae+prev_ex_vars_ae+col_names_x_ae,
            col_names_y,
            col_names,
            normalize_input=True,
            normalize_output=True
            ),
        FlattenSpatialTransform()
        ])
    transform_resmlp = transforms.Compose([
        StandardizeTransform(
            data_means,
            data_stds,
            col_names_x_resmlp+prev_ex_vars_resmlp+col_names_x_resmlp,
            col_names_y,
            col_names,
            normalize_input=True,
            normalize_output=True
            ),
        FlattenSpatialTransform()
        ])

    training_set = PairDatasetDisk(
        file_names,
        curr_input_indices1=input_indices_ae,
        curr_input_indices2=input_indices_resmlp,
        prev_input_indices1=prev_input_indices_ae,
        prev_input_indices2=prev_input_indices_resmlp,
        output_indices1=output_indices_ae,
        output_indices2=output_indices_resmlp,
        multistep=1,
        sample_rate=12,
        is_train=True,
        transform1=transform_ae,
        transform2=transform_resmlp,
        include_filename=True,
        region_mask1d=region_mask
        )
    trainloader = data.DataLoader(training_set, shuffle=False, batch_size=1, 
                                  num_workers=1, collate_fn=filter_collate)
    norm_data_x_ae = []
    norm_data_y_ae = []
    norm_data_x_resmlp = []
    norm_data_y_resmlp = []
    # start_idx, end_idx = get_index_from_colnames(col_names, "dqvls_nn_in")
    for idx, batch in enumerate(trainloader):
        if batch is None:
            continue
        x_ae, y_ae, x_resmlp, y_resmlp, filenames = batch # x shape (batch, n_samples, n_features, ...
        # reshape x into (batch*n_samples, n_features, ...)
        x_ae, y_ae = x_ae.reshape(-1, x_ae.shape[-1]), y_ae.reshape(-1, y_ae.shape[-1])
        print(idx, x_ae.size(), y_ae.size(), filenames)
        x_resmlp, y_resmlp = x_resmlp.reshape(-1, x_resmlp.shape[-1]), \
            y_resmlp.reshape(-1, y_resmlp.shape[-1])
        print(idx, x_resmlp.size(), y_resmlp.size(), filenames)
        norm_data_x_ae.append(x_ae.numpy())
        norm_data_y_ae.append(y_ae.numpy())
        norm_data_x_resmlp.append(x_resmlp.numpy())
        norm_data_y_resmlp.append(y_resmlp.numpy())
    norm_data_x_ae = np.concatenate(norm_data_x_ae, axis=0)
    norm_data_y_ae = np.concatenate(norm_data_y_ae, axis=0)
    norm_data_x_resmlp = np.concatenate(norm_data_x_resmlp, axis=0)
    norm_data_y_resmlp = np.concatenate(norm_data_y_resmlp, axis=0)
    print("norm_data_x_ae shape: ", norm_data_x_ae.shape)
    print("norm_data_x_resmlp shape: ", norm_data_x_resmlp.shape)
    print("norm ae:")
    print_mean_std_by_var(norm_data_x_ae, col_names_x_ae+prev_ex_vars_ae+col_names_x_ae, col_names)
    print_mean_std_by_var(norm_data_x_ae, col_names_x_ae+prev_ex_vars_ae+col_names_x_ae,\
                           col_names, delevelwise=True, reduce_lvl=6)
    print("norm resmlp:")
    print_mean_std_by_var(norm_data_x_resmlp, 
                          col_names_x_resmlp+prev_ex_vars_resmlp+col_names_x_resmlp, col_names)
    print("qtend_check ae: ", norm_data_y_ae.mean(), norm_data_y_ae.std())
    print("qtend_check resmlp: ", norm_data_y_resmlp.mean(), norm_data_y_resmlp.std())

def test_image_multistep1_thres():
    file_names = glob.glob(data_dir + "*.npy")
    file_names.sort()
    # file_names = file_names[:10]
    # file_names = [data_dir + fn for fn in file_names]
    print("file_names:", file_names[:10])
    col_names = np.loadtxt(data_dir + "/col_names.txt", dtype=str)
    # col_names_x = ["QL", "T_nn_in", "dqvls_nn_in", "dTls_nn_in", "SOLIN", "SPPS"]
    col_names_y = ["qtend_check"]
    # varaibles that are used as input in the previous time step
    # prev_ex_vars = ["qtend_check", "stend_check", "SOLL", "SOLLD", "SOLS", "SOLSD", "FSDS"]

    # data_means = dict(np.load(data_dir + "/data_means.npz"))
    # data_stds = dict(np.load(data_dir + "/data_stds.npz"))
    # col_names_x = col_names_x + prev_ex_vars + col_names_x

    input_indices_ae, prev_input_indices_ae, output_indices_ae = gen_multistep_col_indices(
        col_names, prev_ex_vars_ae, col_names_x_ae, [], 1)
    print("input_indices_ae:", col_names[input_indices_ae])
    print("prev_input_indices_ae:", col_names[prev_input_indices_ae])
    print("output_indices_ae:", col_names[output_indices_ae])

    input_indices_resmlp, prev_input_indices_resmlp, output_indices_resmlp = gen_multistep_col_indices(
        col_names, prev_ex_vars_resmlp, col_names_x_resmlp, col_names_y, 1)
    print("input_indices_resmlp:", col_names[input_indices_resmlp])
    print("prev_input_indices_resmlp:", col_names[prev_input_indices_resmlp])
    print("output_indices_resmlp:", col_names[output_indices_resmlp])

    transform_ae = transforms.Compose([
        StandardizeTransform(
            data_means,
            data_stds,
            col_names_x_ae+prev_ex_vars_ae+col_names_x_ae,
            [],
            col_names,
            normalize_input=True,
            normalize_output=True,
            threshold=1
            ),
        # FlattenSpatialTransform()
        ])
    transform_resmlp = transforms.Compose([
        StandardizeTransform(
            data_means,
            data_stds,
            col_names_x_resmlp+prev_ex_vars_resmlp+col_names_x_resmlp,
            col_names_y,
            col_names,
            normalize_input=True,
            normalize_output=True
            ),
        # FlattenSpatialTransform()
        ])

    training_set = PairDatasetDisk(
        file_names,
        curr_input_indices1=input_indices_ae,
        curr_input_indices2=input_indices_resmlp,
        prev_input_indices1=prev_input_indices_ae,
        prev_input_indices2=prev_input_indices_resmlp,
        output_indices1=output_indices_ae,
        output_indices2=output_indices_resmlp,
        multistep=1,
        sample_rate=12,
        is_train=True,
        transform1=transform_ae,
        transform2=transform_resmlp,
        include_filename=True,
        region_mask2d=(region_mask,2)
        )
    trainloader = data.DataLoader(training_set, shuffle=False, batch_size=1, 
                                  num_workers=1, collate_fn=filter_collate)
    norm_data_x_ae = []
    norm_data_y_ae = []
    norm_data_x_resmlp = []
    norm_data_y_resmlp = []
    # start_idx, end_idx = get_index_from_colnames(col_names, "dqvls_nn_in")
    for idx, batch in enumerate(trainloader):
        if batch is None:
            continue
        x_ae, y_ae, x_resmlp, y_resmlp, filenames = batch # x shape (batch, n_samples, n_features, ...
        # reshape x into (batch*n_samples, n_features, ...)
        # x_ae, y_ae = x_ae.reshape(-1, x_ae.shape[-1]), y_ae.reshape(-1, y_ae.shape[-1])
        print(idx, x_ae.size(), y_ae.size(), filenames)
        # x_resmlp, y_resmlp = x_resmlp.reshape(-1, x_resmlp.shape[-1]), \
            # y_resmlp.reshape(-1, y_resmlp.shape[-1])
        print(idx, x_resmlp.size(), y_resmlp.size(), filenames)
        norm_data_x_ae.append(x_ae.numpy())
        norm_data_y_ae.append(y_ae.numpy())
        norm_data_x_resmlp.append(x_resmlp.numpy())
        norm_data_y_resmlp.append(y_resmlp.numpy())
    norm_data_x_ae = np.concatenate(norm_data_x_ae, axis=0)
    norm_data_y_ae = np.concatenate(norm_data_y_ae, axis=0)
    norm_data_x_resmlp = np.concatenate(norm_data_x_resmlp, axis=0)
    norm_data_y_resmlp = np.concatenate(norm_data_y_resmlp, axis=0)
    print("norm_data_x_ae shape: ", norm_data_x_ae.shape)
    print("norm_data_x_resmlp shape: ", norm_data_x_resmlp.shape)
    print("norm ae:")
    print_mean_std_by_var(norm_data_x_ae, col_names_x_ae+prev_ex_vars_ae+col_names_x_ae, col_names)
    print_mean_std_by_var(norm_data_x_ae, col_names_x_ae+prev_ex_vars_ae+col_names_x_ae,\
                           col_names, delevelwise=True, reduce_lvl=6)
    print("norm resmlp:")
    print_mean_std_by_var(norm_data_x_resmlp, 
                          col_names_x_resmlp+prev_ex_vars_resmlp+col_names_x_resmlp, col_names)
    print("qtend_check ae: ", norm_data_y_ae.mean(), norm_data_y_ae.std())
    print("qtend_check resmlp: ", norm_data_y_resmlp.mean(), norm_data_y_resmlp.std())

def test_image_multistep1():
    file_names = glob.glob(data_dir + "*.npy")
    file_names.sort()
    # file_names = file_names[:10]
    # file_names = [data_dir + fn for fn in file_names]
    print("file_names:", file_names[:10])
    col_names = np.loadtxt(data_dir + "/col_names.txt", dtype=str)
    # col_names_x = ["QL", "T_nn_in", "dqvls_nn_in", "dTls_nn_in", "SOLIN", "SPPS"]
    col_names_y = ["qtend_check"]
    # varaibles that are used as input in the previous time step
    # prev_ex_vars = ["qtend_check", "stend_check", "SOLL", "SOLLD", "SOLS", "SOLSD", "FSDS"]

    # data_means = dict(np.load(data_dir + "/data_means.npz"))
    # data_stds = dict(np.load(data_dir + "/data_stds.npz"))
    # col_names_x = col_names_x + prev_ex_vars + col_names_x

    input_indices_ae, prev_input_indices_ae, output_indices_ae = gen_multistep_col_indices(
        col_names, prev_ex_vars_ae, col_names_x_ae, [], 1)
    print("input_indices_ae:", col_names[input_indices_ae])
    print("prev_input_indices_ae:", col_names[prev_input_indices_ae])
    print("output_indices_ae:", col_names[output_indices_ae])

    input_indices_resmlp, prev_input_indices_resmlp, output_indices_resmlp = gen_multistep_col_indices(
        col_names, prev_ex_vars_resmlp, col_names_x_resmlp, col_names_y, 1)
    print("input_indices_resmlp:", col_names[input_indices_resmlp])
    print("prev_input_indices_resmlp:", col_names[prev_input_indices_resmlp])
    print("output_indices_resmlp:", col_names[output_indices_resmlp])

    transform_ae = transforms.Compose([
        StandardizeTransform(
            data_means,
            data_stds,
            col_names_x_ae+prev_ex_vars_ae+col_names_x_ae,
            [],
            col_names,
            normalize_input=True,
            normalize_output=True
            ),
        # FlattenSpatialTransform()
        ])
    transform_resmlp = transforms.Compose([
        StandardizeTransform(
            data_means,
            data_stds,
            col_names_x_resmlp+prev_ex_vars_resmlp+col_names_x_resmlp,
            col_names_y,
            col_names,
            normalize_input=True,
            normalize_output=True
            ),
        # FlattenSpatialTransform()
        ])

    training_set = PairDatasetDisk(
        file_names,
        curr_input_indices1=input_indices_ae,
        curr_input_indices2=input_indices_resmlp,
        prev_input_indices1=prev_input_indices_ae,
        prev_input_indices2=prev_input_indices_resmlp,
        output_indices1=output_indices_ae,
        output_indices2=output_indices_resmlp,
        multistep=1,
        sample_rate=12,
        is_train=True,
        transform1=transform_ae,
        transform2=transform_resmlp,
        include_filename=True,
        region_mask2d=(region_mask,2)
        )
    trainloader = data.DataLoader(training_set, shuffle=False, batch_size=1, 
                                  num_workers=1, collate_fn=filter_collate)
    norm_data_x_ae = []
    norm_data_y_ae = []
    norm_data_x_resmlp = []
    norm_data_y_resmlp = []
    # start_idx, end_idx = get_index_from_colnames(col_names, "dqvls_nn_in")
    for idx, batch in enumerate(trainloader):
        if batch is None:
            continue
        x_ae, y_ae, x_resmlp, y_resmlp, filenames = batch # x shape (batch, n_samples, n_features, ...
        # reshape x into (batch*n_samples, n_features, ...)
        # x_ae, y_ae = x_ae.reshape(-1, x_ae.shape[-1]), y_ae.reshape(-1, y_ae.shape[-1])
        print(idx, x_ae.size(), y_ae.size(), filenames)
        # x_resmlp, y_resmlp = x_resmlp.reshape(-1, x_resmlp.shape[-1]), \
            # y_resmlp.reshape(-1, y_resmlp.shape[-1])
        print(idx, x_resmlp.size(), y_resmlp.size(), filenames)
        norm_data_x_ae.append(x_ae.numpy())
        norm_data_y_ae.append(y_ae.numpy())
        norm_data_x_resmlp.append(x_resmlp.numpy())
        norm_data_y_resmlp.append(y_resmlp.numpy())
    norm_data_x_ae = np.concatenate(norm_data_x_ae, axis=0)
    norm_data_y_ae = np.concatenate(norm_data_y_ae, axis=0)
    norm_data_x_resmlp = np.concatenate(norm_data_x_resmlp, axis=0)
    norm_data_y_resmlp = np.concatenate(norm_data_y_resmlp, axis=0)
    print("norm_data_x_ae shape: ", norm_data_x_ae.shape)
    print("norm_data_x_resmlp shape: ", norm_data_x_resmlp.shape)
    print("norm ae:")
    print_mean_std_by_var(norm_data_x_ae, col_names_x_ae+prev_ex_vars_ae+col_names_x_ae, col_names)
    print_mean_std_by_var(norm_data_x_ae, col_names_x_ae+prev_ex_vars_ae+col_names_x_ae,\
                           col_names, delevelwise=True, reduce_lvl=6)
    print("norm resmlp:")
    print_mean_std_by_var(norm_data_x_resmlp, 
                          col_names_x_resmlp+prev_ex_vars_resmlp+col_names_x_resmlp, col_names)
    print("qtend_check ae: ", norm_data_y_ae.mean(), norm_data_y_ae.std())
    print("qtend_check resmlp: ", norm_data_y_resmlp.mean(), norm_data_y_resmlp.std())

if __name__ == "__main__":
    # test_single_column_multistep0()
    # test_image_multistep0()
    # test_single_column_multistep1()
    # test_image_multistep1()
    test_image_multistep1_thres()

