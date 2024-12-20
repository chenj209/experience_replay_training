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

from preprocess import FlattenSpatialTransform, MinMaxTransformLegacy, StandardizeTransform, MinMaxTransformLegacy2step
from dataloader_utils import gen_multistep_col_indices, get_index_from_colnames, gen_col_indices
from dataloader_legacy_format import DatasetDisk, filter_collate
from debug_utils import print_mean_std_by_var, print_min_max_by_var
from normalization import inverse_legacy
from data_shape import to_inference_shape
# import torch transforms

# data_dir = "/home/users/data/nncam_data/image_set/"
data_dir = "/data/nncam_data/image_set/"
ex_data_dir = "/data/chenj209/ex_dataset"
if not os.path.isdir(data_dir):
    data_dir = "./data"
    ex_data_dir = "./ex_data"
    print("current path: ", os.getcwd(), f", files in {data_dir}: {os.listdir(data_dir)[:10]}")
# if not os.path.isdir(data_dir):
    # data_dir = "./data/"
    # data_dir = "../analysis/test_data/"
# if not os.path.isdir(data_dir):
    # data_dir = "./data/"
    # data_dir = "/pscratch/sd/c/chenjd21/spcam_new_data/"
# COL_NAMES = ["QL", "T_nn_in", ]
COL_NAMES_LEGACY = {
    "X": [
        *[f"QL_lev{i}" for i in range(30)],
        *[f"T_nn_in_lev{i}" for i in range(30)],
        *[f"dqvls_nn_in_lev{i}" for i in range(30)],
        *[f"dTls_nn_in_lev{i}" for i in range(30)],
        "SOLIN",
        "SPPS"
    ],
    "Y": [
        *[f"qtend_check_lev{i}" for i in range(30)],
        *[f"stend_check_lev{i}" for i in range(30)],
        "UNKNOWN",
        "SOLL", "SOLS", "SOLSD", "SOLLD", "FSDS"
    ],
    "EX": [*[f"UL_lev{i}" for i in range(30)],
           *[f"VL_lev{i}" for i in range(30)],
           *[f"CLOUD_lev{i}" for i in range(30)], 
           "CAPE", "FLNS", "FLNT", "SPPRECC", "LWUP"]
}
def test_single_column_multistep0_minmax():
    file_names = glob.glob(data_dir + "/*.npz")
    file_names.sort()
    print(f"file_names in {data_dir}:", file_names[:10])
    col_names = np.loadtxt("col_names.txt", dtype=str)
    col_names_x = ["QL", "T_nn_in", "dqvls_nn_in", "dTls_nn_in", "SOLIN", "SPPS"]
    col_names_y = ["qtend_check", "stend_check", "SOLL", "SOLS", "SOLSD", "SOLLD", "FSDS"]
    col_names_prev = ["qtend_check", "stend_check", "SOLL", "SOLS", "SOLSD", "SOLLD", "FSDS"]
    input_indices = [
        ("X", gen_col_indices(COL_NAMES_LEGACY["X"], col_names_x)),
        ("EX", gen_col_indices(COL_NAMES_LEGACY["EX"], col_names_x))
    ]
    output_indices = [("Y", gen_col_indices(COL_NAMES_LEGACY["Y"], col_names_y))] # index 60 is not used
    prev_input_indices = [
        ("X", gen_col_indices(COL_NAMES_LEGACY["X"], col_names_x)),
        ("EX", gen_col_indices(COL_NAMES_LEGACY["EX"], col_names_x + col_names_prev)),
        ("Y", gen_col_indices(COL_NAMES_LEGACY["Y"], col_names_prev))
    ]
    data_means = dict(np.load("../consts/all_means.npz"))
    data_stds = dict(np.load("../consts/all_stds.npz"))

    transform = transforms.Compose([
        MinMaxTransformLegacy(include_raw=True),
        FlattenSpatialTransform()
        ])

    training_set = DatasetDisk(
        file_names,
        input_indices,
        prev_input_indices,
        output_indices,
        multistep=0,
        sample_rate=144,
        is_train=True,
        transform=transform,
        include_filename=True
        )
    trainloader = data.DataLoader(training_set, shuffle=False, batch_size=1, num_workers=1, collate_fn=filter_collate)
    norm_data_x = []
    norm_data_y = []
    raw_data_x = []
    raw_data_y = []
    for idx, batch in enumerate(trainloader):
        if batch is None:
            continue
        print("batch items: ", len(batch))
        x, y = batch[:2] # x: (batch, n_samples, n_features)
        x_raw, y_raw = batch[2:4]
        x_raw = x_raw.reshape(x_raw.shape[0], -1).T
        y_raw = y_raw.reshape(y_raw.shape[0], -1).T
        print("x shape: ", x.shape)
        print("x_raw shape: ", x_raw.shape)
        print("y_raw shape: ", y_raw.shape)
        x = x.reshape(-1, x.shape[-1]) # x: (batch * n_samples, n_features)
        y = y.reshape(-1, y.shape[-1])
        filenames = batch[-1]
        print(idx, x.size(), y.size(), x_raw.size(), y_raw.size(), filenames)
        norm_data_x.append(x.numpy())
        norm_data_y.append(y.numpy())
    norm_data_x = np.concatenate(norm_data_x, axis=0)
    norm_data_y = np.concatenate(norm_data_y, axis=0)
    raw_data_x = np.concatenate(raw_data_x, axis=0)
    raw_data_y = np.concatenate(raw_data_y, axis=0)
    np.save("debug_norm_y", norm_data_y)
    np.save("debug_raw_y", raw_data_y)
    print("norm_data_x shape: ", norm_data_x.shape)
    print("Mean Std by var")
    print_mean_std_by_var(norm_data_x, col_names_x + col_names_prev + col_names_x, col_names)
    print_mean_std_by_var(norm_data_y, col_names_y, col_names)
    print("Raw Mean Std by var")
    print_mean_std_by_var(raw_data_x, col_names_x + col_names_prev + col_names_x, col_names)
    print_mean_std_by_var(raw_data_y, col_names_y, col_names)
def test_single_column_multistep1_minmax():
    file_names = glob.glob(data_dir + "/*.npz")
    file_names.sort()
    print(f"file_names in {data_dir}:", file_names[:10])
    col_names = np.loadtxt("col_names.txt", dtype=str)
    col_names_x = ["QL", "T_nn_in", "dqvls_nn_in", "dTls_nn_in", "SOLIN", "SPPS"]
    col_names_y = ["qtend_check", "stend_check", "SOLL", "SOLS", "SOLSD", "SOLLD", "FSDS"]
    col_names_prev = ["qtend_check", "stend_check", "SOLL", "SOLS", "SOLSD", "SOLLD", "FSDS"]
    input_indices = [
        ("X", gen_col_indices(COL_NAMES_LEGACY["X"], col_names_x)),
        ("EX", gen_col_indices(COL_NAMES_LEGACY["EX"], col_names_x))
    ]
    output_indices = [("Y", gen_col_indices(COL_NAMES_LEGACY["Y"], col_names_y))] # index 60 is not used
    prev_input_indices = [
        ("X", gen_col_indices(COL_NAMES_LEGACY["X"], col_names_x)),
        ("EX", gen_col_indices(COL_NAMES_LEGACY["EX"], col_names_x + col_names_prev)),
        ("Y", gen_col_indices(COL_NAMES_LEGACY["Y"], col_names_prev))
    ]
    data_means = dict(np.load("../consts/all_means.npz"))
    data_stds = dict(np.load("../consts/all_stds.npz"))

    transform = transforms.Compose([
        MinMaxTransformLegacy2step(include_raw=True),
        FlattenSpatialTransform()
        ])

    training_set = DatasetDisk(
        file_names,
        input_indices,
        prev_input_indices,
        output_indices,
        multistep=1,
        sample_rate=144,
        is_train=True,
        transform=transform,
        include_filename=True
        )
    trainloader = data.DataLoader(training_set, shuffle=False, batch_size=1, num_workers=1, collate_fn=filter_collate)
    norm_data_x = []
    norm_data_y = []
    raw_data_x = []
    raw_data_y = []
    for idx, batch in enumerate(trainloader):
        if batch is None:
            continue
        print("batch items: ", len(batch))
        x, y = batch[:2] # x: (batch, n_samples, n_features)
        x_raw, y_raw = batch[2:4]
        print("x shape: ", x.shape)
        print("x_raw shape: ", x_raw.shape)
        print("y_raw shape: ", y_raw.shape)
        x = x.reshape(-1, x.shape[-1]) # x: (batch * n_samples, n_features)
        y = y.reshape(-1, y.shape[-1])
        filenames = batch[-1]
        print(idx, x.size(), y.size(), x_raw.size(), y_raw.size(), filenames)
        norm_data_x.append(x.numpy())
        norm_data_y.append(y.numpy())
        raw_data_x.append(x_raw.numpy())
        raw_data_y.append(y_raw.numpy())
    norm_data_x = np.concatenate(norm_data_x, axis=0)
    norm_data_y = np.concatenate(norm_data_y, axis=0)
    raw_data_x = np.concatenate(raw_data_x, axis=0)
    raw_data_y = np.concatenate(raw_data_y, axis=0)
    raw_data_x = to_inference_shape(raw_data_x)
    raw_data_x = raw_data_x.reshape(-1, raw_data_x.shape[-1])
    raw_data_y = to_inference_shape(raw_data_y)
    raw_data_y = raw_data_y.reshape(-1, raw_data_y.shape[-1])
    print(raw_data_x.shape)
    print(norm_data_x.shape)
    print(raw_data_y.shape)
    print(norm_data_y.shape)

    assert(np.allclose(raw_data_x[:,:30],inverse_legacy["Q"](norm_data_x[:,:30])))
    assert(np.allclose(raw_data_x[:,30:60], inverse_legacy["T"](norm_data_x[:,30:60])))
    assert(np.allclose(raw_data_x[:,60:90], inverse_legacy["dqls"](norm_data_x[:,60:90])))
    assert(np.allclose(raw_data_x[:,90:120], inverse_legacy["dTls"](norm_data_x[:,90:120])))
    assert(np.allclose(raw_data_x[:,120], inverse_legacy["solin"](norm_data_x[:,120])))
    assert(np.allclose(raw_data_x[:,121], inverse_legacy["ps"](norm_data_x[:,121])))
    assert(np.allclose(raw_data_y[:,:30], inverse_legacy["qtend"](norm_data_y[:,:30])))
    assert(np.allclose(raw_data_y[:,30:60], inverse_legacy["stend"](norm_data_y[:,30:60])))
    assert(np.allclose(raw_data_y[:,60:65], inverse_legacy["radiation"](norm_data_y[:,60:65])))
    np.save("debug_norm_y", norm_data_y)
    np.save("debug_raw_y", raw_data_y)
    print("norm_data_x shape: ", norm_data_x.shape)
    print("Mean Std by var")
    print_mean_std_by_var(norm_data_x, col_names_x + col_names_prev + col_names_x, col_names)
    print_mean_std_by_var(norm_data_y, col_names_y, col_names)
    print("Raw Mean Std by var")
    print_mean_std_by_var(raw_data_x, col_names_x + col_names_prev + col_names_x, col_names)
    print_mean_std_by_var(raw_data_y, col_names_y, col_names)

def test_single_column_multistep1_ex():
    file_names = glob.glob(data_dir + "/*.npz")
    file_names.sort()
    print(f"file_names in {data_dir}:", file_names[:10])
    col_names = np.loadtxt("col_names.txt", dtype=str)
    col_names_x = ["QL", "T_nn_in", "dqvls_nn_in", "dTls_nn_in", "SOLIN", "SPPS", "UL", "VL", "LWUP"]
    col_names_y = ["qtend_check", "stend_check", "SOLL", "SOLS", "SOLSD", "SOLLD", "FSDS"]
    col_names_prev = ["FLNS", "FLNT", "qtend_check", "stend_check", "SOLL", "SOLS", "SOLSD", "SOLLD", "FSDS"]
    input_indices = [
        ("X", gen_col_indices(COL_NAMES_LEGACY["X"], col_names_x)),
        ("EX", gen_col_indices(COL_NAMES_LEGACY["EX"], col_names_x))
    ]
    output_indices = [("Y", gen_col_indices(COL_NAMES_LEGACY["Y"], col_names_y))] # index 60 is not used
    prev_input_indices = [
        ("X", gen_col_indices(COL_NAMES_LEGACY["X"], col_names_x)),
        ("EX", gen_col_indices(COL_NAMES_LEGACY["EX"], col_names_x + col_names_prev)),
        ("Y", gen_col_indices(COL_NAMES_LEGACY["Y"], col_names_prev))
    ]
    data_means = dict(np.load("../consts/all_means.npz"))
    data_stds = dict(np.load("../consts/all_stds.npz"))

    transform = transforms.Compose([
        # MinMaxTransformLegacy(include_raw=True),
        StandardizeTransform(
            data_means,
            data_stds,
            col_names_x + col_names_prev + col_names_x,
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
        include_filename=True
        )
    trainloader = data.DataLoader(training_set, shuffle=False, batch_size=1, num_workers=1, collate_fn=filter_collate)
    norm_data_x = []
    norm_data_y = []
    raw_data_x = []
    raw_data_y = []
    for idx, batch in enumerate(trainloader):
        if batch is None:
            continue
        print("batch items: ", len(batch))
        x, y = batch[:2] # x: (batch, n_samples, n_features)
        x_raw, y_raw = batch[2:4]
        print("x shape: ", x.shape)
        print("x_raw shape: ", x_raw.shape)
        print("y_raw shape: ", y_raw.shape)
        x = x.reshape(-1, x.shape[-1]) # x: (batch * n_samples, n_features)
        y = y.reshape(-1, y.shape[-1])
        filenames = batch[-1]
        print(idx, x.size(), y.size(), x_raw.size(), y_raw.size(), filenames)
        norm_data_x.append(x.numpy())
        norm_data_y.append(y.numpy())
        raw_data_x.append(x_raw.numpy())
        raw_data_y.append(y_raw.numpy())
    norm_data_x = np.concatenate(norm_data_x, axis=0)
    norm_data_y = np.concatenate(norm_data_y, axis=0)
    raw_data_x = np.concatenate(raw_data_x, axis=0)
    raw_data_y = np.concatenate(raw_data_y, axis=0)
    np.save("debug_norm_y", norm_data_y)
    np.save("debug_raw_y", raw_data_y)
    print("norm_data_x shape: ", norm_data_x.shape)
    print("Mean Std by var")
    print_mean_std_by_var(norm_data_x, col_names_x + col_names_prev + col_names_x, col_names)
    print_mean_std_by_var(norm_data_y, col_names_y, col_names)
    print("Raw Mean Std by var")
    print_mean_std_by_var(raw_data_x, col_names_x + col_names_prev + col_names_x, col_names)
    print_mean_std_by_var(raw_data_y, col_names_y, col_names)

def test_single_column_multistep0_ex():
    file_names = glob.glob(data_dir + "/*.npz")
    file_names.sort()
    print(f"file_names in {data_dir}:", file_names[:10])
    col_names = np.loadtxt("col_names.txt", dtype=str)
    prev_ex_vars = None
    col_names_x = ["QL", "T_nn_in", "dqvls_nn_in", "dTls_nn_in", "SOLIN", "SPPS", "UL", "VL", "LWUP"]
    col_names_y = ["qtend_check", "stend_check", "SOLL", "SOLS", "SOLSD", "SOLLD", "FSDS"]
    input_indices = [
        ("X", gen_col_indices(COL_NAMES_LEGACY["X"], col_names_x)),
        ("EX", gen_col_indices(COL_NAMES_LEGACY["EX"], col_names_x))
    ]
    output_indices = [("Y", gen_col_indices(COL_NAMES_LEGACY["Y"], col_names_y))] # index 60 is not used
    prev_input_indices = None
    data_means = dict(np.load("../consts/all_means.npz"))
    data_stds = dict(np.load("../consts/all_stds.npz"))

    transform = transforms.Compose([
        # MinMaxTransformLegacy(include_raw=True),
        StandardizeTransform(
            data_means,
            data_stds,
            col_names_x,
            col_names_y,
            # np.concatenate([COL_NAMES_X,COL_NAMES_Y]),
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
        include_filename=True
        )
    trainloader = data.DataLoader(training_set, shuffle=False, batch_size=1, num_workers=1, collate_fn=filter_collate)
    norm_data_x = []
    norm_data_y = []
    raw_data_x = []
    raw_data_y = []
    for idx, batch in enumerate(trainloader):
        if batch is None:
            continue
        print("batch items: ", len(batch))
        x, y = batch[:2] # x: (batch, n_samples, n_features)
        x_raw, y_raw = batch[2:4]
        print("x shape: ", x.shape)
        print("x_raw shape: ", x_raw.shape)
        print("y_raw shape: ", y_raw.shape)
        x = x.reshape(-1, x.shape[-1]) # x: (batch * n_samples, n_features)
        y = y.reshape(-1, y.shape[-1])
        filenames = batch[-1]
        print(idx, x.size(), y.size(), x_raw.size(), y_raw.size(), filenames)
        norm_data_x.append(x.numpy())
        norm_data_y.append(y.numpy())
        raw_data_x.append(x_raw.numpy())
        raw_data_y.append(y_raw.numpy())
    norm_data_x = np.concatenate(norm_data_x, axis=0)
    norm_data_y = np.concatenate(norm_data_y, axis=0)
    raw_data_x = np.concatenate(raw_data_x, axis=0)
    raw_data_y = np.concatenate(raw_data_y, axis=0)
    np.save("debug_norm_y", norm_data_y)
    np.save("debug_raw_y", raw_data_y)
    print("norm_data_x shape: ", norm_data_x.shape)
    print("Mean Std by var")
    print_mean_std_by_var(norm_data_x, col_names_x, col_names)
    print_mean_std_by_var(norm_data_y, col_names_y, col_names)
    print("Raw Mean Std by var")
    print_mean_std_by_var(raw_data_x, col_names_x, col_names)
    print_mean_std_by_var(raw_data_y, col_names_y, col_names)

def test_single_column_multistep0():
    file_names = glob.glob(data_dir + "/*.npz")
    file_names.sort()
    # file_names = file_names[:10]
    # file_names = [data_dir + fn for fn in file_names]
    print(f"file_names in {data_dir}:", file_names[:10])
    #col_names = np.loadtxt(data_dir + "/col_names.txt", dtype=str)
    col_names = np.loadtxt("col_names.txt", dtype=str)
    col_names_x = ["QL", "T_nn_in", "dqvls_nn_in", "dTls_nn_in", "SOLIN", "SPPS"]
    # col_names_y = ["qtend_check", "stend_check","SOLL", "SOLLD", "SOLS", "SOLSD", "FSDS"]
    col_names_y = ["qtend_check", "stend_check", "SOLL", "SOLS", "SOLSD", "SOLLD", "FSDS"]
    # varaibles that are used as input in the previous time step
    # prev_ex_vars = ["qtend_check", "stend_check", "SOLL", "SOLLD", "SOLS", "SOLSD", "FSDS"]
    # prev_ex_vars = ["qtend_check", "stend_check", "SOLL", "SOLS", "SOLSD", "SOLLD", "FSDS"]
    prev_ex_vars = ["qtend_check", "stend_check", "SOLL", "SOLS", "SOLSD", "SOLLD", "FSDS"]
    col_names_x = ["QL", "T_nn_in", "dqvls_nn_in", "dTls_nn_in", "SOLIN", "SPPS"]
    col_names_y = ["qtend_check", "stend_check", "SOLL", "SOLS", "SOLSD", "SOLLD", "FSDS"]
    input_indices = [("X", np.arange(122))]
    output_indices = [("Y", np.concatenate([np.arange(60), np.arange(61,66)]))] # index 60 is not used
    prev_input_indices = [("X", np.arange(122)), ("Y", np.concatenate([np.arange(60), np.arange(61,66)]))]

    #data_means = dict(np.load(data_dir + "/data_means.npz"))
    data_means = dict(np.load("../consts/all_means.npz"))
    data_stds = dict(np.load("../consts/all_stds.npz"))

    # input_indices, prev_input_indices, output_indices = gen_multistep_col_indices(
    #     COL_NAMES_X, prev_ex_vars, col_names_x, col_names_y, 0)
    # print("input_indices:", input_indices)
    # print("input_indices:", col_names[input_indices])
    # print("prev_input_indices:", col_names[prev_input_indices])
    # print("output_indices:", col_names[output_indices])

    transform = transforms.Compose([
        # MinMaxTransformLegacy(include_raw=True),
        StandardizeTransform(
            data_means,
            data_stds,
            col_names_x,
            col_names_y,
            # np.concatenate([COL_NAMES_X,COL_NAMES_Y]),
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
        include_filename=True
        )
    trainloader = data.DataLoader(training_set, shuffle=False, batch_size=1, num_workers=1, collate_fn=filter_collate)
    norm_data_x = []
    norm_data_y = []
    raw_data_x = []
    raw_data_y = []
    # start_idx, end_idx = get_index_from_colnames(col_names, "dqvls_nn_in")
    # min_max_dict = {}
    # for col in col_names_x + col_names_y:
    #     min_max_dict[col] = [1e9, -1e9]
    for idx, batch in enumerate(trainloader):
        if batch is None:
            continue
        print("batch items: ", len(batch))
        x, y = batch[:2] # x: (batch, n_samples, n_features)
        x_raw, y_raw = batch[2:4]
        print("x shape: ", x.shape)
        print("x_raw shape: ", x_raw.shape)
        print("y_raw shape: ", y_raw.shape)
        x = x.reshape(-1, x.shape[-1]) # x: (batch * n_samples, n_features)
        y = y.reshape(-1, y.shape[-1])
        filenames = batch[-1]
        print(idx, x.size(), y.size(), x_raw.size(), y_raw.size(), filenames)
        norm_data_x.append(x.numpy())
        norm_data_y.append(y.numpy())
        raw_data_x.append(x_raw.numpy())
        raw_data_y.append(y_raw.numpy())
    norm_data_x = np.concatenate(norm_data_x, axis=0)
    norm_data_y = np.concatenate(norm_data_y, axis=0)
    raw_data_x = np.concatenate(raw_data_x, axis=0)
    raw_data_y = np.concatenate(raw_data_y, axis=0)
    np.save("debug_norm_y", norm_data_y)
    np.save("debug_raw_y", raw_data_y)
    print("norm_data_x shape: ", norm_data_x.shape)
    print("Mean Std by var")
    print_mean_std_by_var(norm_data_x, col_names_x, col_names)
    print_mean_std_by_var(norm_data_y, col_names_y, col_names)
    print("Raw Mean Std by var")

    print_mean_std_by_var(raw_data_x, col_names_x, col_names)
    print_mean_std_by_var(raw_data_y, col_names_y, col_names)
    # print("qtend_check: ", norm_data_y.mean(), norm_data_y.std())
    print("qtend_check: ", norm_data_y.mean(), norm_data_y.std())
    
def test_single_column_multistep1():
    file_names = glob.glob(data_dir + "/*.npz")
    file_names.sort()
    # file_names = file_names[:10]
    # file_names = [data_dir + fn for fn in file_names]
    print(f"dataset size {len(file_names)} file_names in {data_dir}:", file_names[:10])
    #col_names = np.loadtxt(data_dir + "/col_names.txt", dtype=str)
    col_names = np.loadtxt("col_names.txt", dtype=str)
    col_names_x = ["QL", "T_nn_in", "dqvls_nn_in", "dTls_nn_in", "SOLIN", "SPPS"]
    # col_names_y = ["qtend_check", "stend_check","SOLL", "SOLLD", "SOLS", "SOLSD", "FSDS"]
    col_names_y = ["qtend_check", "stend_check", "SOLL", "SOLS", "SOLSD", "SOLLD", "FSDS"]
    # varaibles that are used as input in the previous time step
    # prev_ex_vars = ["qtend_check", "stend_check", "SOLL", "SOLLD", "SOLS", "SOLSD", "FSDS"]
    # prev_ex_vars = ["qtend_check", "stend_check", "SOLL", "SOLS", "SOLSD", "SOLLD", "FSDS"]
    prev_ex_vars = ["qtend_check", "stend_check", "SOLL", "SOLS", "SOLSD", "SOLLD", "FSDS"]
    col_names_x = ["QL", "T_nn_in", "dqvls_nn_in", "dTls_nn_in", "SOLIN", "SPPS"]
    col_names_y = ["qtend_check", "stend_check", "SOLL", "SOLS", "SOLSD", "SOLLD", "FSDS"]
    # input_indices = [("X", np.arange(122))]
    input_indices = [("X", gen_col_indices(col_names_x, COL_NAMES_LEGACY["X"]))]
    # output_indices = [("Y", np.concatenate([np.arange(60), np.arange(61,66)]))] # index 60 is not used
    output_indices = [("Y", gen_col_indices(col_names_y, COL_NAMES_LEGACY["Y"]))] # index 60 is not used
    prev_input_indices = [("X", np.arange(122)), ("Y", np.concatenate([np.arange(60), np.arange(61,66)]))]

    #data_means = dict(np.load(data_dir + "/data_means.npz"))
    data_means = dict(np.load("../consts/all_means.npz"))
    data_stds = dict(np.load("../consts/all_stds.npz"))

    # input_indices, prev_input_indices, output_indices = gen_multistep_col_indices(
    #     COL_NAMES_X, prev_ex_vars, col_names_x, col_names_y, 0)
    # print("input_indices:", input_indices)
    # print("input_indices:", col_names[input_indices])
    # print("prev_input_indices:", col_names[prev_input_indices])
    # print("output_indices:", col_names[output_indices])

    transform = transforms.Compose([
        # MinMaxTransformLegacy(include_raw=True),
        StandardizeTransform(
            data_means,
            data_stds,
            col_names_x+col_names_y+col_names_x,
            col_names_y,
            # np.concatenate([COL_NAMES_X,COL_NAMES_Y]),
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
        include_filename=True
        )
    trainloader = data.DataLoader(training_set, shuffle=False, batch_size=1, num_workers=1, collate_fn=filter_collate)
    norm_data_x = []
    norm_data_y = []
    raw_data_x = []
    raw_data_y = []
    # start_idx, end_idx = get_index_from_colnames(col_names, "dqvls_nn_in")
    # min_max_dict = {}
    # for col in col_names_x + col_names_y:
    #     min_max_dict[col] = [1e9, -1e9]
    for idx, batch in enumerate(trainloader):
        if batch is None:
            continue
        print("batch items: ", len(batch))
        x, y = batch[:2] # x: (batch, n_samples, n_features)
        x_raw, y_raw = batch[2:4]
        print("x shape: ", x.shape)
        print("x_raw shape: ", x_raw.shape)
        print("y_raw shape: ", y_raw.shape)
        x = x.reshape(-1, x.shape[-1]) # x: (batch * n_samples, n_features)
        y = y.reshape(-1, y.shape[-1])
        filenames = batch[-1]
        print(idx, x.size(), y.size(), x_raw.size(), y_raw.size(), filenames)
        norm_data_x.append(x.numpy())
        norm_data_y.append(y.numpy())
        raw_data_x.append(x_raw.numpy())
        raw_data_y.append(y_raw.numpy())
    norm_data_x = np.concatenate(norm_data_x, axis=0)
    norm_data_y = np.concatenate(norm_data_y, axis=0)
    raw_data_x = np.concatenate(raw_data_x, axis=0)
    raw_data_y = np.concatenate(raw_data_y, axis=0)
    np.save("debug_norm_y", norm_data_y)
    np.save("debug_raw_y", raw_data_y)
    print("norm_data_x shape: ", norm_data_x.shape)
    print("Mean Std by var")
    print_mean_std_by_var(norm_data_x, col_names_x+col_names_y+col_names_x, col_names)
    print_mean_std_by_var(norm_data_y, col_names_y, col_names)
    print("Raw Mean Std by var")

    print_mean_std_by_var(raw_data_x, col_names_x+col_names_y+col_names_x, col_names)
    print_mean_std_by_var(raw_data_y, col_names_y, col_names)
    # print("qtend_check: ", norm_data_y.mean(), norm_data_y.std())
    print("qtend_check: ", norm_data_y.mean(), norm_data_y.std())

def test_image_multistep0():
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
    trainloader = data.DataLoader(training_set, shuffle=False, batch_size=1, num_workers=1, collate_fn=filter_collate)
    norm_data_x = []
    norm_data_y = []
    # start_idx, end_idx = get_index_from_colnames(col_names, "dqvls_nn_in")
    for idx, batch in enumerate(trainloader):
        if batch is None:
            continue
        x, y = batch[:2] # x: (batch, n_samples, n_features, lat, lon)
        filenames = batch[-1]
        print(idx, x.size(), y.size(), filenames)
        norm_data_x.append(x.numpy())
        norm_data_y.append(x.numpy())
    norm_data_x = np.concatenate(norm_data_x, axis=0)
    norm_data_y = np.concatenate(norm_data_y, axis=0)
    # print("Q: ", norm_data_x[:,:30].mean(), norm_data_x[:,:30].std())
    # print("T: ", norm_data_x[:,30:60].mean(), norm_data_x[:,30:60].std())
    # print("dqvls: ", norm_data_x[:,60:90].mean(), norm_data_x[:,60:90].std())
    # print("dTls: ", norm_data_x[:,90:120].mean(), norm_data_x[:,90:120].std())
    # print("SOLIN: ", norm_data_x[:,120].mean(), norm_data_x[:,120].std())
    # print("SPPS: ", norm_data_x[:,121].mean(), norm_data_x[:,121].std())
    print_mean_std_by_var(norm_data_x, col_names_x, col_names)
    print("qtend_check: ", norm_data_y.mean(), norm_data_y.std())


def test_image_multistep1():
    file_names = glob.glob(data_dir + "*.npy")
    file_names.sort()
    # file_names = file_names[:10]
    # file_names = [data_dir + fn for fn in file_names]
    print("file_names:", file_names[:10])
    col_names = np.loadtxt(data_dir + "/col_names.txt", dtype=str)
    #col_names_x = ["QL", "T_nn_in", "dqvls_nn_in", "dTls_nn_in", "SOLIN", "SPPS"]
    col_names_x = []
    col_names_y = ["qtend_check"]
    # varaibles that are used as input in the previous time step
    #prev_ex_vars = ["qtend_check", "stend_check", "SOLL", "SOLLD", "SOLS", "SOLSD", "FSDS"]
    prev_ex_vars = ["qtend_check"]

    data_means = dict(np.load(data_dir + "/data_means.npz"))
    data_stds = dict(np.load(data_dir + "/data_stds.npz"))
    # col_names_x = col_names_x + prev_ex_vars + col_names_x

    input_indices, prev_input_indices, output_indices = gen_multistep_col_indices(
        col_names, prev_ex_vars, col_names_x, col_names_y, 1)
    print("input_indices:", col_names[input_indices])
    print("prev_input_indices:", col_names[prev_input_indices])
    print("output_indices:", col_names[output_indices])

    transform = transforms.Compose([
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
        # FlattenSpatialTransform()
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
    trainloader = data.DataLoader(training_set, shuffle=False, batch_size=1, num_workers=1, collate_fn=filter_collate)
    norm_data_x = []
    norm_data_y = []
    # start_idx, end_idx = get_index_from_colnames(col_names, "dqvls_nn_in")
    for idx, batch in enumerate(trainloader):
        if batch is None:
            continue
        x, y, x_raw, y_raw = batch[:4]
        filenames = batch[-1]
        print(idx, x.size(), y.size(), x_raw.size(), y_raw.size(), filenames)
        norm_data_x.append(x.numpy())
        norm_data_y.append(x.numpy())
    norm_data_x = np.concatenate(norm_data_x, axis=0)
    norm_data_y = np.concatenate(norm_data_y, axis=0)
    # cur_idx = 0
    # for col in col_names_x + prev_ex_vars + col_names_x:
    #     start_idx, end_idx = get_index_from_colnames(col_names, col)
    #     data_range = end_idx - start_idx
    #     print(col, norm_data_x[:,cur_idx:cur_idx+data_range].mean(), \
    #           norm_data_x[:,cur_idx:cur_idx+data_range].std())
    #     cur_idx += data_range
    print_mean_std_by_var(norm_data_x, col_names_x + prev_ex_vars + col_names_x, col_names)
    print("qtend_check: ", norm_data_y.mean(), norm_data_y.std())

if __name__ == "__main__":
    #test_single_column_multistep0_minmax()
    test_single_column_multistep1_minmax()
    #test_single_column_multistep0_ex()
    #test_single_column_multistep1_ex()
    # test_single_column_multistep1()
    # test_image_multistep0()
    # test_single_column_multistep1()
    # test_image_multistep1()

