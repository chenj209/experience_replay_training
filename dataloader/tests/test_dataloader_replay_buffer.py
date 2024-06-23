import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "utils"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "training_scripts"))
import os
import glob
import argparse
import numpy as np
import torchvision.transforms as transforms
from torch.utils import data

from preprocess import FlattenSpatialTransform, MinMaxTransformLegacy, StandardizeTransform
from dataloader_utils import gen_multistep_col_indices, get_index_from_colnames, gen_col_indices
from dataloader_legacy_format import DatasetDisk, filter_collate
from debug_utils import print_mean_std_by_var, print_min_max_by_var
from replay_buffer import ReplayBuffer
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
        include_filename=True,
        include_idx=True
        )
    

    trainloader = data.DataLoader(training_set, shuffle=True, batch_size=2, num_workers=1, collate_fn=filter_collate)
    replay_buffer = ReplayBuffer(training_set, inp_shape=[96*144, 65], max_size=300, weighted=False)
    for _ in range(2):
        # get next item in trainloader
        batch = next(iter(trainloader))
        x, y = batch[:2]
        data_idx = batch[-2]
        pred_data = y*(-1)
        test_data_idx = []
        sampled_files = batch[-1][-1]
        for sf in sampled_files:
            test_data_idx.append(training_set.file_names.index(sf))
        print(f"Saving pred_data: {pred_data.shape}, data_idx: {data_idx}, \
            filenames: {batch[-1]}, index in files: {test_data_idx}")
        replay_buffer.store(pred_data, data_idx)
    replay_sample = replay_buffer.sample(2)
    norm_data_x = []
    norm_data_y = []
    raw_data_x = []
    raw_data_y = []
    
    trainloader = data.DataLoader(training_set, shuffle=True, batch_size=2, num_workers=1, collate_fn=filter_collate)
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
        dl_idx = batch[-2]
        print(idx, x.size(), y.size(), x_raw.size(), y_raw.size(), filenames, dl_idx)
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
    print_mean_std_by_var(norm_data_x, col_names_prev + col_names_x, col_names)
    print_mean_std_by_var(norm_data_y, col_names_y, col_names)
    print("Raw Mean Std by var")
    print_mean_std_by_var(raw_data_x, col_names_prev + col_names_x, col_names)
    print_mean_std_by_var(raw_data_y, col_names_y, col_names)


    

if __name__ == "__main__":
    #test_single_column_multistep0_ex()
    test_single_column_multistep1_ex()
    # test_single_column_multistep1()
    # test_image_multistep0()
    # test_single_column_multistep1()
    # test_image_multistep1()

