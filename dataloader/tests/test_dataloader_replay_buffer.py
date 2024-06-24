import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "utils"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "training_scripts"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "models"))
import os
import glob
import argparse
import numpy as np
import torchvision.transforms as transforms
from torch.utils import data
from tqdm.autonotebook import tqdm
import torch

from preprocess import FlattenSpatialTransform, MinMaxTransformLegacy, StandardizeTransform
from dataloader_utils import gen_multistep_col_indices, get_index_from_colnames, gen_col_indices
from dataloader_stride import DatasetDisk, filter_collate
from debug_utils import print_mean_std_by_var, print_min_max_by_var
from load_models import load_resmlp_newformat2
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
        sample_stride=12,
        is_train=True,
        transform=transform,
        include_filename=True,
        include_idx=True
        )
    

    trainloader = data.DataLoader(training_set, shuffle=True, batch_size=2, num_workers=1, collate_fn=filter_collate)
    replay_buffer = ReplayBuffer(training_set, inp_shape=[96*144, 65], max_size=300, weighted=False)
    for batch in trainloader:
        # get next item in trainloader
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
    # replay_sample = replay_buffer.sample(2)
    trainloader = data.DataLoader(training_set, shuffle=True, batch_size=2, num_workers=1, collate_fn=filter_collate)
    for _ in tqdm(range(20), desc="Verify replay buffer sample"):
        input_data, target_data, tar_idx = replay_buffer.sample(2)
        for i in range(2):
            id = input_data[i]
            td = target_data[i]
            ti = tar_idx[i][0]
            data_to_compare = training_set.get_index(ti)
            dx, dy = data_to_compare[:2]
            id[:,185:185+65] *= -1
            assert np.allclose(dx, id), "Input data not equal"
            assert np.allclose(dy, td), "Target data not equal"

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
        print("x shape: ", x.shape, ", filenames: ", batch[-1])
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
    print_mean_std_by_var(norm_data_x, col_names_x + col_names_prev + col_names_x, col_names)
    print_mean_std_by_var(norm_data_y, col_names_y, col_names)
    print("Raw Mean Std by var")
    print_mean_std_by_var(raw_data_x, col_names_x + col_names_prev + col_names_x, col_names)
    print_mean_std_by_var(raw_data_y, col_names_y, col_names)

def test_single_column_multistep1_ex_model_pred(args):
    input_size = 309
    all_models = {
        '0_29': load_resmlp_newformat2(args.model029, input_size, 30),
        '30_59': load_resmlp_newformat2(args.model3059, input_size, 30),
        '61_65': load_resmlp_newformat2(args.model6165, input_size, 5)
    }
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
        sample_stride=12,
        is_train=True,
        transform=transform,
        include_filename=True,
        include_idx=True
        )
    

    trainloader = data.DataLoader(training_set, shuffle=True, batch_size=2, num_workers=1, collate_fn=filter_collate)
    replay_buffer = ReplayBuffer(training_set, inp_shape=[96*144, 65], max_size=300, weighted=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for batch in trainloader:
        # get next item in trainloader
        x, y = batch[:2]
        data_idx = batch[-2]
        test_data_idx = []
        batch_size = x.shape[0]
        points_x = x.reshape(-1, x.shape[-1]).float().to(device)
        pred029 = all_models["0_29"](points_x).detach().cpu().numpy().reshape(batch_size,96*144,30)
        pred3059 = all_models["30_59"](points_x).detach().cpu().numpy().reshape(batch_size,96*144,30)
        pred6165 = all_models["61_65"](points_x).detach().cpu().numpy().reshape(batch_size,96*144,5)
        model_preds = np.concatenate([pred029, pred3059, pred6165], axis=2)
        print(model_preds.shape)
        # sampled_files = batch[-1][-1]
        print(f"Saving pred_data: {model_preds.shape}, data_idx: {data_idx}, \
            filenames: {batch[-1]}, index in files: {test_data_idx}")
        replay_buffer.store(model_preds, data_idx)
    # replay_sample = replay_buffer.sample(2)
    trainloader = data.DataLoader(training_set, shuffle=True, batch_size=2, num_workers=1, collate_fn=filter_collate)
    replay_sample_x = []
    replay_sample_y = []
    norm_data_x = []
    norm_data_y = []
    for _ in tqdm(range(20), desc="Verify replay buffer sample"):
        input_data, target_data, tar_idx = replay_buffer.sample(2)
        for i in range(2):
            id = input_data[i]
            td = target_data[i]
            ti = tar_idx[i][0]
            print(f"sample x shape: {id.shape}, y shape: {td.shape}")
            data_to_compare = training_set.get_index(ti)
            dx, dy = data_to_compare[:2]
            print(f"compare x shape: {dx.shape}, y shape: {dy.shape}")
            replay_sample_x.append(id)
            replay_sample_y.append(td)
            norm_data_x.append(dx)
            norm_data_y.append(dy)

    
    norm_data_x = np.concatenate(norm_data_x, axis=0)
    norm_data_y = np.concatenate(norm_data_y, axis=0)
    sample_data_x = np.concatenate(replay_sample_x, axis=0)
    sample_data_y = np.concatenate(replay_sample_y, axis=0)
    print("norm_data_x shape: ", norm_data_x.shape)
    print("Mean Std by var")
    print_mean_std_by_var(norm_data_x, col_names_x + col_names_prev + col_names_x, col_names)
    print_mean_std_by_var(norm_data_y, col_names_y, col_names)
    print("Sample Mean Std by var")
    print_mean_std_by_var(sample_data_x, col_names_x + col_names_prev + col_names_x, col_names)
    print_mean_std_by_var(sample_data_y, col_names_y, col_names)

    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model029", type=str, default="../training_scripts/ckpts_sampled12/conv_mem/baseline_model029_sampled12_multistep1_EX__EXPREV__region_all/checkpoint_epoch35.pth.tar")
    parser.add_argument("--model3059", type=str, default="../training_scripts/ckpts_sampled12/conv_mem/baseline_model3059_sampled12_multistep1_EX__EXPREV__region_all/checkpoint_epoch35.pth.tar")
    parser.add_argument("--model6165", type=str, default="../training_scripts/ckpts_sampled12/conv_mem/baseline_model6165_sampled12_multistep1_EX__EXPREV__region_all/checkpoint_epoch35.pth.tar")
    args = parser.parse_args()
    #test_single_column_multistep0_ex()
    #test_single_column_multistep1_ex()
    test_single_column_multistep1_ex_model_pred(args)
    # test_single_column_multistep1()
    # test_image_multistep0()
    # test_single_column_multistep1()
    # test_image_multistep1()

