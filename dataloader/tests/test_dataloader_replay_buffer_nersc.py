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
from sklearn.metrics import r2_score

from preprocess import FlattenSpatialTransform, MinMaxTransformLegacy, StandardizeTransformNext
from dataloader_utils import gen_multistep_col_indices, get_index_from_colnames, gen_col_indices
from dataloader_replay_buffer_nersc import DatasetDisk, filter_collate
from debug_utils import print_mean_std_by_var, print_min_max_by_var
from load_models import load_resmlp_newformat2
from replay_buffer import ReplayBuffer
# import torch transforms

# data_dir = "/home/users/data/nncam_data/image_set/"
# data_dir = "/data/nncam_data/image_set/"
# ex_data_dir = "/data/chenj209/ex_dataset"
data_dir = "./spcam_new_data_local/"

def test_single_column_multistep1_nersc():
    file_names = glob.glob(data_dir + "/*.npy")
    file_names.sort()
    print(f"file_names in {data_dir}:", file_names[:10])
    col_names = np.loadtxt("col_names.txt", dtype=str)
    col_names_x = ["QL", "T_nn_in", "dqvls_nn_in", "dTls_nn_in", "SOLIN", "SPPS"]
    # col_names_x = ["QL", "T_nn_in", "dqvls_nn_in", "dTls_nn_in", "SOLIN", "SPPS"]
    col_names_y = ["qtend_check", "stend_check", "SOLL", "SOLS", "SOLSD", "SOLLD", "FSDS"]
    col_names_prev = ["qtend_check", "stend_check", "SOLL", "SOLS", "SOLSD", "SOLLD", "FSDS"]
    input_indices, prev_input_indices, output_indices = gen_multistep_col_indices(
        col_names, col_names_prev, col_names_x, col_names_y, 1)
    data_means = dict(np.load("../consts/all_means.npz"))
    data_stds = dict(np.load("../consts/all_stds.npz"))


    transform = transforms.Compose([
        StandardizeTransformNext(
            data_means,
            data_stds,
            col_names_x + col_names_prev + col_names_x,
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
        sample_stride=12,
        is_train=True,
        transform=transform,
        include_filename=True,
        include_idx=True
        )
    

    trainloader = data.DataLoader(training_set, shuffle=True, batch_size=2, num_workers=1, collate_fn=filter_collate)

    norm_data_x = []
    norm_data_y = []
    norm_x_next = []
    norm_y_next = []
    
    for idx, batch in enumerate(trainloader):
        if batch is None:
            continue
        print("batch items: ", len(batch))
        x, y = batch[:2] # x: (batch, n_samples, n_features)
        x_next, y_next = batch[2:4]
        # check X data is equal to X_next previous X
        assert np.allclose(x[:,:,122+65:], x_next[:, :, :122]), "X data is not equal to X_next previous X"
        # check Y data is equal to Y_next previous Y
        assert np.allclose(y[:,:,:], x_next[:,:,122:122+65]), "Y data is not equal to Y_next previous Y"
        print("x shape: ", x.shape, ", filenames: ", batch[-1])
        print("x_next shape: ", x_next.shape)
        print("y_next shape: ", y_next.shape)
        x = x.reshape(-1, x.shape[-1]) # x: (batch * n_samples, n_features)
        y = y.reshape(-1, y.shape[-1])
        filenames = batch[-1]
        dl_idx = batch[-2]
        print(idx, x.size(), y.size(), x_next.size(), y_next.size(), filenames, dl_idx)
        norm_data_x.append(x.numpy())
        norm_data_y.append(y.numpy())
        norm_x_next.append(x_next.numpy())
        norm_y_next.append(y_next.numpy())
    norm_data_x = np.concatenate(norm_data_x, axis=0)
    norm_data_y = np.concatenate(norm_data_y, axis=0)
    norm_data_x_next = np.concatenate(norm_x_next, axis=0)
    norm_data_y_next = np.concatenate(norm_y_next, axis=0)
    np.save("debug_norm_y", norm_data_y)
    np.save("debug_norm_y_next", norm_data_y_next)
    print("norm_data_x shape: ", norm_data_x.shape)
    print("Mean Std by var")
    print_mean_std_by_var(norm_data_x, col_names_x + col_names_prev + col_names_x, col_names)
    print_mean_std_by_var(norm_data_y, col_names_y, col_names)
    print("Mean Mean Std by var for Next")
    print_mean_std_by_var(norm_data_x_next, col_names_x + col_names_prev + col_names_x, col_names)
    print_mean_std_by_var(norm_data_y_next, col_names_y, col_names)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    test_single_column_multistep1_nersc()

