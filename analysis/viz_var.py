import sys
import os
import numpy as np
import xgboost as xgb
from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils import data
import matplotlib.pyplot as plt
from joblib import dump

sys.path.append(os.path.join(sys.path[0], "..", "dataloader"))
from dataloader_newformat import DatasetDisk, filter_collate
sys.path.append(os.path.join(sys.path[0], "..", "utils"))
from data_shape import to_inference_shape

if __name__ == "__main__":
    import os
    import glob
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("var_name", type=str)
    parser.add_argument("--out_name", type=str)
    parser.add_argument("--sample_rate", type=int, default=12)
    args = parser.parse_args()
    data_dir = "/home/users/data/nncam_data/image_set/"
    if not os.path.isdir(data_dir):
        data_dir = "/data/nncam_data/image_set/"
    if not os.path.isdir(data_dir):
        # data_dir = "./data/"
        data_dir = "/pscratch/sd/c/chenjd21/spcam_new_data/"
        #data_dir = "./test_data/"
    if not os.path.isdir(data_dir):
        # data_dir = "./data/"
        #data_dir = "/pscratch/sd/c/chenjd21/spcam_new_data/"
        data_dir = "./test_data/"
    pred_dir = "/pscratch/sd/c/chenjd21/resmlp_pred/"
    if not os.path.isdir(pred_dir):
        pred_dir = "./pred_data/"
    all_files = glob.glob(data_dir + "*.npy")
    all_files.sort()
    # file_names = file_names[35041:35041+17530:12]
    # file_names = [data_dir + fn for fn in file_names]
    print(all_files[:10])
    col_names = np.loadtxt(data_dir + "/col_names.txt", dtype=str)
    col_names_x = [args.var_name]
    #col_names_x = ["QL", "T_nn_in", "dqvls_nn_in", "dTls_nn_in", "SOLIN", "SPPS"]
    #col_names_y = ["qtend_check", "stend_check", "SOLL", "SOLLD", "SOLS", "SOLSD", "FSDS"]
    col_names_y = ["qtend_check"]
    # filter problematic files
    filter_set = DatasetDisk(all_files, col_names, col_names_x, col_names_y, is_train=True, noise_std=0, filename=True, input_normalized=False, output_normalized=False, multistep=0, image=True, sample_rate=args.sample_rate)
    filterloader = data.DataLoader(filter_set, shuffle=False, batch_size=1, num_workers=1, collate_fn=filter_collate)
    problem_files = []
    vars = []
    for idx, batch in enumerate(filterloader):
        x, y, x_raw, file_names = batch
        print(f"{idx}/{len(filterloader)},", "x:", x.size(), "y:", y.size(), "x_raw:", x_raw.size(), file_names, flush=True)
        fn = file_names[0][-1].split("/")[-1]
        bad_file_flag = False
        preds = to_inference_shape(np.load(pred_dir + file_names[0][-1].split("/")[-1])[None,])
        print("preds:", preds.shape)
        r2 = r2_score(y[0], preds, multioutput="variance_weighted")
        if r2 < 0:
            problem_files.append(file_names[0][-1])
        else:
            vars.append(x_raw)
    print("problem_files:", problem_files)
    vars = np.concatenate(vars, axis=0)
    np.save(f"{args.out_name}.npy", vars)
