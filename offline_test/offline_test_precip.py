import sys
import argparse
import yaml
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import re
import torch.backends.cudnn as cudnn
import torch.optim as optim
import numpy as np
import json
import time
import glob
from torch.utils import data
from torchvision.transforms import Compose
from sklearn.metrics import r2_score
import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from configs import *
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'consts'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'dataloader'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))
import phys_consts
from load_models import load_resmlp_newformat, load_resmlp_newformat2
#from dataloader_refactor import DatasetDisk
from dataloader_newformat import DatasetDisk, filter_collate
from dataloader_utils import gen_multistep_col_indices
from preprocess import MinMaxTransformLegacy, StandardizeTransform, \
FlattenSpatialTransform, get_min_max_coords
from normalization import get_inverse_newformat, inverse_data_var_names
# from dataloader_time_embedded import TimeDatasetDisk as DatasetDisk
from load_models import load_models
from metrics import get_thickness_from_ps_1d, report_metric, report_metric_vert, report_metric_spatial
sys.path.append(os.path.join(sys.path[0], '..', 'utils'))
from data_shape import to_inference_shape, inverse_to_inference_shape
from precip_analysis import compute_scores, get_precip_hist

def inverse_61_65(x):
    x[:,0] = (x[:,0]) * (1412 - 0)
    x[:,1] = (x[:,1]) * (1412 - 0)
    x[:,2] = (x[:,2]) * (1412 - 0)
    x[:,3] = (x[:,3]) * (1412 - 0)
    x[:,4] = (x[:,4]) * (1412 - 0)
    return x

def inverse_61_64(x):
    x[:,0] = (x[:,0]+1)/2*(332+53)-53       # flns
    x[:,1] = (x[:,1]+1)/2*(419-83)+83       # flnt
    x[:,2] = (x[:,2]+1)/2*(1063+2.13)-2.13  # fsns
    x[:,3] = (x[:,3]+1)/2*(1299)+0          # fsnt
    return x

legacy_inverse = {}
legacy_inverse['0_29']  = lambda x: (x+1)/2*(3.11e-6*2)-3.11e-6
legacy_inverse['30_59'] = lambda x: (x+1)/2*(3.63*2)-3.63
legacy_inverse['60']    = lambda x: (x+1)/2*(2.12e-6)
legacy_inverse['61_64'] = lambda x: inverse_61_64(x)
legacy_inverse['61_65'] = lambda x: inverse_61_65(x)

def inverse_output(args, y, data_means=None, data_stds=None):
    if args.norm_type == "std":
        if args.output_type == "0_29":
            y = y*data_stds["qtend_check"]+data_means["qtend_check"]
        elif args.output_type == "30_59":
            y = y*data_stds["stend_check"]+data_means["stend_check"]
        elif args.output_type == "61_65":
            # TODO: need to deal with 61_65 have different ordering
            raise NotImplementedError("61_65 is not implemented")
    elif args.norm_type == "minmax_legacy":
        if args.output_type == "0_29":
            y = legacy_inverse['0_29'](y)
        elif args.output_type == "30_59":
            y = legacy_inverse['30_59'](y)
        elif args.output_type == "61_65":
            raise NotImplementedError("61_65 is not implemented")
    return y

def filename_to_datetime(filename):
    ts = int(filename.split(".")[0])
    start_date = datetime.datetime(1998, 1, 1, 0, 0, 0)
    return start_date + datetime.timedelta(hours=(ts-1)*0.5)

#def offline_test(args, all_models, testloader, get_thickness, inverse_output, silent=False, save=False):
def offline_test(args, all_models, testloader, get_thickness, silent=False, save=False):
    problem_files = []
    test_time_begin = time.time()
    y_pred = []
    y_gt = []
    if args.region_mask == "all":
        region_mask = np.ones((1,1,96,144))
    else:
        region_mask = np.load(args.region_mask)[None, None, :, :]
    region_mask = to_inference_shape(region_mask)
    #print("region_mask shape: ", region_mask.shape)
    region_mask = region_mask.squeeze().astype(bool)
    best_loss = 1e10
    best_filenames = None
    worst_loss = 0
    worst_filenames = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.precip_analysis:
        ets_scores = []
        far_scores = []
        mar_scores = []
        hist_pred = []
        hist_gt = []
    for iter, batch in enumerate(testloader):
        # allow empty batch
        if batch[0].shape[0] == 0:
            continue
        # file_names = batch[-1]
        all_models['model'].eval()
        with torch.no_grad():
            points_x, points_y, x_raw, y_raw = batch[:4]
            print("DEBUG1210: batch[-1]", batch[-1])
            filenames = [batch[-1][-1][i].split("/")[-1] for i in range(len(batch[-1][-1]))]
            print("DEBUG1210: filenames", filenames)
            datetimes = [filename_to_datetime(filename) for filename in filenames]
            print("DEBUG1210: datetimes", datetimes)
            points_x, points_y = points_x.reshape(-1, points_x.shape[-1]), \
                points_y.reshape(-1, points_y.shape[-1])
            if args.no_prevQT:
                points_x = points_x[:, 60:]
            elif args.no_prevQTLS:
                points_x = points_x[:, 120:]
            if get_thickness is not None:
                x_raw = x_raw.permute((0,2,1))
                x_raw = x_raw.reshape(-1, x_raw.shape[-1])
                thickness = get_thickness(x_raw[:,-1].numpy())
            points_x = (points_x.float()).to(device)
            pred = all_models['model'](points_x).detach().cpu().numpy()
            points_y = points_y.cpu().numpy()

            if args.inverse_output:
                pred = inverse_output(args, pred)
                points_y = inverse_output(args, points_y)

            if get_thickness is not None:
                pred = pred*thickness*phys_consts.LATVAP
                points_y = points_y*thickness*phys_consts.LATVAP
            y_pred.append(pred)
            y_gt.append(points_y)
            loss = np.mean((pred - points_y)**2)
            if loss < best_loss:
                best_loss = loss
                best_filenames = batch[-1]
            if loss > worst_loss:
                worst_loss = loss
                worst_filenames = batch[-1]
            if args.precip_analysis:
                ets, far, mar = compute_scores(pred, points_y)
                ets_scores.append(ets)
                far_scores.append(far)
                mar_scores.append(mar)
                hist_pred.append(get_precip_hist(pred))
                hist_gt.append(get_precip_hist(points_y))
        #print(f"testing {iter}/{len(testloader)}, r2: {r2_score(points_y.flatten(), y1.flatten())}", end='\r')
        print(f"testing {iter}/{len(testloader)}, r2: {r2_score(points_y.flatten(), pred.flatten())}")
        # print(f"filename: {batch[-1]}, Q lev 0 mean std: {x_raw[0,0].mean()}, {x_raw[0,0].std()}")
    print(f"Best loss: {best_loss}, filenames: {best_filenames}")
    print(f"Worst loss: {worst_loss}, filenames: {worst_filenames}")

    np.save(
        f"ex_{output_name}_{args.out_json.rstrip('.json')}_avg_pred.npy",
        np.mean(np.concatenate([y[None,] for y in y_1], axis=0), axis=0)
        )
    np.save(
        f"ex_{output_name}_{args.out_json.rstrip('.json')}_avg_gt.npy",
        np.mean(np.concatenate([y[None,] for y in y_gt], axis=0), axis=0)
        )

    y_pred = np.concatenate(y_pred, axis=0)
    y_gt = np.concatenate(y_gt, axis=0)

    log = report_metric(y_gt, y_pred)
    log_lvl = report_metric_vert(y_gt, y_pred)
    if args.region_mask == "all":
        log_spatial = report_metric_spatial(y_gt, y_1, (y_1.shape[-1], 96, 144))
    del y_1

    res = {
        "log": log,
        "log_lvl": log_lvl,
    }
    if args.region_mask == "all":
        res["log_spatial"] = log_spatial
    return res, problem_files

def prep_dataloaders(
    args,
    test_files,
    input_indices,
    prev_input_indices,
    output_indices,
    transform,
    region_mask):

    testing_set = DatasetDisk(
        test_files,
        input_indices,
        prev_input_indices,
        output_indices,
        is_train=False,
        transform=transform,
        multistep=int(args.multistep),
        sample_rate=int(args.sample_rate),
        include_filename=True,
        region_mask1d=region_mask
        )
    testloader = data.DataLoader(testing_set, shuffle=False,
                                 batch_size=4,
                                 num_workers=4,
                                 collate_fn=filter_collate,
                                 pin_memory=True)

    return testloader

if __name__ == "__main__":
    import argparse
    import random
    #dh_settings.get_weight()
    #dh_settings.get_hyai_hybi()
    torch.multiprocessing.set_sharing_strategy('file_system')
    random.seed(0)
    np.random.seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_type", "-ot", help="choose from 0-29, 30-59, 61-65, or other single output")
    #parser.add_argument("--resume", "-re", help="path to selected model")
    parser.add_argument("resume", help="path to configuration file")
    parser.add_argument("out_json", help="path to output json file")
    parser.add_argument("--sample_rate", type=int, help="sample frequency to use", default=12)
    parser.add_argument("--thick", action="store_true")
    #parser.add_argument("--save_path", type=str)
    parser.add_argument("--region_mask", type=str, default="all")
    parser.add_argument("--start_ts", type=int, default=35040)
    parser.add_argument("--multistep", type=int, default=1)
    parser.add_argument("--ex_input", type=str, nargs="*", default=[])
    parser.add_argument("--ex_input_prev", type=str, nargs="*", default=[])
    parser.add_argument("--norm_type", type=str, help="choose from [std, minmax_legacy], default to std", default="std")
    parser.add_argument("--inverse_output", action="store_true", help="inverse the output to original scale")
    parser.add_argument("--data_means", type=str)
    parser.add_argument("--data_stds", type=str)
    parser.add_argument("--train_configs", type=str, nargs="?",
                        help="path to training configuration file, this overwrites \
                        all previous arguments if conflicts")
    parser.add_argument("--precip_analysis", action="store_true")    
    parser.add_argument("--no_prevQT", action="store_true")
    parser.add_argument("--no_prevQTLS", action="store_true")
    parser.add_argument("--legacy_order", action="store_true")
    args = parser.parse_args()
    print(args)
    # prioritize args from train_configs
    if args.train_configs is not None:
        with open(args.train_configs, "r") as f:
            train_configs = yaml.safe_load(f)
            for key in train_configs:
                if train_configs[key] is not None \
                    and key != "resume" \
                    and key != "sample_rate": # resume is always chosen from args.resume args
                    setattr(args, key, train_configs[key])
        print("After train_configs overwrite:", args)

    model_ckpt_path = args.resume
    np.random.seed(0)
    data_dir = DATA_DIR
    print("Test set path: ", data_dir)

    cudnn.benchmark = True
    col_names = np.loadtxt(data_dir + "/col_names.txt", dtype=str)
    col_names_x = ["QL", "T_nn_in", "dqvls_nn_in", "dTls_nn_in", "SOLIN", "SPPS"]+args.ex_input
    if args.legacy_order:
        prev_ex_vars = ["qtend_check", "stend_check", 'SOLL', 'SOLS', 'SOLSD', 'SOLLD', 'FSDS']+args.ex_input_prev
    else:
        prev_ex_vars = ["qtend_check", "stend_check", "SOLL", "SOLLD", "SOLS", "SOLSD", "FSDS"]+args.ex_input_prev
    # col_names_y = ["qtend_check"]
    if args.output_type == '0-29':
        #batch[1] = batch[1][:, :, :30]
        col_names_y = ["qtend_check"]
        output_size = 30
    elif args.output_type == '30-59':
        # batch[1] = batch[1][:, :, 30:60]
        col_names_y = ["stend_check"]
        output_size = 30
    elif args.output_type == '61-65':
        if args.legacy_order:
            col_names_y = ["SOLL","SOLS","SOLSD","SOLLD","FSDS"]
        else:
            col_names_y = ["SOLL","SOLLD","SOLS","SOLSD","FSDS"]
        output_size = 5
    else:
        col_names_y = [args.output_type]
        output_size = 1
    output_name = '_'.join(col_names_y)

    all_files = glob.glob(data_dir+'/*.npy')
    all_files.sort()
    # testing data starts from 35040
    # all files are formatted in name 00010.npy, find idx where name is 35040
    test_files = all_files[args.start_ts+1:]

    input_indices, prev_input_indices, output_indices = gen_multistep_col_indices(
        col_names, prev_ex_vars, col_names_x, col_names_y, int(args.multistep)
    )
    print("Input indices: ", col_names[input_indices])
    print("Prev input indices: ", col_names[prev_input_indices])
    print("Output indices: ", col_names[output_indices])

    multistep_col_names_x = []
    for i in range(int(args.multistep)):
        multistep_col_names_x.extend(col_names_x)
        multistep_col_names_x.extend(prev_ex_vars)
    multistep_col_names_x.extend(col_names_x)

    region_mask = None
    if args.region_mask is not None and args.region_mask != "all":
        region_mask = np.load(args.region_mask)
    else:
        region_mask = np.ones((96,144))
    min_x, max_x, min_y, max_y = get_min_max_coords(region_mask, 2)
    lon = np.linspace(0,357.5,144)
    lat = np.linspace(-90,90,96)
    print("Region window coordinates: ", lon[min_y], lon[max_y-1], lat[min_x], lat[max_x-1])
    sub_region_mask = region_mask[min_x:max_x, min_y:max_y]

    if args.thick:
        pconsts = np.load(os.path.join(sys.path[0],"..","consts","phys_consts.npz"))
        hyai = pconsts["hyai"]
        hybi = pconsts["hybi"]
        get_thickness = lambda x : get_thickness_from_ps_1d(x,hyai, hybi)
    else:
        get_thickness = None

    if args.norm_type == "std":
        data_means = dict(np.load(args.data_means))
        data_stds = dict(np.load(args.data_stds))
        transform = Compose([
            StandardizeTransform(
                data_means,
                data_stds,
                multistep_col_names_x,
                col_names_y,
                col_names,
                normalize_input=True,
                normalize_output=True,
                include_raw=True,
                threshold=1e10
                ),
            FlattenSpatialTransform(),
            ])
    elif args.norm_type == "minmax_legacy":
        transform = Compose([
            MinMaxTransformLegacy(include_raw=True),
            FlattenSpatialTransform(),
            ])
    else:
        raise ValueError("Invalid norm_type")
            

    testloader = prep_dataloaders(args, test_files, input_indices,
                                  prev_input_indices, output_indices,
                                  transform, region_mask)

    if args.thick:
        pconsts = np.load(os.path.join(sys.path[0],"..","consts","phys_consts.npz"))
        hyai = pconsts["hyai"]
        hybi = pconsts["hybi"]
        get_thickness = lambda x : get_thickness_from_ps_1d(x,hyai, hybi)
    else:
        get_thickness = None
    input_size = len(input_indices)+int(args.multistep)*len(prev_input_indices)
    if args.no_prevQTLS and args.no_prevQT:
        raise ValueError("no_prevQTLS and no_prevQT cannot be both True")
    if args.no_prevQT:
        input_size = input_size - 60
    if args.no_prevQTLS:
        input_size = input_size - 120
    # all_models = {'0_29': load_resmlp_newformat(model_ckpt_path, input_size)}
    all_models = {'model': load_resmlp_newformat2(model_ckpt_path, input_size, output_size)}


    logs, problem_files = offline_test(args, all_models, testloader, get_thickness)
    if len(problem_files) > 0:
        with open(f"{args.out_json.rstrip('.json')}_problem_files.txt", "w") as f:
            f.write(str(problem_files))

    with open(args.out_json, "w") as f:
        json.dump({
            args.output_type: logs["log"],
            }, f, indent=4)

    if args.region_mask == "all":
        np.savez(f"ex_{output_name}_{args.out_json.rstrip('.json')}_spatial.npz", **logs["log_spatial"])
    np.savez(f"ex_{output_name}_{args.out_json.rstrip('.json')}_vert.npz", **logs["log_lvl"])
