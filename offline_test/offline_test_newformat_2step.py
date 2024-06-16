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
from metrics import Regression_Metrics, Regression_Metrics_axis, reverse_operations, \
    report_qtend, report_stend, report_rad_prog, report_rad_prog_individual, \
    report_qtend_vert, report_stend_vert, report_qtend_spatial, report_stend_spatial, \
    get_thickness_from_ps_1d, report_qtend_vert_quantile, report_qtend_vert_tail, \
    report_metric, report_metric_vert, report_metric_spatial
sys.path.append(os.path.join(sys.path[0], '..', 'utils'))
from data_shape import to_inference_shape, inverse_to_inference_shape

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

inverse = {}
inverse['0_29']  = lambda x: (x+1)/2*(3.11e-6*2)-3.11e-6
inverse['30_59'] = lambda x: (x+1)/2*(3.63*2)-3.63
inverse['60']    = lambda x: (x+1)/2*(2.12e-6)
inverse['61_64'] = lambda x: inverse_61_64(x)
inverse['61_65'] = lambda x: inverse_61_65(x)

#def offline_test(args, all_models, testloader, get_thickness, inverse_output, silent=False, save=False):
def offline_test(args, all_models, testloader, get_thickness, silent=False, save=False):
    problem_files = []
    test_time_begin = time.time()
    epoch = 1
    criterion = nn.MSELoss()
    prev_preds = {
        '0_29': [],
        '30_59': [],
        '61_65': []
    }
    prev_gt = {
        '0_29': [],
        '30_59': [],
        '61_65': []
    }
    curr_preds = {
        '0_29': [],
        '30_59': [],
        '61_65': []
    }
    curr_gt = {
        '0_29': [],
        '30_59': [],
        '61_65': []
    }
    curr_preds_2step = {
        '0_29': [],
        '30_59': [],
        '61_65': []
    }
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
    for iter, batch in enumerate(testloader):
        # allow empty batch
        if batch[0].shape[0] == 0:
            continue
        suffix = 'testing- epoch:{}| iters:{}/{} |'.format(epoch, iter+1, len(testloader))
        # file_names = batch[-1]
        #model.eval()
        with torch.no_grad():
            points_x, points_y, x_raw, y_raw = batch[:4]
            # points_x is of 3 timestep (122+65+122+65+122)
            filename = batch[-1][0][0].split("/")[-1]
            points_x, points_y = points_x.reshape(-1, points_x.shape[-1]), \
                points_y.reshape(-1, points_y.shape[-1])

            prev_points_x = points_x[:, :309]
            previous_points_y = points_x[:, 309:309+65]
            prev_points_x = (prev_points_x.float()).to(device)
            prev_qtend = all_models["0_29"](prev_points_x)
            prev_stend = all_models["30_59"](prev_points_x)
            prev_rad = all_models["61_65"](prev_points_x)
            prev_preds["0_29"].append(prev_qtend.detach().cpu().numpy())
            prev_preds["30_59"].append(prev_stend.detach().cpu().numpy())
            prev_preds["61_65"].append(prev_rad.detach().cpu().numpy())
            prev_gt["0_29"].append(previous_points_y[:, :30].cpu().numpy())
            prev_gt["30_59"].append(previous_points_y[:, 30:60].cpu().numpy())
            prev_gt["61_65"].append(previous_points_y[:, 60:65].cpu().numpy())
            curr_points_x = points_x[:, -309:]
            curr_points_x = (curr_points_x.float()).to(device)
            curr_points_y = points_y
            curr_preds["0_29"].append(all_models["0_29"](curr_points_x).detach().cpu().numpy())
            curr_preds["30_59"].append(all_models["30_59"](curr_points_x).detach().cpu().numpy())
            curr_preds["61_65"].append(all_models["61_65"](curr_points_x).detach().cpu().numpy())
            curr_gt["0_29"].append(curr_points_y[:, :30].cpu().numpy())
            curr_gt["30_59"].append(curr_points_y[:, 30:60].cpu().numpy())
            curr_gt["61_65"].append(curr_points_y[:, 60:65].cpu().numpy())
            curr_points_x_2step = torch.cat([
                points_x[:-309:-(309+122)], 
                prev_qtend, prev_stend, prev_rad,
                points_x[:,-122:]
                ], dim=1)
            curr_preds_2step["0_29"].append(all_models["0_29"](curr_points_x_2step).detach().cpu().numpy())
            curr_preds_2step["30_59"].append(all_models["30_59"](curr_points_x_2step).detach().cpu().numpy())
            curr_preds_2step["61_65"].append(all_models["61_65"](curr_points_x_2step).detach().cpu().numpy())
            


            #points_y = points_y[0]
            #y1 = y1[0]
            #print("points_y:", points_y.shape, points_y.mean(), points_y.std())
            #print("y1:", y1.shape, y1.mean(), y1.std())
            if True:
                if iter % 200 == 0:
                    print(batch[-1])
                    filename = batch[-1][0][-1].split("/")[-1][:-4]
                    np.savez(f"offline_test_{filename}", gt=points_y, pred=y1)
            if get_thickness is not None:
                y1 = y1* thickness * phys_consts.LATVAP
                points_y = points_y*thickness*phys_consts.LATVAP
            y_1.append(y1)
            y_gt.append(points_y)
            loss = np.mean((y1 - points_y)**2)
            if loss < best_loss:
                best_loss = loss
                best_filenames = batch[-1]
            if loss > worst_loss:
                worst_loss = loss
                worst_filenames = batch[-1]
        #print(f"testing {iter}/{len(testloader)}, r2: {r2_score(points_y.flatten(), y1.flatten())}", end='\r')
        print(f"testing {iter}/{len(testloader)}, r2: {r2_score(points_y.flatten(), y1.flatten())}")
        print(f"filename: {batch[-1]}, Q lev 0 mean std: {x_raw[0,0].mean()}, {x_raw[0,0].std()}")
    print(f"Best loss: {best_loss}, filenames: {best_filenames}")
    print(f"Worst loss: {worst_loss}, filenames: {worst_filenames}")

    np.save(f"ex_{output_name}_{args.out_json.rstrip('.json')}_avg_pred.npy", np.mean(np.concatenate([y[None,] for y in y_1], axis=0), axis=0))
    np.save(f"ex_{output_name}_{args.out_json.rstrip('.json')}_avg_gt.npy", np.mean(np.concatenate([y[None,] for y in y_gt], axis=0), axis=0))

    y_1 = np.concatenate(y_1, axis=0)
    # y_2 = np.concatenate(y_2, axis=0)
    #y_4 = np.concatenate(y_4, axis=0)
    y_gt = np.concatenate(y_gt, axis=0)

    test_time = time.time() - test_time_begin

    # qtend_log = report_qtend(y_gt, y_1)
    # qtend_log_lvl = report_qtend_vert(y_gt, y_1)
    log = report_metric(y_gt, y_1)
    log_lvl = report_metric_vert(y_gt, y_1)
    if False:
        for quantile in [0.5,0.7,0.9,1]:
            print(f"Quantile {quantile} results:")
            qtend_log_lvl = report_qtend_vert_quantile(y_gt, y_1, quantile)
            print(qtend_log_lvl["r2"])
        for tail in [0.1,0.2,0.3]:
            print(f"tail {tail} results:")
            qtend_log_lvl = report_qtend_vert_tail(y_gt, y_1, tail, top=True)
            print("Top:", qtend_log_lvl["r2"])
            qtend_log_lvl = report_qtend_vert_tail(y_gt, y_1, tail, top=False)
            print("Bottom:", qtend_log_lvl["r2"])
    if args.region_mask == "all":
        #qtend_log_spatial = report_qtend_spatial(y_gt, y_1)
        log_spatial = report_metric_spatial(y_gt, y_1, (y_1.shape[-1], 96, 144))
    # qtend_log_spatial = report_qtend_spatial(y_gt, y_1)
    #print(json.dumps(qtend_log_lvl, indent=4))
    del y_1

    res = {
        "log": log,
        # "stend_log": stend_log,
        #"rad_log": rad_log,
        #"rad_log_individual": rad_log_individual,
        "log_lvl": log_lvl,
        # "qtend_log_spatial": qtend_log_spatial,
        # "stend_log_lvl": stend_log_lvl,
        # "stend_log_spatial": stend_log_spatial
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
        multistep=3,
        sample_rate=int(args.sample_rate),
        include_filename=True,
        region_mask1d=region_mask
        )
    testloader = data.DataLoader(testing_set, shuffle=False,
                                 batch_size=1,
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
    parser.add_argument("--data_means", type=str)
    parser.add_argument("--data_stds", type=str)
    parser.add_argument("--train_configs", type=str, nargs="?",
                        help="path to training configuration file, this overwrites \
                        all previous arguments if conflicts")
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

    #print(args.config)
    # with open(args.config, "r") as f:
        # config = json.load(f)
    # model_ckpt_path = config["0-29"]["ckpt_path"]
    model_ckpt_path = args.resume
    # all_models = load_models(
    #     config["0-29"]["ckpt_path"],
    #     # config["30-59"]["ckpt_path"],
    #     None,
    #     None,
    #     #config["61-65"]["ckpt_path"]
    #     None,
    #     model_type="resmlp_122_30"
    # )
    np.random.seed(0)
    #data_dir = "/home/users/data/nncam_data/image_testset/"
    #if not os.path.isdir(data_dir):
    #    data_dir = "/data/nncam_data/image_testset/"
    #if not os.path.isdir(data_dir):
    #    #data_dir = "/global/cfs/cdirs/m4359/zhangtao/nncam/image_testset/"
    #    data_dir = "/pscratch/sd/c/chenjd21/spcam_new_data_32/"
    data_dir = DATA_DIR
    print("Test set path: ", data_dir)

    cudnn.benchmark = True
    col_names = np.loadtxt(data_dir + "/col_names.txt", dtype=str)
    col_names_x = ["QL", "T_nn_in", "dqvls_nn_in", "dTls_nn_in", "SOLIN", "SPPS"]+args.ex_input
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
        #batch[1] = batch[1][:, :, 61:66]
        #col_names_y = ["SOLL","SOLLD","SOLS","SOLSD","FSDS"]
        col_names_y = ["SOLL","SOLS","SOLSD","SOLLD","FSDS"]
        output_size = 5
#             if args.output_type == '61-65':
    else:
        col_names_y = [args.output_type]
        output_size = 1
    output_name = '_'.join(col_names_y)
    #data_means = dict(np.load(data_dir + "/data_means.npz"))
    #data_stds = dict(np.load(data_dir + "/data_stds.npz"))

    all_files = glob.glob(data_dir+'/*.npy')
    all_files.sort()
    # testing data starts from 35040
    # all files are formatted in name 00010.npy, find idx where name is 35040
    test_files = all_files[args.start_ts:]

    input_indices, prev_input_indices, output_indices = gen_multistep_col_indices(
        col_names, prev_ex_vars, col_names_x, col_names_y, multistep=3
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
    # all_models = {'0_29': load_resmlp_newformat(model_ckpt_path, input_size)}
    all_models = {'model': load_resmlp_newformat2(model_ckpt_path, input_size, output_size)}


    logs, problem_files = offline_test(args, all_models, testloader, get_thickness)
    if len(problem_files) > 0:
        with open(f"{args.out_json.rstrip('.json')}_problem_files.txt", "w") as f:
            f.write(str(problem_files))

    with open(args.out_json, "w") as f:
        json.dump({
            # "dq/dt": logs["qtend_log"],
            args.output_type: logs["log"],
            #"dT/dt": logs["stend_log"],
            #"radiation": rad_log,
            #"dqdt_lvl": qtend_log_lvl,
            #"dTdt_lvl": stend_log_lvl,
            #**logs["rad_log_individual"]
            }, f, indent=4)
    # with open("ex_"+args.out_json, "w") as f:
    #     json.dump({
    #         #"dq/dt": qtend_log,
    #         #"dT/dt": stend_log,
    #         "radiation": logs["rad_log"],
    #         "dqdt_lvl": logs["qtend_log_lvl"],
    #         "dTdt_lvl": logs["stend_log_lvl"],
    #         #**rad_log_individual
    #         }, f, indent=4)

    if args.region_mask == "all":
        np.savez(f"ex_{output_name}_{args.out_json.rstrip('.json')}_spatial.npz", **logs["log_spatial"])
    #np.savez(f"ex_stend_{args.out_json.rstrip('.json')}_spatial.npz", **logs["stend_log_spatial"])
    np.savez(f"ex_{output_name}_{args.out_json.rstrip('.json')}_vert.npz", **logs["log_lvl"])
    #np.savez(f"ex_stend_{args.out_json.rstrip('.json')}_vert.npz", **logs["stend_log_lvl"])





