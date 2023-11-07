import os
import sys
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import numpy as np
from torch.utils import data
import json
import time
import glob
sys.path.append(os.path.join(sys.path[0], "utils"))
import argparse
sys.path.append(os.path.join(sys.path[0], "dataloader"))
from dataloader_refactor import TimeDatasetDisk, get_inverse
sys.path.append(os.path.join(sys.path[0], "models"))
import models

HYAI = np.load(os.path.join(sys.path[0], "consts", "hyai.npy"))
HYBI = np.load(os.path.join(sys.path[0], "consts", "hybi.npy"))
LATVAP = 2.5104e6

def get_thickness_from_ps_1d(ps):
    """
    ps: (N): shape N samples
    """
    GRAVIT = 9.8066

    hybi_diff = np.diff(HYBI)
    hyai_diff = np.diff(HYAI)
    thick = ps[:, np.newaxis] * hybi_diff[np.newaxis, :]
    thick += hyai_diff[np.newaxis, :] * 100000
    thick /= GRAVIT
    #return thick.compressed().reshape(-1, 30)
    return thick


def load_models(model029, model3059, model6164, model6165, silent=False):
    """
    Load four models from given ckpt path

    Parameters:
      model029(str): path to ckpt file of model for 0-29
      model3059(str): path to ckpt file of model for 30-59
      model6164(str): path to ckpt file of model for 61-64
      model6165(str): path to ckpt file of model for 61-65
    """
    # define model
    all_models = {}
    # hyperparameters (fixed)
    input_size = 122
    num_blocks = 7
    node_size  = 512
    activation = 'relu'
    dropout    = 0
    # Assign GPUs to 3 models
    gpu_index = 0


    for output_type in ['0_29', '30_59', '61_64', '61_65']:
        if output_type == '0_29' or output_type == '30_59':
            model = models.ResMLP(input_size, 30, node_size, activation, num_blocks)
        if not silent:
            print("\nLoading DNN model to GPU_{}".format(gpu_index))
        if model is not None:
            model = torch.nn.DataParallel(model, device_ids=[gpu_index])

        if not silent:
            print('------------------------output type: {}-----------------------'.format(output_type))
            print('Total number of params: {}'.format(sum(p.numel() for p in model.parameters())))

        if output_type == '0_29':
            #resume = path + 'rmbaddata_wxnorm_subset_mlp_0-29_mlp_output30_nodesize512_num_blocks7_actrelu
            #resume = path + 'rmbaddata_wxnorm_subset_mlp_0-29_mlp_output30_nodesize512_num_blocks7_actrelu
            resume = model029
            if not silent:
                print('\nloading the checkpoint: {}'.format(resume))

        if output_type == '30_59':
            #resume = path + 'rmbaddata_wxnorm_subset_30-59_resnet_output30_nodesize512_num_blocks7_actrelu
            #resume = path + 'rmbaddata_wxnorm_subset_30-59_resnet_output30_nodesize512_num_blocks7_actrelu
            resume = model3059
            if not silent:
                print('\nloading the checkpoint: {}'.format(resume))

        elif output_type == '61_64':

            resume = model6164
            if not silent:
                print('\nloading the checkpoint: {}'.format(resume))

        elif output_type == '61_65':

            resume = model6165
            if not silent:
                print('\nloading the checkpoint: {}'.format(resume))

        if resume is not None:
            checkpoint = torch.load(resume)['state_dict']
            model.load_state_dict(checkpoint)

        all_models[output_type] = model
        gpu_index = gpu_index + 1
    return all_models




def Regression_Metrics(y_true, y_pred):
    
    var  = np.var(y_true)
    std  = np.std(y_true)
    
    mse  = np.mean((y_true-y_pred)**2)
    rmse = mse**0.5
       
    mae    = np.mean(np.absolute(y_pred-y_true))  # Mean absolute error
    me    = np.mean(y_pred-y_true)  # Mean absolute error
    max_ae = np.max(np.absolute(y_pred-y_true))   # Max absolute error
    
    bias = np.mean(y_pred-y_true)
    r2   = 1 - mse/var 
    
    return float(var), float(std), float(mse), float(rmse), float(mae), float(max_ae), float(bias), float(r2), float(me)

def report_stend_vert(y_gt, stend):
    out =  {
            "var": [],
            "std": [], 
            "mse": [], 
            "rmse": [], 
            "mae": [], 
            "max_ae": [], 
            "bias": [], 
            "r2": [],
            "me": []
          }
    for i in range(30):
      #var, std, mse, rmse, mae, max_ae, bias, r2 = Regression_Metrics(y_gt[:,30+i:30+i+1]*3600*24/1004, stend[:,i:i+1]*3600*24/1004)
        var, std, mse, rmse, mae, max_ae, bias, r2, me = Regression_Metrics(y_gt[:,30+i:30+i+1], stend[:,i:i+1])
        stend_metrics = '{}\t{:.4}\t{:.4}\t{:.4}({:.4%})\t{:.4}\t{:.4}\t{:.4}({:.4%})\t'.format(50, r2, mse, rmse, rmse/std, mae, max_ae, bias, bias/std)
        stend_log = ""
        stend_log += f"stend lvl {i} metrics:\n"
        stend_log += "epoch\tr2\tmse\trmse\t\tmae\tmax_ae\tbias\n"
        stend_log += stend_metrics
        stend_log += "\n"
        #print(stend_log)
        out["var"].append(var)
        out["std"].append(std)
        out["mse"].append(mse)
        out["rmse"].append(rmse)
        out["mae"].append(mae)
        out["max_ae"].append(max_ae)
        out["bias"].append(bias)
        out["r2"].append(r2)
        out["me"].append(me)
    return out

def report_stend(y_gt, stend):
  #var, std, mse, rmse, mae, max_ae, bias, r2 = Regression_Metrics(y_gt[:,30:60]*3600*24/1004, stend*3600*24/1004)
    var, std, mse, rmse, mae, max_ae, bias, r2, me = Regression_Metrics(y_gt[:,30:60], stend)
    stend_metrics = '{}\t{:.4}\t{:.4}\t{:.4}({:.4%})\t{:.4}\t{:.4}\t{:.4}({:.4%})\t'.format(50, r2, mse, rmse, rmse/std, mae, max_ae, bias, bias/std)
    stend_log = ""
    stend_log += "stend metrics:\n"
    stend_log += "epoch\tr2\tmse\trmse\t\tmae\tmax_ae\tbias\n"
    stend_log += stend_metrics
    stend_log += "\n"
    #print(stend_log)
    return {
            "var": var,
            "std": std, 
            "mse": mse, 
            "rmse": rmse, 
            "mae": mae, 
            "max_ae": max_ae, 
            "bias": bias, 
            "r2": r2,
            "me": me
          }

def report_qtend_vert(y_gt, qtend):
    out =  {
            "var": [],
            "std": [], 
            "mse": [], 
            "rmse": [], 
            "mae": [], 
            "max_ae": [], 
            "bias": [], 
            "r2": [],
            "me": []
          }
    for i in range(30):
      #var, std, mse, rmse, mae, max_ae, bias, r2 = Regression_Metrics(y_gt[:,0+i:0+i+1]*3600*24*1000, qtend[:,i:i+1]*3600*24*1000)
        var, std, mse, rmse, mae, max_ae, bias, r2, me = Regression_Metrics(y_gt[:,0+i:0+i+1], qtend[:,i:i+1])
        if std == 0:
            std = 1e-16
        qtend_metrics = '{}\t{:.4}\t{:.4}\t{:.4}({:.4%})\t{:.4}\t{:.4}\t{:.4}({:.4%})\t'.format(50, r2, mse, rmse, rmse/std, mae, max_ae, bias, bias/std)
        qtend_log = ""
        qtend_log += f"qtend lvl {i} metrics:\n"
        qtend_log += "epoch\tr2\tmse\trmse\t\tmae\tmax_ae\tbias\n"
        qtend_log += qtend_metrics
        qtend_log += "\n"
        #print(qtend_log)
        out["var"].append(var)
        out["std"].append(std)
        out["mse"].append(mse)
        out["rmse"].append(rmse)
        out["mae"].append(mae)
        out["max_ae"].append(max_ae)
        out["bias"].append(bias)
        out["r2"].append(r2)
        out["me"].append(me)
    return out

def report_qtend(y_gt, qtend):
  #var, std, mse, rmse, mae, max_ae, bias, r2 = Regression_Metrics(y_gt[:,0:30]*3600*24*1000, qtend*3600*24*1000)
    var, std, mse, rmse, mae, max_ae, bias, r2, me = Regression_Metrics(y_gt[:,0:30], qtend)
    qtend_metrics = '{}\t{:.4}\t{:.4}\t{:.4}({:.4%})\t{:.4}\t{:.4}\t{:.4}({:.4%})\t'.format(50, r2, mse, rmse, rmse/std, mae, max_ae, bias, bias/std)
    qtend_log = ""
    qtend_log += "qtend metrics:\n"
    qtend_log += "epoch\tr2\tmse\trmse\t\tmae\tmax_ae\tbias\n"
    qtend_log += qtend_metrics
    qtend_log += "\n"
    print(qtend_log)
    return {
            "var": var,
            "std": std, 
            "mse": mse, 
            "rmse": rmse, 
            "mae": mae, 
            "max_ae": max_ae, 
            "bias": bias, 
            "r2": r2,
            "me": me
          }

def report_rad_prog(y_gt, rad_prog):
    var, std, mse, rmse, mae, max_ae, bias, r2, me = Regression_Metrics(y_gt[:,61:66], rad_prog)
    rad_prog = '{}\t{:.4}\t{:.4}\t{:.4}({:.4%})\t{:.4}\t{:.4}\t{:.4}({:.4%})\t'.format(50, r2, mse, rmse, rmse/std, mae, max_ae, bias, bias/std)
    rad_log = ""
    rad_log += "rad metrics:\n"
    rad_log += "epoch\tr2\tmse\trmse\t\tmae\tmax_ae\tbias\n"
    rad_log += rad_prog
    rad_log += "\n"
    #print(rad_log)
    return {
            "var": var,
            "std": std, 
            "mse": mse, 
            "rmse": rmse, 
            "mae": mae, 
            "max_ae": max_ae, 
            "bias": bias, 
            "r2": r2,
            "me": me
          }

def report_rad_prog_individual(y_gt, rad_pred):
    rad_vars = [
     "soll",
     "sols", 
     "solsd",
     "solld",
     "fsds"
    ]
    out = {}
    for i in range(len(rad_vars)):
        var, std, mse, rmse, mae, max_ae, bias, r2, me = Regression_Metrics(y_gt[:,61+i:61+i+1], rad_pred[:,i:i+1])
        rad_prog = '{}\t{:.4}\t{:.4}\t{:.4}({:.4%})\t{:.4}\t{:.4}\t{:.4}({:.4%})\t'.format(50, r2, mse, rmse, rmse/std, mae, max_ae, bias, bias/std)
        rad_log = ""
        rad_log += f"rad {rad_vars[i]} metrics:\n"
        rad_log += "epoch\tr2\tmse\trmse\t\tmae\tmax_ae\tbias\n"
        rad_log += rad_prog
        rad_log += "\n"
        #print(rad_log)
        out[rad_vars[i]] = {
          "var": var,
          "std": std, 
          "mse": mse, 
          "rmse": rmse, 
          "mae": mae, 
          "max_ae": max_ae, 
          "bias": bias, 
          "r2": r2,
          "me": me
        }
    return out

def offline_test(args, all_models, testloader, silent=False):

    #test_losses = AverageMeter()
    #loss_name = [output_type + '_r2: {:.4e}']
    test_time_begin = time.time()
    epoch = 1
    criterion = nn.MSELoss()
    y_pred = []
    y_1 = []
    y_2 = []
    y_3 = []
    y_4 = []
    y_gt = []
    #for iter, batch in tqdm(enumerate(testloader), total=len(testloader)):
    for iter, batch in enumerate(testloader):
        suffix = 'testing- epoch:{}| iters:{}/{} |'.format(epoch, iter+1, len(testloader))
        batch[0] = batch[0].reshape(-1, batch[0].shape[-1])
        batch[1] = batch[1].reshape(-1, batch[1].shape[-1])
        batch[2] = batch[2].reshape(-1, batch[2].shape[-1])
        #model.eval()
        with torch.no_grad():
            points_x, points_y, x_raw = batch
            thickness = get_thickness_from_ps_1d(x_raw.detach().cpu().numpy()[:,121])
            #points_x, points_y = (points_x.float()).cuda(), (points_y.float()).cuda()
            points_x = (points_x.float()).cuda()

            #outputs_y = model(points_x)
            y_1.append(get_inverse()['0_29'](all_models['0_29'](points_x).detach().cpu().numpy())*thickness*LATVAP)
            y_2.append(get_inverse()['30_59'](all_models['30_59'](points_x).detach().cpu().numpy())*thickness)
            #y_3 = np.concatenate(([y_3, get_inverse()['61_64'](all_models['61_64'](points_x).detach().cpu().numpy())]),axis=0)
            #y_4.append(get_inverse()['61_65'](all_models['61_65'](points_x).detach().cpu().numpy()))
            batch[1][:,:30] *= thickness*LATVAP
            batch[1][:,30:60] *= thickness
            y_gt.append(batch[1].numpy())


    y_1 = np.concatenate(y_1, axis=0)
    y_2 = np.concatenate(y_2, axis=0)
    #y_4 = np.concatenate(y_4, axis=0)
    y_gt = np.concatenate(y_gt, axis=0)


    test_time = time.time() - test_time_begin

    qtend_log = report_qtend(y_gt, y_1)
    qtend_log_lvl = report_qtend_vert(y_gt, y_1)
    #print(json.dumps(qtend_log_lvl, indent=4))
    del y_1

    stend_log = report_stend(y_gt, y_2)
    stend_log_lvl = report_stend_vert(y_gt, y_2)
    del y_2

    #rad_log = report_rad_prog(y_gt, y_4)

    #rad_log_individual = report_rad_prog_individual(y_gt, y_4)

    return {
        "qtend_log": qtend_log,
        "stend_log": stend_log,
        #"rad_log": rad_log,
        #"rad_log_individual": rad_log_individual,
        "qtend_log_lvl": qtend_log_lvl,
        "stend_log_lvl": qtend_log_lvl
    }

if __name__ == "__main__":
    import argparse
    import random
    #dh_settings.get_weight()
    #dh_settings.get_hyai_hybi()
    torch.multiprocessing.set_sharing_strategy('file_system')
    random.seed(0)
    np.random.seed(0)
    parser = argparse.ArgumentParser()
    #parser.add_argument("--output_type", "-ot", help="choose from 0-29, 30-59, 61-65")
    #parser.add_argument("--resume", "-re", help="path to selected model")
    parser.add_argument("config", help="path to configuration file")
    parser.add_argument("out_json", help="path to output json file")
    parser.add_argument("--sample", type=int, help="sample frequency to use", default=1)
    args = parser.parse_args()
    print(args.config)
    with open(args.config, "r") as f:
        config = json.load(f)
    all_models = load_models(
        config["0-29"]["ckpt_path"],
        config["30-59"]["ckpt_path"],
        None,
        #config["61-65"]["ckpt_path"]
        None
    )
    np.random.seed(0)
    #data_dir = "/data/nncam_data/image_testset/"
    data_dir = "/home/users/data/nncam_data/image_testset/"
    print("Test set path: ", data_dir)

    cudnn.benchmark = True

    all_files = glob.glob(data_dir+'/*')[2::args.sample]
    for fn in all_files:
        if "08691" in fn:
            all_files.remove(fn)
    all_files.sort()
    print("all_files len ", len(all_files))
    test_idx = np.random.choice(len(all_files), len(all_files), replace=False)
    print(test_idx[:10])
    test_files = [all_files[i] for i in test_idx]
    print("Test file size: " ,len(test_files))
    print(test_files[:3])

    testing_set = TimeDatasetDisk(file_names=test_files, is_train=False, noise_std=0, output_normalized=False, silent=True, raw_input=True)
    testloader = data.DataLoader(testing_set, shuffle=False, batch_size=1, num_workers=1)

    logs = offline_test(args, all_models, testloader)

    with open(args.out_json, "w") as f:
        json.dump({
            "dq/dt": logs["qtend_log"],
            "dT/dt": logs["stend_log"],
            #"radiation": rad_log,
            #"dqdt_lvl": qtend_log_lvl,
            #"dTdt_lvl": stend_log_lvl,
#            **logs["rad_log_individual"]
            }, f, indent=4)
    with open("ex_"+args.out_json, "w") as f:
        json.dump({
            #"dq/dt": qtend_log,
            #"dT/dt": stend_log,
#            "radiation": logs["rad_log"],
            "dqdt_lvl": logs["qtend_log_lvl"],
            "dTdt_lvl": logs["stend_log_lvl"],
            #**rad_log_individual
            }, f, indent=4)





