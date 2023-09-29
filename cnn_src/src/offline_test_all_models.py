import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import numpy as np
#from utils import Logger, AverageMeter, mkdir_p
#from dataloader_subset_files import Dataset
#from .dataloader_time_embedded import TimeDataset, get_inverse, DatasetDisk
from dataloader_pad_test import Dataset
#from dataloader_subset_files_offline_test import Dataset
#from offline_test.dataloader_subset_files import Dataset
from torch.utils import data
import models_pad as models
#import train_tools as tools
import json
import time
import glob
#from nncam_models.load_models import load_models, get_inverse
#from tqdm.autonotebook import tqdm

def inverse_61_65(y):
    y[:,0] = (y[:,0]) * (1412 - 0)
    y[:,1] = (y[:,1]) * (1412 - 0)
    y[:,2] = (y[:,2]) * (1412 - 0)
    y[:,3] = (y[:,3]) * (1412 - 0)
    y[:,4] = (y[:,4]) * (1412 - 0)
    return y

def inverse_61_64(y):
    y[:,0] = (y[:,0]+1)/2*(332+53)-53       # flns
    y[:,1] = (y[:,1]+1)/2*(419-83)+83       # flnt
    y[:,2] = (y[:,2]+1)/2*(1063+2.13)-2.13  # fsns
    y[:,3] = (y[:,3]+1)/2*(1299)+0          # fsnt
    return y

def get_inverse():
    inverse = {}
    inverse[ '0_29'] = lambda y: (y+1)/2*(3.11e-6*2)-3.11e-6
    inverse['30_59'] = lambda y: (y+1)/2*(3.63*2)-3.63
    inverse['60']    = lambda y: (y+1)/2*(2.12e-6)
    inverse['61_64'] = lambda y: inverse_61_64(y)
    inverse['61_65'] = lambda y: inverse_61_65(y)

    return inverse

def load_models(model029, model3059, model6165):
    all_models = {}
    for model_type in ['0_29', '30_59', '61_65']:
        # define model
        input_dim = 122
        if model_type == '0_29':
            output_dim = 30
            resume = model029
        if model_type == '30_59':
            output_dim = 30
            resume = model3059
        if model_type == '60':
            output_dim = 1
        if model_type == '61_65':
            output_dim = 5
            resume = model6165

        #if args.network == 'cnn':
        #    model = models.cnn(input_dim, output_dim)
        #elif args.network == 'cnn2_pad':
        model = models.cnn2_pad(input_dim, output_dim)
        #else:
        #    model = None

        print('Total params: %.2f' % (sum(p.numel() for p in model.parameters())))

        model = torch.nn.DataParallel(model).cuda()
        cudnn.benchmark = True
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint['state_dict'])
        all_models[model_type] = model
    return all_models



def Regression_Metrics(y_true, y_pred):
    
    var  = np.var(y_true)
    std  = np.std(y_true)
    
    mse  = np.mean((y_true-y_pred)**2)
    rmse = mse**0.5
       
    mae    = np.mean(np.absolute(y_pred-y_true))  # Mean absolute error
    max_ae = np.max(np.absolute(y_pred-y_true))   # Max absolute error
    
    bias = np.mean(y_pred-y_true)
    r2   = 1 - mse/var 
    
    return float(var), float(std), float(mse), float(rmse), float(mae), float(max_ae), float(bias), float(r2)

def report_stend_vert(y_gt, stend):
  out =  {
          "var": [],
          "std": [], 
          "mse": [], 
          "rmse": [], 
          "mae": [], 
          "max_ae": [], 
          "bias": [], 
          "r2": []
        }
  for i in range(30):
    var, std, mse, rmse, mae, max_ae, bias, r2 = Regression_Metrics(y_gt[:,30+i:30+i+1]*3600*24/1004, stend[:,i:i+1]*3600*24/1004)
    stend_metrics = '{}\t{:.4}\t{:.4}\t{:.4}({:.4%})\t{:.4}\t{:.4}\t{:.4}({:.4%})\t'.format(50, r2, mse, rmse, rmse/std, mae, max_ae, bias, bias/std)
    stend_log = ""
    stend_log += f"stend lvl {i} metrics:\n"
    stend_log += "epoch\tr2\tmse\trmse\t\tmae\tmax_ae\tbias\n"
    stend_log += stend_metrics
    stend_log += "\n"
    print(stend_log)
    out["var"].append(var)
    out["std"].append(std)
    out["mse"].append(mse)
    out["rmse"].append(rmse)
    out["mae"].append(mae)
    out["max_ae"].append(max_ae)
    out["bias"].append(bias)
    out["r2"].append(r2)
  return out

def report_stend(y_gt, stend):
  var, std, mse, rmse, mae, max_ae, bias, r2 = Regression_Metrics(y_gt[:,30:60]*3600*24/1004, stend*3600*24/1004)
  stend_metrics = '{}\t{:.4}\t{:.4}\t{:.4}({:.4%})\t{:.4}\t{:.4}\t{:.4}({:.4%})\t'.format(50, r2, mse, rmse, rmse/std, mae, max_ae, bias, bias/std)
  stend_log = ""
  stend_log += "stend metrics:\n"
  stend_log += "epoch\tr2\tmse\trmse\t\tmae\tmax_ae\tbias\n"
  stend_log += stend_metrics
  stend_log += "\n"
  print(stend_log)
  return {
          "var": var,
          "std": std, 
          "mse": mse, 
          "rmse": rmse, 
          "mae": mae, 
          "max_ae": max_ae, 
          "bias": bias, 
          "r2": r2
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
          "r2": []
        }
  for i in range(30):
    var, std, mse, rmse, mae, max_ae, bias, r2 = Regression_Metrics(y_gt[:,0+i:0+i+1]*3600*24*1000, qtend[:,i:i+1]*3600*24*1000)
    if std == 0:
        std = 1e-16
    qtend_metrics = '{}\t{:.4}\t{:.4}\t{:.4}({:.4%})\t{:.4}\t{:.4}\t{:.4}({:.4%})\t'.format(50, r2, mse, rmse, rmse/std, mae, max_ae, bias, bias/std)
    qtend_log = ""
    qtend_log += f"qtend lvl {i} metrics:\n"
    qtend_log += "epoch\tr2\tmse\trmse\t\tmae\tmax_ae\tbias\n"
    qtend_log += qtend_metrics
    qtend_log += "\n"
    print(qtend_log)
    out["var"].append(var)
    out["std"].append(std)
    out["mse"].append(mse)
    out["rmse"].append(rmse)
    out["mae"].append(mae)
    out["max_ae"].append(max_ae)
    out["bias"].append(bias)
    out["r2"].append(r2)
  return out

def report_qtend(y_gt, qtend):
  var, std, mse, rmse, mae, max_ae, bias, r2 = Regression_Metrics(y_gt[:,0:30]*3600*24*1000, qtend*3600*24*1000)
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
          "r2": r2
        }

def report_rad_prog(y_gt, rad_prog):
  var, std, mse, rmse, mae, max_ae, bias, r2 = Regression_Metrics(y_gt[:,61:66], rad_prog)
  rad_prog = '{}\t{:.4}\t{:.4}\t{:.4}({:.4%})\t{:.4}\t{:.4}\t{:.4}({:.4%})\t'.format(50, r2, mse, rmse, rmse/std, mae, max_ae, bias, bias/std)
  rad_log = ""
  rad_log += "rad metrics:\n"
  rad_log += "epoch\tr2\tmse\trmse\t\tmae\tmax_ae\tbias\n"
  rad_log += rad_prog
  rad_log += "\n"
  print(rad_log)
  return {
          "var": var,
          "std": std, 
          "mse": mse, 
          "rmse": rmse, 
          "mae": mae, 
          "max_ae": max_ae, 
          "bias": bias, 
          "r2": r2
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
      var, std, mse, rmse, mae, max_ae, bias, r2 = Regression_Metrics(y_gt[:,61+i:61+i+1], rad_pred[:,i:i+1])
      rad_prog = '{}\t{:.4}\t{:.4}\t{:.4}({:.4%})\t{:.4}\t{:.4}\t{:.4}({:.4%})\t'.format(50, r2, mse, rmse, rmse/std, mae, max_ae, bias, bias/std)
      rad_log = ""
      rad_log += f"rad {rad_vars[i]} metrics:\n"
      rad_log += "epoch\tr2\tmse\trmse\t\tmae\tmax_ae\tbias\n"
      rad_log += rad_prog
      rad_log += "\n"
      print(rad_log)
      out[rad_vars[i]] = {
        "var": var,
        "std": std, 
        "mse": mse, 
        "rmse": rmse, 
        "mae": mae, 
        "max_ae": max_ae, 
        "bias": bias, 
        "r2": r2
      }
  return out

if __name__ == "__main__":
    import argparse
    import random
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
        config["61-65"]["ckpt_path"]
    )
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

    testing_set = Dataset(files=test_files, is_train=False, train62=False, noise_std=0)
    testloader = data.DataLoader(testing_set, shuffle=False, batch_size=1, num_workers=1)

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
    for iter, batch in enumerate(testloader):
        suffix = 'testing- epoch:{}| iters:{}/{} |'.format(epoch, iter+1, len(testloader))
        #batch[0] = batch[0].reshape(-1, batch[0].shape[-1])
        #batch[1] = batch[1].reshape(-1, batch[1].shape[-1])
        #model.eval()
        with torch.no_grad():
            points_x, points_y = batch
            #points_x, points_y = (points_x.float()).cuda(), (points_y.float()).cuda()
            points_x = (points_x.float()).cuda()

            #outputs_y = model(points_x)
            y_1.append(get_inverse()['0_29'](all_models['0_29'](points_x).detach().cpu().numpy()))
            y_2.append(get_inverse()['30_59'](all_models['30_59'](points_x).detach().cpu().numpy()))
            #y_3 = np.concatenate(([y_3, get_inverse()['61_64'](all_models['61_64'](points_x).detach().cpu().numpy())]),axis=0)
            y_4.append(get_inverse()['61_65'](all_models['61_65'](points_x).detach().cpu().numpy()))
            y_gt.append(batch[1].numpy())


    y_1 = np.concatenate(y_1, axis=0)
    y_2 = np.concatenate(y_2, axis=0)
    y_4 = np.concatenate(y_4, axis=0)
    y_gt = np.concatenate(y_gt, axis=0)


    test_time = time.time() - test_time_begin

    qtend_log = report_qtend(y_gt, y_1)
    qtend_log_lvl = report_qtend_vert(y_gt, y_1)
    #print(json.dumps(qtend_log_lvl, indent=4))
    del y_1

    stend_log = report_stend(y_gt, y_2)
    stend_log_lvl = report_stend_vert(y_gt, y_2)
    del y_2

    rad_log = report_rad_prog(y_gt, y_4)

    rad_log_individual = report_rad_prog_individual(y_gt, y_4)

    with open(args.out_json, "w") as f:
        json.dump({
            "dq/dt": qtend_log,
            "dT/dt": stend_log,
            #"radiation": rad_log,
            #"dqdt_lvl": qtend_log_lvl,
            #"dTdt_lvl": stend_log_lvl,
            **rad_log_individual
            }, f, indent=4)
    with open("ex_"+args.out_json, "w") as f:
        json.dump({
            #"dq/dt": qtend_log,
            #"dT/dt": stend_log,
            "radiation": rad_log,
            "dqdt_lvl": qtend_log_lvl,
            "dTdt_lvl": stend_log_lvl,
            #**rad_log_individual
            }, f, indent=4)





