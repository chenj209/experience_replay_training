import os
import time
import json
import glob
import random
import sys
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import numpy as np
from torch.utils import data
from datetime import datetime
# from torch.utils.data.dataloader import default_collate

sys.path.append(os.path.join(sys.path[0], "..", "models"))
import autoencoder
sys.path.append(os.path.join(sys.path[0], ".."))
from utils import Logger, AverageMeter, mkdir_p
sys.path.append(os.path.join(sys.path[0], "..", "utils"))
import argsparser
import tools
from data_shape import to_inference_shape, inverse_to_inference_shape
sys.path.append(os.path.join(sys.path[0], "..", "dataloader"))
from dataloader_newformat import DatasetDisk

def prep_dataloaders(args):
    #data_dir = args.data_dir
    #if not os.path.isdir(data_dir):
        # data_dir = "./data/"
    data_dir = "/pscratch/sd/c/chenjd21/spcam_new_data/"
    col_names = np.loadtxt(data_dir + "/col_names.txt", dtype=str)
    col_names_x = ["QL", "T_nn_in", "dqvls_nn_in", "dTls_nn_in", "SOLIN", "SPPS"]
    prev_ex_vars = ["qtend_check", "stend_check", "SOLL", "SOLLD", "SOLS", "SOLSD", "FSDS"]
    col_names_y = ["qtend_check"]
    data_means = dict(np.load(data_dir + "/data_means.npz"))
    data_stds = dict(np.load(data_dir + "/data_stds.npz"))
    #################### 屏蔽掉一些可能存在异常的数据集 ###############################
    #all_files = glob.glob(args.data_dir+'/*')[::13]#[::7]
    all_files = glob.glob(data_dir+'/*.npy')
    all_files.sort()
    all_files = all_files[35040:]

    print('org file num:', len(all_files))
    #for i in range(17507,17530):
    for i in range(17507,17531):
        for file_name in all_files:
            if str(i) in file_name:
                all_files.remove(file_name)
    print('after file num:', len(all_files))

    testing_set = DatasetDisk(
        all_files,
        col_names,
        col_names_x,
        col_names_y,
        data_stds,
        data_means,
        is_train=False,
        noise_std=0,
        multistep=int(args.multistep),
        sample_rate=192,
        prev_ex_vars=prev_ex_vars+args.ex_input,
        image=True)
    testloader = data.DataLoader(testing_set, shuffle=True, batch_size=args.train_batch, num_workers=args.workers)

    return testloader 

def prep_models(args):
    with open(args.ae_config, 'r') as f:
        ae_config = json.load(f)
    print(f"Model input size: {ae_config['input_size']}")
    #model = autoencoder.AutoencoderResMLP(input_size, 30, args.node_size, args.activation, args.num_blocks, args.latent_dim, region_mask=region_mask, resmlp=(args.pred_weight!=0))
    print(f"Loading model config: {json.dumps(ae_config, indent=4)}")
    model = autoencoder.Autoencoder(ae_config)
    print("Model structure:")
    print(model)
    print('Total params: %.2f' % (sum(p.numel() for p in model.parameters())))

    model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(torch.load(args.resume))
    cudnn.benchmark = True

    return model

import math
def get_min_max_coords(mask, pad):
    min_x = np.min(np.where(mask)[0])
    max_x = np.max(np.where(mask)[0])
    min_y = np.min(np.where(mask)[1])
    max_y = np.max(np.where(mask)[1])
    wid_x = math.ceil((max_x - min_x)/2)*2
    wid_y = math.floor((max_y - min_y)/2)*2
    return int(min_x-pad), int(min_x+wid_x+pad), \
        int(min_y-pad), int(min_y+wid_y+pad)

def main(args):
    # region_mask = to_inference_shape(region_mask).squeeze()
    testloader = prep_dataloaders(args)
    model = prep_models(args)
    region_mask = np.load(args.region_mask)
    min_x, max_x, min_y, max_y = get_min_max_coords(region_mask.squeeze(), 3)
    lon = np.linspace(0,357.5,144)
    lat = np.linspace(-90,90,96)
    print("Region window coordinates: ", lon[min_y], lon[max_y], lat[min_x], lat[max_x])
    criterion = nn.MSELoss()
    avg_mse = 0
    avg_mse_by_variable = np.zeros(309)
    for iter, batch in enumerate(testloader):
            if batch[0].size() == 1 and batch[0] == 0:
                continue
            batch[0] = batch[0][:, :, min_x: max_x, min_y: max_y]

            model.eval()
            points_x = batch[0]
            points_x = (points_x.float()).cuda()
            x_rec = model(points_x)
            loss_rec = criterion(x_rec, points_x).item()
            avg_mse += loss_rec
            avg_mse_by_variable += np.mean((x_rec - points_x).cpu().detach().numpy()**2, axis=(0,2,3))
            if iter < 10:
                np.save(f"{args.save_path}/offline_test_ae_{iter}_x.npy", points_x.cpu().detach().numpy())
                np.save(f"{args.save_path}/offline_test_ae_{iter}_x_rec.npy", x_rec.cpu().detach().numpy())

            current_iters += 1
            print('testing: iters:{}/{}| mse:{:.6f}|'.format(iter+1, len(testloader), loss_rec))
    avg_mse /= current_iters
    avg_mse_by_variable /= current_iters
    np.save(f"{args.save_path}/offline_test_ae_avg_mse.npy", avg_mse)
    print(f"Average MSE: {avg_mse}")
