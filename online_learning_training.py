import os
import sys 
import datetime
import time
import numpy as np
import torch
import json
import subprocess
import random

sys.path.append("/cust_users/chenj209/prog_val_mod/src.baseline/") 

from tools import normalization, normalization_by_level
from tools import inverse_61_64, inverse_61_65
from tools import load_ckpts, load_ckpts_manual
from tools import gen_inputs, gen_outputs, gen_inputs_q_only, gen_outputs_q_only

from atm_log_process.parse_config import config_to_path
from nncam_data_explore.src.utility import filename_to_idx

# online learning training related
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim import lr_scheduler
from dataloader_subset_files import Dataset
from torch.utils import data
import train_tools
from utils import mkdir_p, Logger, AverageMeter

os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2, 3'

import glob
import json
import argparse
import shutil
from copy import deepcopy
import re

BASE_EPOCH = 0
M = 16 #  Online training frequency
CKPT_FREQ = 5 # Save online ckpt frequency
MODELS_TO_TRAIN = ["0_29", "30_59", "61_65"]
MAX_TIMEOUT = 30
GW_PATH = "/temp_share/stabilities.analysis/Gravity-waves/GW_dqv.npy"
GW_DS_PATH = "/temp_share/stabilities.analysis/Gravity-waves/GW_ds.npy"
CKPT_DIR = "/cust_users/x-w19/nncam.ckpts"
ONLINE_DATA_BASE = "/cust_users/online_data/chenj209_temp/"
KEY_MODEL_TYPES = ["0-29", "30-59", "61-65"]
EX_MODEL_TYPES = ["0-29", "30-59", "61-64", "61-65"]
MODEL_TYPES = KEY_MODEL_TYPES + EX_MODEL_TYPES
CKPT_CONFIG_MAPPING = {
        "40%sampled": "0.4sampled",
        "0.4sampled": "0.4sampled",
        "newdata": "newData",
        "2years": "2years",
        "25GB": "25GB"
}

CKPT_CONFIG_DEFAULT = {
        "61-64": "/cust_users/x-w19/nncam.ckpts/resmlp.2years.50epochs/61_64_nodesize512_num_blocks7_actrelu_bs1024_scheduler_coslr_lr0.001_ep50_noise0.0_wd0_dropout0/checkpoint.pth.tar"
}


def print_config(config):
    print(json.dumps(parsed_config, sort_keys=True, indent=4))

def generate_config(config, online_path):
    config["datetime"] = str(datetime.datetime.now())
    with open(f"{online_path}/resmlp_config.json", "w") as f:
        json.dump(config, f, sort_keys=True, indent=4)

def buffer_lock(file, timeout=True):
    start_time = time.time()
    while 1:
        if os.path.exists(file):
            return os.path.exists(file)
            # break
        else:
            cur_time = time.time()
            if timeout and cur_time - start_time > MAX_TIMEOUT:
              return cur_time - start_time
            continue

def parse_config(config):

    return config_to_path(config)
    """
    Parameters:
        config(dict): configuration in readable format

    Returns
        parsed_config(dict): configuration with absolute path to checkpoints
    """
    ckpt_sub_dirs = os.listdir(CKPT_DIR)
    parsed_config = deepcopy(config)

    # if config provides a dictionary like config object for each model
    # generate online data path based on the dictionary config
    today = datetime.date.today()
    config["date"] = today.strftime("%y_%m_%d")
    if "online_data_path" not in config:
        opath = []
        if "case_no" not in config:
            for model_type in KEY_MODEL_TYPES:

              # check if each key model config exists
                if model_type not in config:
                    raise Exception(f"Missing necessary {model_type} ckpt config")

                # use model_type_ckpt_name if ckpt_name is provided, e.g. "0-29_ckpt_name": "test_ckpt"
                if f"{model_type}_ckpt_name" in config:
                    opath.append(model_config["ckpt_name"])
                    continue


                # generate ckpt_name based on model_config
                model_config = config[model_type]
                if "training_dataset" in model_config:
                    opath.append(f"{model_type}_{config[model_type]['training_dataset']}" \
                        + f"_noise{config[model_type]['noise_level']}" \
                        + f"_ep{config[model_type]['epoch']}")
                else:
                    raise Exception(f"{model_type} cannot be converted to str")
        else:
            opath.append(config["date"])
            opath.append(f"case{config['case_no']}")

        config["online_data_path"] = ONLINE_DATA_BASE + "_".join(opath)
        if config["qtend[:10]=0.0"]:
            config["online_data_path"] += "_qtend_0-10=0.0"
        config["online_data_path"] += "/"

    # for each model type, compute the checkpoint file path
    for model_type in MODEL_TYPES:

        if model_type not in config:
            # 61-64 is not used right now, so use the default 61-64 checkpoint file
            if model_type in CKPT_CONFIG_DEFAULT: 
                parsed_config[model_type] = CKPT_CONFIG_DEFAULT[model_type]
            else:
                raise Exception(f"Missing {model_type} config")
            continue

        model_config = config[model_type]

        # if the model config is a string, then don't parse it
        # used for cases that does not follow resmlp.{training_dataset}.noise{noise} rule
        if isinstance(model_config, str):
            continue

        # use lower to allow mistyped capped letters
        training_dataset = CKPT_CONFIG_MAPPING[model_config["training_dataset"].lower()]

        # format noise to be 0.0, 0.01, 0.005 liked float strings
        noise = "{:.5f}".format(float(model_config["noise_level"]))
        noise = noise[:3] + noise[3:].rstrip("0")

        # only covers resmlp.{training_dataset}.noise{noise} typed checkpoints
        model_path = f"{CKPT_DIR}/resmlp.{training_dataset}.noise{noise}"

        # matches sub checkpoint directory that starts with 0_29, etc
        checkpoint_path = f"{model_path}/{model_type.replace('-', '_')}*"
        checkpoint_dir = glob.glob(checkpoint_path)
        if len(checkpoint_dir) == 0:
            raise Exception(checkpoint_path, " does not exist")
        checkpoint_dir = checkpoint_dir[0]

        # matches epoch tar file
        checkpoint_file = f"{checkpoint_dir}/checkpoint_epoch{model_config['epoch']}.pth.tar"
        if not os.path.isfile(checkpoint_file):
            raise Exception(checkpoint_file, " does not exist")
        parsed_config[model_type] = checkpoint_file
    
    return parsed_config



def run_experiment(all_models, online_data_path, data_buffer_path, qtend_post_process, args):
    curr_lr = None
    print("qtend post process: ", qtend_post_process)
    inverse = {}
    inverse['0_29']  = lambda x: (x+1)/2*(3.11e-6*2)-3.11e-6
    inverse['30_59'] = lambda x: (x+1)/2*(3.63*2)-3.63
    inverse['61_64'] = lambda x: inverse_61_64(x) # 现在不用的辐射target
    inverse['61_65'] = lambda x: inverse_61_65(x)

    #all_models = load_ckpts('/cust_users/x-w19/nncam.ckpts', 'resmlp.2years', 'noise0.0', '50')

    # crash
    #all_models = load_ckpts('/cust_users/x-w19/nncam.ckpts', 'resmlp.newData', 'noise0.0', '25')

    # load newly trained model with input level norm
    # 加载checkpoint, 四个神经网络

    #computed_min_max_x = np.load("/cust_users/chenj209/neuroGCM_training/neuroParameterization/src/data-nncam_data-image_set--level_min_max-data-x.npy")

    step = 0

    skip_first = True
    
    # Setup training optimizer
    optimizers = {}
    lr_schedulers = {}
    if args.optim == 'sgd':
        for model_type in MODELS_TO_TRAIN:
            optimizers[model_type] = optim.SGD(all_models[model_type].parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
            lr_schedulers[model_type] = lr_scheduler.StepLR(optimizers[model_type], step_size=args.lr_step_size, gamma=0.1)
    elif args.optim == 'adam':
        for model_type in MODELS_TO_TRAIN:
            optimizers[model_type] = optim.Adam(all_models[model_type].parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.weight_decay)
            lr_schedulers[model_type] = lr_scheduler.StepLR(optimizers[model_type], step_size=args.lr_step_size, gamma=0.1)
    else:
        optimizers = None

    #lr_scheduler = {'coslr': train_tools.cosine_lr,
    #                'constant': train_tools.constant}


    # setup Logger
    #if args.resume:
        # Load checkpoint.
    #    print('==> Resuming from checkpoint..')
    #    assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
    #    checkpoint = torch.load(args.resume)
    #    model.load_state_dict(checkpoint['state_dict'])
    #    logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
    #else:
    logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title='',resume=BASE_EPOCH>0)
    logger.set_names([
        'Epoch',
        *[model_type+' LR' for model_type in MODELS_TO_TRAIN], 
        *[model_type+' train mse' for model_type in MODELS_TO_TRAIN]
        ])



    while 1:
        # check if cam has gen the data_buffer.bin
        print("\n\033[1;35mWaiting for Fortran2Python...\033[0m\n")

        # 在一个文件夹下创建lock同步
        buffer_flag_1 = buffer_lock(f"{data_buffer_path}/buffer_flag_1.log", timeout=(not skip_first))
        if buffer_flag_1 == 1:
            print("Fortran2Python start...")
            os.remove(f"{data_buffer_path}/buffer_flag_1.log")
            buffer_flag_1 = False
            skip_first = False
        else:
            if buffer_flag_1 >= MAX_TIMEOUT:
              print(f"kill for {buffer_flag_1}")
              curr_lr = float(online_training(all_models, optimizers, lr_schedulers, logger, step, online_data_path, args, force_save=True))
              break



        # init data_x
        # fortran 写的数组
        print("Data initialization...")
        time_start = time.time()

        solin = np.fromfile(f"{data_buffer_path}/solin.bin", dtype='>f8') # dtype='>f8' 指 big_endian 的 double
        solin = solin.astype(np.float64)
        solin = solin.reshape(96,144).T

        ps = np.fromfile(f"{data_buffer_path}/ps.bin", dtype='>f8') # dtype='>f8' 指 big_endian 的 double
        ps = ps.astype(np.float64)
        ps = ps.reshape(96,144).T

        Q = np.fromfile(f"{data_buffer_path}/Q.bin", dtype='>f8') # dtype='>f8' 指 big_endian 的 double
        Q = Q.astype(np.float64)
        Q = Q.reshape(30,96,144).transpose((2,1,0))

        if os.path.isfile(f"{data_buffer_path}/qtend_check.bin") and (step <= 10 and step > 0): 
            # reading from previous step output
            dQ = np.fromfile(f"{data_buffer_path}/qtend_check.bin", dtype='>f8') # dtype='>f8' 指 big_endian 的 double
            dQ = dQ.astype(np.float64)
            dQ = dQ.reshape(30,96,144)

            dS = np.fromfile(f"{data_buffer_path}/stend_check.bin", dtype='>f8') # dtype='>f8' 指 big_endian 的 double
            dS = dS.astype(np.float64)
            dS = dS.reshape(30,96,144)

        dQ_crm = None
        if os.path.isfile(f"{data_buffer_path}/spdt_crm.bin"): 
            # reading from previous step crm output
            dQ_crm = np.fromfile(f"{data_buffer_path}/spdq_crm.bin", dtype='>f8') # dtype='>f8' 指 big_endian 的 double
            dQ_crm = dQ_crm.astype(np.float64)
            dQ_crm = dQ_crm.reshape(30,96,144)

            dS_crm = np.fromfile(f"{data_buffer_path}/spdt_crm.bin", dtype='>f8') # dtype='>f8' 指 big_endian 的 double
            dS_crm = dS_crm.astype(np.float64)
            #dS_crm = dS_crm.reshape(144,96,30)
            dS_crm = dS_crm.reshape(30,96,144)
            #soll  = y_4[:, 0].reshape(144,96).T.astype('>f8')
            #sols  = y_4[:, 1].reshape(144,96).T.astype('>f8')
            #solsd = y_4[:, 2].reshape(144,96).T.astype('>f8')
            #solld = y_4[:, 3].reshape(144,96).T.astype('>f8')
            #fsds  = y_4[:, 4].reshape(144,96).T.astype('>f8')
            soll_crm = np.fromfile(f"{data_buffer_path}/soll_crm.bin", dtype='>f8') # dtype='>f8' 指 big_endian 的 double
            soll_crm = soll_crm.astype(np.float64)
            #soll_crm = soll_crm.reshape(144,96,1)
            soll_crm = soll_crm.reshape(144*96,1)

            sols_crm = np.fromfile(f"{data_buffer_path}/sols_crm.bin", dtype='>f8') # dtype='>f8' 指 big_endian 的 double
            sols_crm = sols_crm.astype(np.float64)
            sols_crm = sols_crm.reshape(144*96,1)
            #sols_crm = sols_crm.reshape(144,96,1)

            solsd_crm = np.fromfile(f"{data_buffer_path}/solsd_crm.bin", dtype='>f8') # dtype='>f8' 指 big_endian 的 double
            solsd_crm = solsd_crm.astype(np.float64)
            solsd_crm = solsd_crm.reshape(144*96,1)
            #solsd_crm = solsd_crm.reshape(144,96,1)

            solld_crm = np.fromfile(f"{data_buffer_path}/solld_crm.bin", dtype='>f8') # dtype='>f8' 指 big_endian 的 double
            solld_crm = solld_crm.astype(np.float64)
            solld_crm = solld_crm.reshape(144*96,1)
            #solld_crm = solld_crm.reshape(144,96,1)

            fsds_crm = np.fromfile(f"{data_buffer_path}/fsds_crm.bin", dtype='>f8') # dtype='>f8' 指 big_endian 的 double
            fsds_crm = fsds_crm.astype(np.float64)
            fsds_crm = fsds_crm.reshape(144*96,1)
            #fsds_crm = fsds_crm.reshape(144,96,1)

        T = np.fromfile(f"{data_buffer_path}/T.bin", dtype='>f8') # dtype='>f8' 指 big_endian 的 double
        T = T.astype(np.float64)
        T = T.reshape(30,96,144).transpose((2,1,0))

        # 不用管 start
        omega = np.fromfile(f"{data_buffer_path}/omega.bin", dtype='>f8') # dtype='>f8' 指 big_endian 的 double
        omega = omega.astype(np.float64)
        omega = omega.reshape(30,96,144)

        pmid = np.fromfile(f"{data_buffer_path}/pmid.bin", dtype='>f8') # dtype='>f8' 指 big_endian 的 double
        pmid = pmid.astype(np.float64)
        pmid = pmid.reshape(30,96,144)

        pint = np.fromfile(f"{data_buffer_path}/pint.bin", dtype='>f8') # dtype='>f8' 指 big_endian 的 double
        pint = pint.astype(np.float64)
        pint = pint.reshape(31,96,144)

        s = np.fromfile(f"{data_buffer_path}/s.bin", dtype='>f8') # dtype='>f8' 指 big_endian 的 double
        s = s.astype(np.float64)
        s = s.reshape(30,96,144)

        zm = np.fromfile(f"{data_buffer_path}/zm.bin", dtype='>f8') # dtype='>f8' 指 big_endian 的 double
        zm = zm.astype(np.float64)
        zm = zm.reshape(30,96,144)

        zi = np.fromfile(f"{data_buffer_path}/zi.bin", dtype='>f8') # dtype='>f8' 指 big_endian 的 double
        zi = zi.astype(np.float64)
        zi = zi.reshape(31,96,144)
        # 不用管 end

        dqvls = np.fromfile(f"{data_buffer_path}/dqvls.bin", dtype='>f8') # dtype='>f8' 指 big_endian 的 double
        dqvls = dqvls.astype(np.float64)
        dqvls = dqvls.reshape(30,96,144).transpose((2,1,0))

        dTls = np.fromfile(f"{data_buffer_path}/dTls.bin", dtype='>f8') # dtype='>f8' 指 big_endian 的 double
        dTls = dTls.astype(np.float64)
        dTls = dTls.reshape(30,96,144).transpose((2,1,0))

        # xy拉成一维
        Q = Q.reshape(144*96,30)
        T = T.reshape(144*96,30)
        dqvls = dqvls.reshape(144*96,30)
        dTls  = dTls.reshape(144*96,30)
        solin = solin.reshape(144*96,1)
        ps = ps.reshape(144*96,1)

        # input and extend input    
        data_x  = np.concatenate((Q, T, dqvls, dTls, solin, ps), axis = 1)
        # 仅分析使用
        inputs  = gen_inputs(data_x) # get online data(inputs) by Wang Xin on 2021-09-02
        # inputs  = gen_inputs_q_only(data_x) # only keep Q and dQls in the input

        print('Initialization, using time: {} sec\n'.format(time.time() - time_start))

        points_x1 = torch.cuda.FloatTensor(normalization(data_x))
        #points_x = torch.cuda.FloatTensor(normalization_by_level(data_x, computed_min_max_x))

        # Inference
        print("Inferencing...")
        time_start = time.time()
        # set all models to eval model for inference
        all_models["0_29"].eval()
        all_models["30_59"].eval()
        all_models["61_64"].eval()
        all_models["61_65"].eval()

        with torch.no_grad():
            y_1 = inverse[ '0_29'](all_models[ '0_29'](points_x1).detach().cpu().numpy())
            y_2 = inverse['30_59'](all_models['30_59'](points_x1).detach().cpu().numpy())
            y_3 = inverse['61_64'](all_models['61_64'](points_x1).detach().cpu().numpy())
            y_4 = inverse['61_65'](all_models['61_65'](points_x1).detach().cpu().numpy())

        print('Inference finished, using time: {} sec\n'.format(time.time() - time_start))


        # write back to bin file
        print("Writing back to data_buffer(.bin file)...")
        if skip_first:
            skip_first = False
            last_time = time.time()
        time_start = time.time()

        qtend = y_1.reshape(144,96,30).transpose((2,1,0)).astype('>f8')
        stend = y_2.reshape(144,96,30).transpose((2,1,0)).astype('>f8')
      # cp    = y_3.reshape(144,96).T.astype('>f8')
        # 现在不使用 start, 气候分析使用
        flns  = y_3[:,0].reshape(144,96).T.astype('>f8')
        flnt  = y_3[:,1].reshape(144,96).T.astype('>f8')
        fsns  = y_3[:,2].reshape(144,96).T.astype('>f8')
        fsnt  = y_3[:,3].reshape(144,96).T.astype('>f8')
        # 现在不使用 end 

        # ex output ...
        soll  = y_4[:, 0].reshape(144,96).T.astype('>f8')
        sols  = y_4[:, 1].reshape(144,96).T.astype('>f8')
        solsd = y_4[:, 2].reshape(144,96).T.astype('>f8')
        solld = y_4[:, 3].reshape(144,96).T.astype('>f8')
        fsds  = y_4[:, 4].reshape(144,96).T.astype('>f8')

        # 辐射后处理
        soll[soll<0]   = 0.0
        sols[sols<0]   = 0.0
        solsd[solsd<0] = 0.0
        solld[solld<0] = 0.0
        fsds[fsds<0]   = 0.0

        flns[flns<0] = 0.0
        flnt[flnt<0] = 0.0
        fsns[fsns<0] = 0.0
        fsnt[fsnt<0] = 0.0

        # 水汽后处理
        if qtend_post_process:
            qtend[:10] = 0.0 

        #if 4tep >= 10 and step < 20:
        # chenj209(20230301): let spcam run the first 10 step
        #if step < 10:
        #    add_data = np.load(GW_PATH)
        #    add_ds_data = np.load(GW_DS_PATH)
        #    print(f"qtend shape: {qtend.shape}, noise path: {add_data.shape}, {add_ds_data.shape}")
        #    # print(f"qtend shape: {qtend.shape}, noise path: {add_data.shape}")
        #    filter_mask = np.zeros(add_data.shape)
        #    filter_mask[:,53:57,26:30] = 4
        #    qtend += add_data*filter_mask
#            stend += add_ds_data*filter_mask
#
        # test new post processing
        #data = qtend
        #q99 = np.quantile(data.reshape(-1),0.99)
        #q1 = np.quantile(data.reshape(-1),0.01)
        #spread = q99 - q1
        #multiplier = 1
        #prev_ratio = 0

        #Q99 = np.quantile(Q.reshape(-1),0.99)
        #Q1 = np.quantile(Q.reshape(-1),0.01)
        
        #if spread > 0:
            #max_val = np.max(qtend)
            #qtend[qtend>q99] = np.tanh((qtend[qtend>q99]-q99)/spread)*(multiplier-prev_ratio)*spread+q99
            #qtend[qtend<q1] = q1-np.tanh((q1-qtend[qtend<q1])/spread)*(multiplier-prev_ratio)*spread
        #    qtend[qtend>q99] = q99
        #    qtend[qtend<q1] = q1
            # prev_ratio = np.abs(np.tanh((max_val - q99) / spread))
        #    print(f"Q outlier ratio: {(np.max(Q)-Q99) / (Q99-Q1)}")
            #print(f"After process min outlier ratio: {(q1 - np.min(qtend)) / spread}")
            #print(f"prev_ratio {prev_ratio}")


        qtend.tofile(f"{data_buffer_path}/qtend.bin")
        stend.tofile(f"{data_buffer_path}/stend.bin")
      # cp.tofile(f"{data_buffer_path}/cp.bin")
        flns.tofile(f"{data_buffer_path}/flns.bin")
        flnt.tofile(f"{data_buffer_path}/flnt.bin")
        fsns.tofile(f"{data_buffer_path}/fsns.bin")
        fsnt.tofile(f"{data_buffer_path}/fsnt.bin")

        # ex output to bin file ...
        soll.tofile(f"{data_buffer_path}/soll.bin")
        sols.tofile(f"{data_buffer_path}/sols.bin")
        solsd.tofile(f"{data_buffer_path}/solsd.bin")
        solld.tofile(f"{data_buffer_path}/solld.bin")
        fsds.tofile(f"{data_buffer_path}/fsds.bin")

        print('Writing back finished, using time: {} sec\n'.format(time.time() - time_start))

        # Generate Prognostic Validation results (npz) by Wang Xin on 2021-11-29
        # 仅分析使用
        if (step < 17520 or (step >= 17520 and (step+1)%12 == 0 and step < 35240)):
#            if step > 0 and step <= 10:
#                # chenj209(20230301): from step 1 to step 10, read CRM output
#                # CRM output is generated after every step
#                dQ = np.fromfile(f"{data_buffer_path}/qtend_check.bin", dtype='>f8') # dtype='>f8' 指 big_endian# 的 double
#                dQ = dQ.astype(np.float64)
#                dQ = dQ.reshape(30,96,144)
#
#                dS = np.fromfile(f"{data_buffer_path}/stend_check.bin", dtype='>f8') # dtype='>f8' 指 big_endian# 的 double
#                dS = dS.astype(np.float64)
#                dS = dS.reshape(30,96,144)
#                outputs = gen_outputs(dQ, dS, y_3, y_4)
#                np.savez(online_data_path +  '/prog-val_'   + "%005d"%(step), data_x = inputs, data_y = outputs)

        
#            if step >= 10:
                # chenj209(20230301): save python output after 10 step
            outputs = gen_outputs(qtend, stend, y_3, y_4)
    #outputs = gen_outputs_q_only(qtend) # only keeps dQ in the outputs
            np.savez(online_data_path +  '/prog-val_'   + "%005d"%(step+1), data_x = inputs, data_y = outputs)
#            else:
#                # chenj209(20230301): still save not used qtend stend for diagnostic
#                outputs = gen_outputs(qtend, stend, y_3, y_4)
#                np.savez(online_data_path +  '/diag_prog-val_'  + "%005d"%(step+1), data_x = inputs, data_y = outputs)

            if step > 0 and dQ_crm is not None:
                # save crm outputs for online labels
                y_4 = np.concatenate((soll_crm, sols_crm, solsd_crm, solld_crm, fsds_crm),axis=1)
                outputs = gen_outputs(dQ_crm, dS_crm, y_3, y_4)
                np.savez(online_data_path +  '/crm-val_'    + "%005d"%(step), data_x = inputs, data_y = outputs)
                dQ_crm = None

            # Online training logic
            if step > 0 and step % M == 0:
                #def online_training(all_models, args, M, step, online_data_path):
                curr_lr = float(online_training(all_models, optimizers, lr_schedulers, logger, step, online_data_path, args))


        
#   if (step < 10240):
#       np.savez(online_data_path + '/diag-extend_' + "%005d"%(step+1), omega = omega, pmid = pmid, pint = pint, s = s, zm = zm, zi = zi)

        step = step + 1
        print("Step", step, "integration\n")

        file = open(f"{data_buffer_path}/buffer_flag_2.log","w")
        file.close()
        if curr_lr is not None and curr_lr <= 1e-9:
            curr_lr = float(online_training(all_models, optimizers, lr_schedulers, logger, step, online_data_path, args, force_save=True))
            return curr_lr

#def run_cesm():
#    cur_dir = os.getcwd()
#    #os.chdir("/cust_users/chenj209/neuroGCM/scripts/reproduce_cases/")
#    os.chdir("/cust_users/chenj209/ONLINE_LEARNING_STARTUP//scripts/online_startup0322_checked/")
#    bashCommand = "./online_startup0322_checked.submit"
#    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
#    output, error = process.communicate() 
#    os.chdir(cur_dir)

def run_cesm():
    pattern = "job (\d+)"
    cur_dir = os.getcwd()
#    os.chdir("/cust_users/chenj209/neuroGCM/scripts/reproduce_cases/")
#    bashCommand = "./reproduce_cases.submit"
    os.chdir("/cust_users/chenj209/ONLINE_LEARNING_STARTUP//scripts/online_startup0322_checked/")
    bashCommand = "./online_startup0322_checked.submit"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    m = re.search(pattern, output.decode())
    os.chdir(cur_dir)
    return m.group(1)


#def cleanup():
#    bashCommand = "scancel -u chenj209"
#    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
#    output, error = process.communicate() 
#    while True:
#        bashCommand = "squeue -u chenj209"
#        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
#        output, error = process.communicate() 
#        if (len(output.split(b"\n")) == 2):
#          break
def cleanup(case_no):
    bashCommand = "scancel " + case_no
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    while True:
        bashCommand = "squeue --job " + case_no
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        if (len(output.split(b"\n")) == 2):
          break


def online_training(all_models, optimizers, lr_schedulers, logger, step, online_data_path, args, force_save=False):
    """
    Train current models with past M labels from CRM with one pass

    Parameters:
        models:
            model to be trained
        args:
            options to be used
        M: int
            Online learning training frequency
        step: int
            Current online learning step
    """

    print(f"[Online Training] Loading past {M} steps") 

    # Load M step data
    # load crm_output from last M step
    crm_file_pattern = "(crm).*\d{5}\.npz"
    start_step = step - M + 1
    if force_save:
        # force save happens when dynamics fails and there is less than M step run
        # in this case, use all data starting after last checkpoint
        last_checkpoint = (step // M) * M
        start_step = last_checkpoint + 1
    train_files = []
    for fn in os.listdir(online_data_path):
        m = re.match(crm_file_pattern, fn) 
        if m is not None:
            file_step = filename_to_idx(m.group(0))
            if start_step <= file_step <= step: 
                train_files.append(online_data_path + "/" + fn)
    print("[Online Learning] Loading train files:")
    print(train_files)
    if len(train_files) == 0:
        return
    training_set = Dataset(file_names=train_files, is_train=True, noise_std=args.noise_std)
    trainloader = data.DataLoader(training_set, shuffle=True, batch_size=args.train_batch, num_workers=args.workers)
    gpus_to_use = [0, 1, 3]

    save_log = [step / M + BASE_EPOCH]
    lrs = []
    mses = []
    criterion = nn.MSELoss()
    # One pass for all the models
    for mi, model_type in enumerate(MODELS_TO_TRAIN):
        model = all_models[model_type]

        # Define loss and optimizer
        optimizer = optimizers[model_type]
        lr_scheduler = lr_schedulers[model_type]


        # set models to train mode
        model.train()


        #train_losses = AverageMeter()
        #train_time_begin = time.time()

        train_losses = AverageMeter()
        current_iters = 0
        for iter, batch in enumerate(trainloader):

            # Dont' use lr scheduler for now
            # lr = lr_scheduler[args.lr_strategy](optimizer, args.lr, current_iters, len(trainloader) * args.epoch)
            if model_type == '0_29':
                batch[1] = batch[1][:, :30]
            if model_type == '30_59':
                batch[1] = batch[1][:, 30:60]
            if model_type == '60':
                batch[1] = batch[1][:, 60:61]
            if model_type == '61_65':
                batch[1] = batch[1][:, 61:66]
    #             if args.output_type == '61-65':
    #                 train_mse = train_tools.train_penalty(batch, model, criterion, optimizer)
    #             else:
            #train_mse = train_tools.train(batch, model, criterion, optimizer, gpus_to_use[mi])
            model.train()
            device = gpus_to_use[mi]

            points_x, points_y = batch
            points_x, points_y = (points_x.float()).cuda(device), (points_y.float()).cuda(device)
            
            
        #     print('!!!!!!!!!!!!!!!!!batch',points_x.size())  #1024 122???

            # compute output
            outputs_y = model(points_x)
            # print(outputs_y.size(), points_y.size())
            loss = criterion(outputs_y, points_y)

            # print(points_y)
            # print(torch.min(points_y))
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_mse = loss.item()
            train_losses.update(train_mse, batch[0].size(0))
            #train_losses.update(train_mse, batch[0].size(0))
            current_iters += 1
            print('training- | iters:{}/{}| lr:{:.4e} | train mse:{:.6f}|'.format(iter+1, len(trainloader), lr_scheduler.get_last_lr()[0], train_mse))
            #print('training- epoch:{}/{} | iters:{}/{}| lr:{:.6f} | train mse:{:.6f}|'.format(epoch, args.epoch, iter+1, len(trainloader), lr, train_mse))
        if ((step) / M - 1) % CKPT_FREQ == 0:
            print(f"[Online Learning] Saving checkpoint for {model_type}, Iter {int(step / M) + BASE_EPOCH}")
            train_tools.save_checkpoint({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, checkpoint=args.checkpoint + f"/{model_type}", filename='checkpoint_iter'+str(int(step/M+BASE_EPOCH))+'.pth.tar')
        if force_save:
            print(f"[Online Learning] Force Saving checkpoint for {model_type}, Iter {step // M + 1 + BASE_EPOCH}")
            train_tools.save_checkpoint({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, checkpoint=args.checkpoint + f"/{model_type}", filename='checkpoint_iter'+str(step//M+1+BASE_EPOCH)+'.pth.tar')
        lr_scheduler.step()
        lrs.append("{:.4e}".format(lr_scheduler.get_last_lr()[0]))
        mses.append(train_losses.avg)
    save_log.extend(lrs)
    save_log.extend(mses)
    logger.append(save_log)
    return lrs[0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file_path", type=str)
    parser.add_argument("-l", action="store_true")

    args = parser.parse_args()
    print(args)

    # load config
    """
    Configuration format:
    {
        "0-29": {
            "training_dataset": "40%sampled",
            "noise_level": "0.0",
            "epoch": 50,
        },
        "30-59": {
            "training_dataset": "40%sampled",
            "noise_level": "0.0",
            "epoch": 50,
        },
        "61-65": {
            "training_dataset": "40%sampled",
            "noise_level": "0.0",
            "epoch": 50,
        },
        "qtend[:10]=0.0": False,

        # optional
        "data_buffer_path": "/cust_users/chenj209/data_buffer/",
        "online_data_path": "/cust_users/chenj209/online_data_path/",
        "61-64": {
            "training_dataset": "40%sampled",
            "noise_level": "0.0",
            "epoch": 50,
        },
    }
    """
    if not args.l:
        with open(args.config_file_path, "r") as f:
            configs = [json.load(f)]
    else:
        with open(args.config_file_path, "r") as f:
            configs = json.load(f)

    print(configs)
    for i,config in enumerate(configs):
        parsed_config = parse_config(config)

    #    commands = f"CUDA_VISIBLE_DEVICES=1 python online_training.py --data_dir {DATA_DIR}" + " --output_type 0-29 --noise_std {} " \
    #               '--network {} --node_size {} --num_blocks {} --activation {} --dropout {} ' \
    #               '--train-batch {} --lr_strategy {} --lr {} --epoch {} --wd {} ' \
    #               '--checkpoint online_ckpt/{} --resume {}'.format(str(noise_std),
        #parser.add_argument("--optim", type=str)
        #parser.add_argument("--noise_std", type=float)
        #parser.add_argument("--train-batch", type=int)
        #parser.add_argument("--lr-strategy", type=str)
        #parser.add_argument("--wd", type=float)
        #parser.add_argument("--checkpoint", type=str)
        # change online learning related args to json values

        # check all checkpoint file paths are valid
        for model_type in MODEL_TYPES:
            if not os.path.isfile(parsed_config[model_type]["ckpt_path"]):
                raise Exception(f"Parsed path {parsed_config[model_type]['ckpt_path']} does not exist")

        # define post process flag to use 
        qtend_post_process = parsed_config["qtend[:10]=0.0"]

        # check running environment: databuffer empty, online folder empty

        parsed_config = parse_config(config)

        # print configuration
        print_config(parsed_config)


        # ask for input prompt
        if i==0:
            proceed = input("Proceed? (y/n)\n")
            if proceed != "y":
                raise Exception("Abort")

        while True:
            # load online training configs
            ol_args = [
                    "optim", 
                    "noise_std", 
                    "train_batch", 
                    "lr_strategy", 
                    "weight_decay", 
                    "checkpoint", 
                    "manualSeed", 
                    "workers", 
                    "lr", 
                    "momentum", 
                    "epoch",
                    "M",
                    "lr_step_size"
                    ]
            for arg in ol_args:
                args.__dict__[arg] = parsed_config[arg]
            print("[Online Learning] Args:\n", args)

            if args.manualSeed is None:
                args.manualSeed = 1
            random.seed(args.manualSeed)
            torch.manual_seed(args.manualSeed)
            np.random.seed(args.manualSeed)
            if args.M:
                M = args.M
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.manualSeed)
            for model_type in MODELS_TO_TRAIN:
                if not os.path.isdir(args.checkpoint + "/" + model_type):
                    mkdir_p(args.checkpoint + "/" + model_type)

            ckpt_pattern = "checkpoint_iter(\d+)\.pth"
            if os.path.isdir(args.checkpoint + "/0_29") and len(os.listdir(args.checkpoint + "/0_29")) > 0:
                resume_ckpt_paths = {}
                max_iter = BASE_EPOCH
                for model_type in MODELS_TO_TRAIN:
                    all_ckpts = os.listdir(args.checkpoint + "/" + model_type)
                    max_ckpt = ""
                    for ckpt in all_ckpts:
                        m = re.search(ckpt_pattern, ckpt)
                        if m is not None and int(m.group(1)) >= max_iter:
                            max_iter = int(m.group(1))
                            max_ckpt = ckpt
                    resume_ckpt_paths[model_type] = args.checkpoint + "/" + model_type + "/"  + max_ckpt
                BASE_EPOCH = max_iter
                print(f"[Online Learning] Resuming from BASE_EPOCH {BASE_EPOCH}:\n{resume_ckpt_paths}")

                all_models = load_ckpts_manual(
                               model029=resume_ckpt_paths["0_29"],
                               model3059=resume_ckpt_paths["30_59"],
                               model6164=parsed_config["61-64"]["ckpt_path"],
                               model6165=resume_ckpt_paths["61_65"],
                             )
            else:
                # load checkpoints
                all_models = load_ckpts_manual(
                               model029=parsed_config["0-29"]["ckpt_path"],
                               model3059=parsed_config["30-59"]["ckpt_path"],
                               model6164=parsed_config["61-64"]["ckpt_path"],
                               model6165=parsed_config["61-65"]["ckpt_path"],
                             )
                  
            #online_data_path = parsed_config["online_data_path"]
            online_data_path = parsed_config["online_data_path"]
            if online_data_path[-1] == "/":
                online_data_path = online_data_path[:-1]
            online_data_path += f"BASE{BASE_EPOCH}"
            data_buffer_path = parsed_config["data_buffer_path"]
            if not os.path.isdir(online_data_path):
                os.mkdir(online_data_path)
            for filename in os.listdir(data_buffer_path):
                os.remove(data_buffer_path + "/" + filename)
            assert(os.listdir(online_data_path) == [])
            assert(os.listdir(data_buffer_path) == [])

            # generate config file from running configuration into online data path
            generate_config(parsed_config, online_data_path)


            # run experiment
            case_no = run_cesm()
            curr_lr = run_experiment(all_models, online_data_path, data_buffer_path, qtend_post_process, args)
            cleanup(case_no)
            #if float(curr_lr) <= 1e-9:
            #    break
