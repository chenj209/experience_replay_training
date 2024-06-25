import os
import time
import sys
import numpy as np
import re
from collections import OrderedDict
#from tqdm.notebook import tqdm

#from dataloader_subset_files import Dataset
from torch.utils import data
import os
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn

import models
#import models_ex
import models_v2
import learnable_vector

network = 'resnet'    # 'resnet' / 'fcn'
train62 = False       #  True - 62  or  False - 122
os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2, 3'


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

def get_inverse():
  inverse = {}
  inverse[ '0_29'] = lambda x: (x+1)/2*(3.11e-6*2)-3.11e-6
  inverse['30_59'] = lambda x: (x+1)/2*(3.63*2)-3.63
  inverse['60']    = lambda x: (x+1)/2*(2.12e-6)
  inverse['61_64'] = lambda x: inverse_61_64(x)
  inverse['61_65'] = lambda x: inverse_61_65(x)

  return inverse

def load_resmlp_newformat2(ckpt_path, input_size, output_size, gpu_index=0, parallel=True):

    """
    Load four models from given ckpt path

    Parameters:
      model029(str): path to ckpt file of model for 0-29
      model3059(str): path to ckpt file of model for 30-59
      model6164(str): path to ckpt file of model for 61-64
      model6165(str): path to ckpt file of model for 61-65
    """



    # hyperparameters (fixed)
    num_blocks = 7
    node_size  = 512
    activation = 'relu'
    dropout    = 0
    model = models.ResMLP(input_size, output_size, node_size, activation, num_blocks)
    resume = ckpt_path
    real_gpu_id = gpu_index
    device = torch.device(f"cuda:{real_gpu_id}" if torch.cuda.is_available() else "cpu")
    print("\nLoading DNN model to {}".format(device))
    model = torch.nn.DataParallel(model).to(device)

    #print('------------------------output type: {}-----------------------'.format(output_type))
    print('Total number of params: {}'.format(sum(p.numel() for p in model.parameters())))

    checkpoint = torch.load(resume,map_location=device)['state_dict']
    if not parallel:
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            #name = k[7:] if k.startswith('module.') else k  # remove `module.` prefix
            name = "module."+k  # adding `module.` prefix
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(checkpoint)

    return model

def load_resmlp_newformat(ckpt_path, input_size, gpu_index=0):

    """
    Load four models from given ckpt path

    Parameters:
      model029(str): path to ckpt file of model for 0-29
      model3059(str): path to ckpt file of model for 30-59
      model6164(str): path to ckpt file of model for 61-64
      model6165(str): path to ckpt file of model for 61-65
    """



    # hyperparameters (fixed)
    num_blocks = 7
    node_size  = 512
    activation = 'relu'
    dropout    = 0
    model = models.ResMLP(input_size, 30, node_size, activation, num_blocks)
    resume = ckpt_path
    real_gpu_id = gpu_index
    print("\nLoading DNN model to GPU_{}".format(real_gpu_id))
    model = torch.nn.DataParallel(model, device_ids=[real_gpu_id])

    #print('------------------------output type: {}-----------------------'.format(output_type))
    print('Total number of params: {}'.format(sum(p.numel() for p in model.parameters())))

    checkpoint = torch.load(resume)['state_dict']
    model.load_state_dict(checkpoint)

    return model

def load_locresmlp(ckpt_path, input_size, latent_size, sub_region_mask, gpu_index=0, parallel=False):
    num_blocks = 7
    node_size  = 512
    activation = 'relu'
    dropout    = 0
    #model = models.ResMLP(input_size, 30, node_size, activation, num_blocks)
    model = learnable_vector.LearnableLocationResMLP(
        config={
           "input_size": input_size,
           "latent_size": latent_size,
        },
        input_size=input_size[0],
        output_size=30,
        m=512,
        activation='relu',
        num_blocks=7,
        sub_region_mask=sub_region_mask
    )
    resume = ckpt_path
    real_gpu_id = gpu_index
    print("\nLoading DNN model to GPU_{}".format(real_gpu_id))
    if parallel:
        model = torch.nn.DataParallel(model, device_ids=[real_gpu_id])
    else:
        model = model.cuda(real_gpu_id)

    #print('------------------------output type: {}-----------------------'.format(output_type))
    print('Total number of params: {}'.format(sum(p.numel() for p in model.parameters())))

    checkpoint = torch.load(resume)['state_dict']
    model.load_state_dict(checkpoint)

    return model

def load_models(model029, model3059, model6164, model6165, visible_gpus=[0,1,2,3],
                model_type=None):

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
    num_blocks = 7
    node_size  = 512
    activation = 'relu'
    dropout    = 0

    # Assign GPUs to 3 models
    gpu_index = 0


    for output_type in ['0_29', '30_59', '61_64', '61_65']:
        if output_type == '0_29' or output_type == '30_59':
            #if 'mlp' in model029.split("/")[-2]:
                #model = models.mlp_output30(node_size, activation, num_blocks)
            if "FCN" in model029.split("/")[-2]:
            #elif "FCN" in model029.split("/")[-2]:
                print("Loading FCN")
                model = models_v2.FCN(122,30)
            elif "Unet" in model029.split("/")[-2]:
                print("Loading Unet")
                model = models_v2.Unet(122,30)
            elif model_type == "post_rh":
                model = models.ResNet_output30_RH(node_size, activation, num_blocks)
            elif model_type == "pmid":
                model = models.ResMLP(151,30,node_size, activation, num_blocks)
            elif model_type == "pos":
                model = models.ResMLP(130,30,node_size, activation, num_blocks)
            elif model_type == "pos2":
                model = models.ResMLP(124,30,node_size, activation, num_blocks)
            elif model_type is not None and model_type.startswith("resmlp"):
                m = re.match(r"resmlp_(\d+)_(\d+)", model_type)
                input_size = int(m.group(1))
                output_size = int(m.group(2))
                print(f"Loading  Resmlp {input_size}:{output_size}")
                model = models.ResMLP(input_size,output_size,node_size, activation, num_blocks)
            else:
                print("Loading regular resnet30")
                model = models.ResNet_output30(node_size, activation, num_blocks)
        # elif output_type == '30_59':
        #     if "FCN" in model3059.split("/")[-2]:
        #         print("Loading FCN")
        #         model = models_v2.FCN(122,30)
        #     elif "Unet" in model3059.split("/")[-2]:
        #         print("Loading Unet")
        #         model = models_v2.Unet(122,30)
        #     else:
        #         model = models.ResNet_output30(node_size, activation, num_blocks)
        #elif output_type == '61_64':
            #model = models.ResNet_output4(node_size, activation, num_blocks)
        #elif output_type == '61_65':
            #model = models_ex.ResNet_output5(node_size, activation, num_blocks)
        resume = None
        if output_type == '0_29':
            #resume = path + 'rmbaddata_wxnorm_subset_mlp_0-29_mlp_output30_nodesize512_num_blocks7_actrelu_bs1024_scheduler_coslr_lr0.001_ep50_noise0.0_wd0_dropout0_sample_ratio0.3_norm_typeinput_level_norm/checkpoint.pth.tar'
            #resume = path + 'rmbaddata_wxnorm_subset_mlp_0-29_mlp_output30_nodesize512_num_blocks7_actrelu_bs1024_scheduler_coslr_lr0.001_ep50_noise0.0_wd0_dropout0_sample_ratio1.0_norm_type_input_level_norm/checkpoint.pth.tar'
            resume = model029
            print('\nloading the checkpoint: {}'.format(resume))

        if output_type == '30_59':
            #resume = path + 'rmbaddata_wxnorm_subset_30-59_resnet_output30_nodesize512_num_blocks7_actrelu_bs1024_scheduler_coslr_lr0.001_ep50_noise0.0_wd0_dropout0_sample_ratio0.3_norm_typeinput_level_norm/checkpoint.pth.tar'
            #resume = path + 'rmbaddata_wxnorm_subset_30-59_resnet_output30_nodesize512_num_blocks7_actrelu_bs1024_scheduler_coslr_lr0.001_ep50_noise0.0_wd0_dropout0_sample_ratio1.0_norm_type_input_level_norm/checkpoint.pth.tar'
            resume = model3059
            print('\nloading the checkpoint: {}'.format(resume))

        elif output_type == '61_64':

#            resume = path + 'rmbaddata_wxnorm_subset_nopenalty_61-65_resnet_output5_nodesize512_num_blocks7_actrelu_bs1024_scheduler_coslr_lr0.001_ep50_noise0.0_wd0_dropout0_sample_ratio0.3_norm_typeinput_level_norm/checkpoint.pth.tar'
            #resume = '/cust_users/x-w19/nncam.ckpts/resmlp.newData.61-65/61-65_resnet_ep50_noise0.01_wd0_dropout0/checkpoint_epoch50.pth.tar'
            resume = model6164
            print('\nloading the checkpoint: {}'.format(resume))

        elif output_type == '61_65':

#            resume = path + 'rmbaddata_wxnorm_subset_nopenalty_61-65_resnet_output5_nodesize512_num_blocks7_actrelu_bs1024_scheduler_coslr_lr0.001_ep50_noise0.0_wd0_dropout0_sample_ratio0.3_norm_typeinput_level_norm/checkpoint.pth.tar'
            #resume = '/cust_users/x-w19/nncam.ckpts/resmlp.newData.61-65/61-65_resnet_ep50_noise0.01_wd0_dropout0/checkpoint_epoch50.pth.tar'
            resume = model6165
            print('\nloading the checkpoint: {}'.format(resume))

        if resume is None:
            continue

        real_gpu_id = visible_gpus[gpu_index%len(visible_gpus)]
        print("\nLoading DNN model to GPU_{}".format(real_gpu_id))
        model = torch.nn.DataParallel(model, device_ids=[real_gpu_id])

        print('------------------------output type: {}-----------------------'.format(output_type))
        print('Total number of params: {}'.format(sum(p.numel() for p in model.parameters())))

        checkpoint = torch.load(resume)['state_dict']
        model.load_state_dict(checkpoint)

        all_models[output_type] = model

        gpu_index = gpu_index + 1

    return all_models


if __name__ == "__main__":
  from atm_log_process import parse_config, settings
  config = parse_config.load_config_file(settings.SRC_PATH + "/crash_case1/" + settings.CONFIG_FILE)
  all_models = load_models(
      config["0-29"]["ckpt_path"],
      config["30-59"]["ckpt_path"],
      config["61-64"]["ckpt_path"],
      config["61-65"]["ckpt_path"]
  )

