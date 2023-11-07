import os
import torch
import time
#from tqdm.autonotebook import tqdm
import numpy as np
import pickle
import math
import sys
import json
import glob

#from atm_log_process.load_data_from_config import load_data_from_config
#from nncam_data_explore.stiff_scripts.filter_stiff import filter_stiff_structured_batched
#from atm_log_process import settings
#from atm_log_process.plot_log import compact_config_repr
#from nncam_models.load_models import load_models, get_inverse
#from .load_dataset import load_test_data, load_stiff_test_data, load_test_data_time
from .dataloader_time_embedded import TimeDataset
from .dataloader_subset_files import inverse_reshape_y, inverse_reshape_x
from . import models as local_models

# import consts
from .settings import *
#from nncam_data_explore.stiff_scripts import filter_stru

#PREFIX = "TESTRUN_"
PREFIX = "PAPER"
STIFF_EVAL = False

def load_ckpts_time(model029, model3059, model6164, model6165):
    
    '''
      eg. lab_path    = '/cust_users/x-w19/nncam.ckpts'
          resume_type = '/mlp.newData'
          noise_type  = 'noise0.01'
          epochs      = '50'           # epoch is a string instead of an integer

      crash: 
    '''
    
        
    # define model
    all_models = {}
    
    # hyperparameters (fixed)
    num_blocks = 7
    node_size  = 512
    activation = 'relu'
    dropout    = 0
    
    # Assign GPUs to 3 models
    gpu_index = 0


    for output_type in ['0_29', '30_59', '61_65']:
        if output_type == '0_29':
            if len(model029.split("/")) > 1 and 'mlp' in model029.split("/")[-2]:
                model = local_models.mlp_output30(node_size, activation, num_blocks)
            else:
                print("Loading time model 0_29")
                model = local_models.ResNet_output30_Time(node_size, activation, num_blocks)
        elif output_type == '30_59':
            print("Loading time model 30_59")
            model = local_models.ResNet_output30_Time(node_size, activation, num_blocks)
        elif output_type == '61_64':
            model = local_models.ResNet_output4(node_size, activation, num_blocks)
        elif output_type == '61_65':
            print("Loading time model 61_65")
            model = local_models.ResNet_output5_Time(node_size, activation, num_blocks)

        print("\nLoading DNN model to GPU_{}".format(gpu_index))
        model = torch.nn.DataParallel(model, device_ids=[gpu_index])

        print('------------------------output type: {}-----------------------'.format(output_type))
        print('Total number of params: {}'.format(sum(p.numel() for p in model.parameters())))

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
    max_ae = np.max(np.absolute(y_pred-y_true))   # Max absolute error
    
    bias = np.mean(y_pred-y_true)
    r2   = 1 - mse/var 
    
    return var, std, mse, rmse, mae, max_ae, bias, r2

def npz_out_file(tag, chunk_no):
  return f"{tag}-resmlp-offline-results-c{chunk_no}.npz"

def npz_out_dir(tag):
  return f"{PREFIX}{tag}-outs"

def check_outs(idx, tag):
  chunk_no = idx // CHUNK_SIZE
  return os.path.isfile(f"{npz_out_dir(tag)}/{npz_out_file(tag, chunk_no)}")

def load_outs(max_chunk_no, data_name,tag):
  data = []
  for chunk_no in range(max_chunk_no):
    chunk = np.load(f"{npz_out_dir(tag)}/{npz_out_file(tag, chunk_no)}", mmap_mode="w+")
    data.append(chunk[data_name])
  return np.concatenate((data), axis=0)

def store_outs(chunk_no, y1,y2,y3,y4,tag,force=False):
  if not os.path.isdir(npz_out_dir(tag)):
    os.mkdir(npz_out_dir(tag)) 
  while (y1.shape[0] > CHUNK_SIZE or force) and (y1.shape[0] > 0):
    qtend    = y1[:CHUNK_SIZE] # 0_29
    stend    = y2[:CHUNK_SIZE] # 30_59
    rad_diag = y3[:CHUNK_SIZE] # 61_64
    rad_prog = y4[:CHUNK_SIZE] # 61_65
#   qtend    = np.concatenate((y1[:CHUNK_SIZE]), axis = 0) # 0_29
#   stend    = np.concatenate((y2[:CHUNK_SIZE]), axis = 0) # 30_59
#   rad_diag = np.concatenate((y3[:CHUNK_SIZE]), axis = 0) # 61_64
#   rad_prog = np.concatenate((y4[:CHUNK_SIZE]), axis = 0) # 61_65
    np.savez(f"{npz_out_dir(tag)}/{npz_out_file(tag,chunk_no)}", qtend = qtend, stend = stend, rad_diag = rad_diag, rad_prog = rad_prog)

    # if saved, go to next chunk_no
    chunk_no,y1,y2,y3,y4 = chunk_no+1, y1[CHUNK_SIZE:], y2[CHUNK_SIZE:], y3[CHUNK_SIZE:], y4[CHUNK_SIZE:]
    # if not saved, keep current chunk_no
  return chunk_no, y1, y2, y3, y4

def report_stend(max_chunk_no, tag, y, load_y : bool):
  stend = load_outs(max_chunk_no, "stend",tag)
  var, std, mse, rmse, mae, max_ae, bias, r2 = Regression_Metrics(y[:,30:60]*3600*24/1004, stend*3600*24/1004)
  if STIFF_EVAL:
    if load_y:
      stend_gt = inverse_reshape_y(y[:,30:60])
      with open(f"{PREFIX}stend_gt_stiff.pkl", "wb") as f:
        pickle.dump(filter_stiff_structured_batched(stend_gt, 1), f)
    with open(f"{PREFIX}{tag}_stend_stiff.pkl", "wb") as f:
      pickle.dump(filter_stiff_structured_batched(inverse_reshape_y(stend), 1), f)

  del stend
  stend_metrics = '{}\t{:.4}\t{:.4}\t{:.4}({:.4%})\t{:.4}\t{:.4}\t{:.4}({:.4%})\t'.format(50, r2, mse, rmse, rmse/std, mae, max_ae, bias, bias/std)
  stend_log = ""
  stend_log += "stend metrics:\n"
  stend_log += "epoch\tr2\tmse\trmse\t\tmae\tmax_ae\tbias\n"
  stend_log += stend_metrics
  stend_log += "\n"

  return stend_log

def report_qtend(max_chunk_no, tag, y, load_y : bool):
  qtend = load_outs(max_chunk_no, "qtend",tag)
  var, std, mse, rmse, mae, max_ae, bias, r2 = Regression_Metrics(y[:,0:30]*3600*24*1000, qtend*3600*24*1000)
  # store stiff info
  if STIFF_EVAL:
    if load_y:
      qtend_gt = inverse_reshape_y(y[:,0:30])
      with open(f"{PREFIX}qtend_gt_stiff.pkl", "wb") as f:
        pickle.dump(filter_stiff_structured_batched(qtend_gt, 1), f)
   
    with open(f"{PREFIX}{tag}_qtend_stiff.pkl", "wb") as f:
      pickle.dump(filter_stiff_structured_batched(inverse_reshape_y(qtend), 1), f)

  del qtend
  qtend_metrics = '{}\t{:.4}\t{:.4}\t{:.4}({:.4%})\t{:.4}\t{:.4}\t{:.4}({:.4%})\t'.format(50, r2, mse, rmse, rmse/std, mae, max_ae, bias, bias/std)
  qtend_log = ""
  qtend_log += "qtend metrics:\n"
  qtend_log += "epoch\tr2\tmse\trmse\t\tmae\tmax_ae\tbias\n"
  qtend_log += qtend_metrics
  qtend_log += "\n"
  return qtend_log

def report_rad_prog(max_chunk_no, tag, y, load_y : bool):
  rad_prog = load_outs(max_chunk_no, "rad_prog",tag)
  var, std, mse, rmse, mae, max_ae, bias, r2 = Regression_Metrics(y[:,61:66], rad_prog)
  
  if STIFF_EVAL:
    if load_y:
      rad_prog_gt = inverse_reshape_y(y[:,61:66])
      with open(f"{PREFIX}rad_prog_gt_stiff.pkl", "wb") as f:
        pickle.dump(filter_stiff_structured_batched(rad_prog_gt, 1), f)
    with open(f"{PREFIX}{tag}_rad_prog_stiff.pkl", "wb") as f:
      pickle.dump(filter_stiff_structured_batched(inverse_reshape_y(rad_prog), 1), f)
  del rad_prog
  rad_prog = '{}\t{:.4}\t{:.4}\t{:.4}({:.4%})\t{:.4}\t{:.4}\t{:.4}({:.4%})\t'.format(50, r2, mse, rmse, rmse/std, mae, max_ae, bias, bias/std)
  rad_log = ""
  rad_log += "rad metrics:\n"
  rad_log += "epoch\tr2\tmse\trmse\t\tmae\tmax_ae\tbias\n"
  rad_log += rad_prog
  rad_log += "\n"

  return rad_log


def inference(config, dataloader, inverse : dict, do_inference: bool, load_y : bool):
  chunk_no = 0
  y_1 = np.zeros((0,30))
  y_2 = np.zeros((0,30))
  y_3 = np.zeros((0,4))
  y_4 = np.zeros((0,5))
  y_raw = []
  y_label = []
  y = None

  all_models = None
  #for idx, batch in enumerate(tqdm(dataloader)):
  for idx, batch in enumerate(dataloader):

    if load_y:
      # load y
      batch_y_raw = batch[2]
      batch_y_label = batch[3]
      y_raw.append(batch_y_raw.numpy())
      y_label.append(batch_y_label.numpy())

    if do_inference:
      if all_models is None:
        all_models = load_ckpts_time(
            config["0-29"]["ckpt_path"],
            config["30-59"]["ckpt_path"],
            config["61-64"]["ckpt_path"],
            config["61-65"]["ckpt_path"]
        )
      # do inference
      batch_x = batch[0]
      print(batch_x.shape)

      points_x = torch.cuda.FloatTensor(batch_x.numpy())

      with torch.no_grad():

        y_1 = np.concatenate(([y_1, inverse['0_29'](all_models['0_29'](points_x).detach().cpu().numpy())]),axis=0)
        y_2 = np.concatenate(([y_2, inverse['30_59'](all_models['30_59'](points_x).detach().cpu().numpy())]),axis=0)
        #y_3 = np.concatenate(([y_3, inverse['61_64'](all_models['61_64'](points_x).detach().cpu().numpy())]),axis=0)
        #y_3 = np.concatenate(([y_3, inverse['61_64'](all_models['61_64'](points_x).detach().cpu().numpy())]),axis=0)
        y_4 = np.concatenate(([y_4, inverse['61_65'](all_models['61_65'](points_x).detach().cpu().numpy())]),axis=0)
        #y_2 = append(inverse['30_59'](all_models['30_59'](points_x).detach().cpu().numpy()))
        #y_3.append(inverse['61_64'](all_models['61_64'](points_x).detach().cpu().numpy()))
        #y_4.append(inverse['61_65'](all_models['61_65'](points_x).detach().cpu().numpy()))
        chunk_no,y_1,y_2,y_3,y_4 = store_outs(chunk_no,y_1,y_2,y_3,y_4,tag)


      torch.cuda.empty_cache()


  if do_inference and len(y_1) > 0:
    chunk_no,_,_,_,_ = store_outs(chunk_no,y_1,y_2,y_3,y_4,tag,force=True)

  if load_y:
    y = np.concatenate((y_raw), axis=0)
    y_label = np.concatenate((y_label), axis=0)

  if not do_inference:
    chunk_no = len(dataloader) // CHUNK_SIZE + 1

  return chunk_no, y, y_label


if __name__ == "__main__":
#configs = load_data_from_config([
# {
#   "0-29": {
#     #"training_dataset": "0.4sampled",
#     #"noise": 0,
#     #"epoch": 50
#   },
#   #"qtend[:10]=0.0": False
#   "notes": "ZM_CONV"
# }
#]) + load_data_from_config([
# {
#   "0-29": {
#     #"training_dataset": "0.4sampled",
#     #"noise": 0,
#     #"epoch": 50
#   },
#   #"qtend[:10]=0.0": False
#   "notes": "zisocl"
# }
#])
  #if CHUNK_SIZE % BATCH_SIZE != 0:
  #  raise (f"CHUNK_SIZE is not divisible by BATCH_SIZE")
  #configs = load_data_from_config([
  #  {
  #    "0-29": {
  #      "training_dataset": "0.4sampled",
  #      "epoch": 50
  #    },
  ##   "notes": "cnn"
  #  }
    #{
    # "0-29": {
      #"training_dataset": "0.4sampled",
      #"epoch": 50
    # },
    #"notes": "unet_noise0"
    #},
  #  ])
  with open(sys.argv[1], "r") as f:
      configs = [json.load(f)]
  for config in configs:
    print(compact_config_repr(config), ":", config.get("notes"))
  process = input("Proceed?")
  if process != "y":
    quit()
  torch.multiprocessing.set_sharing_strategy('file_system')
  TEST_DATAPATH = "/home/users/data/nncam_data/image_testset/"
  #TEST_DATAPATH = "/data/nncam_data/image_set/"
  #TEST_DATAPATH = "/share1/neuroGCM/chenj209_temp/0-29_0.4sampled_noise0_ep50_30-59_0.4sampled_noise0_ep50_61-65_newdata_noise0_ep50/"

  metric_eval = True

  #testloader = load_test_data(TEST_DATAPATH)

  torch.cuda.empty_cache()

  testloader = None
  data_size = -1

  # load y_raw data for once
  y = None
  load_y = True

  # create data loader
  #testloader = load_stiff_test_data(TEST_DATAPATH, thres=1, data_pattern=".*\d{5}\.npz")
  #testloader = load_test_data_time(TEST_DATAPATH, sampled=True, batch_size=BATCH_SIZE)
  test_files = glob.glob(TEST_DATAPATH + "/*")
  test_files.sort()
  test_files = test_files[100:200]
  test_set = TimeDataset(test_files, is_train=False, noise_std=0)
  testloader = data.DataLoader(test_set, shuffle=False, batch_size=1,num_workers=1)

  #testloader = load_test_data(TEST_DATAPATH, sampled=True, batch_size=BATCH_SIZE)
  data_size = len(testloader)

  print("Testloader data size", data_size)
  for iter, batch in enumerate(testloader):


  inverse = get_inverse()
  config1 = configs[0]

  max_chunk_no = math.ceil(data_size*BATCH_SIZE / CHUNK_SIZE)
  for config in configs:

    tag = compact_config_repr(config)

    # determine if we need to do inference
    do_inference = not check_outs(data_size, tag)

    # if y is loaded and model outs are saved
    if not load_y and not do_inference:
      print(f"Skipping {tag}")

    else:
      _max_chunk_no, y_loaded, y_label = inference(config, testloader, inverse, do_inference, load_y)
      if not (max_chunk_no == _max_chunk_no):
        print(max_chunk_no, _max_chunk_no)

      # load y only once
      if load_y:
        y = y_loaded
        y_label_by_document = inverse_reshape_y(y_label)[:,0,0,0]
        np.save(f"{PREFIX}y-label.npy", y_label_by_document)

    if metric_eval:
      metrics_log = open(f"{PREFIX}{tag}_offline_metrics.log", "w")

      stend_log = report_stend(max_chunk_no, tag, y, load_y)
      print(stend_log)
      metrics_log.write(stend_log)

      qtend_log = report_qtend(max_chunk_no, tag, y, load_y)
      print(qtend_log)
      metrics_log.write(qtend_log)


      rad_log = report_rad_prog(max_chunk_no, tag, y, load_y)
      print(rad_log)
      metrics_log.write(rad_log)

      metrics_log.close()
      load_y = False




