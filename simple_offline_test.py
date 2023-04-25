import argsparser
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import numpy as np
from utils import Logger, AverageMeter, mkdir_p
from nncam_data_explore.src.utility import filename_to_idx
from dataloader_subset_files import Dataset
from dataloader_time_embedded import TimeDatasetDisk, TimeDataset, get_inverse
from torch.utils import data
import models
import tools
import time
import glob

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

if __name__ == "__main__":
    import argparse
    import random
    random.seed(0)
    np.random.seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_type", "-ot", help="choose from 0-29, 30-59, 61-65")
    parser.add_argument("--resume", "-re", help="path to selected model")
    args = parser.parse_args()
    #output_type = "0-29"
    #output_type = "30-59"
    #output_type = "61-65"
    output_type = args.output_type
    if args.output_type == "0-29" or args.output_type == "30-59":
        network = "resnet_output30"
    elif output_type == "61-65":
        network = "resnet_output5"
    else:
        raise Exception("output type problem")
    #resume = "ckpts_time/time_model029_0412/checkpoint.pth.tar"
    #resume = "ckpts_time/time_model029_0423/checkpoint_epoch10.pth.tar"
    resume = args.resume
    #resume = "ckpts_time/time_model3059_0423/checkpoint_epoch5.pth.tar"
    #resume = "ckpts_time/time_model6165_0423/checkpoint_epoch15.pth.tar"
    #resume = "ckpts_time/time_model029_0423/checkpoint_epoch15.pth.tar"
    #resume = "ckpts_time/time_model3059_0412/checkpoint.pth.tar"
    #resume = "ckpts_time/time_model6165_0412/checkpoint.pth.tar"
    
    data_dir = "/data/nncam_data/image_testset/"
    #data_dir = "/home/users/data/nncam_data/image_set/"
    print("Resume: ", resume)
    print("Test set path: ", data_dir)
    print("Output type: ", output_type)
    dropout = 0
    weight_decay = 0
    activation = 'relu'
    num_blocks = 7
    node_size = 512
    if network == 'resnet':
        model = models.ResNet(node_size, activation)
    elif network == 'resnet_output30':
        model = models.ResNet_output30_Time(node_size, activation, num_blocks)
    elif network == 'resnet_output5':
        model = models.ResNet_output5_Time(node_size, activation, num_blocks)
    elif network == 'resnet_output1':
        model = models.ResNet_output1(node_size, activation, num_blocks)
    elif network == 'mlp_output30':
        model = models.mlp_output30(node_size, activation, num_blocks)
    elif network == 'mlp_output5':
        model = models.mlp_output5(node_size, activation, num_blocks)
    elif network == 'mlp_output1':
        model = models.mlp_output1(node_size, activation, num_blocks)
    else:
        model = None

    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True

    print('==> Resuming from checkpoint..')
    assert os.path.isfile(resume), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(resume)
    model.load_state_dict(checkpoint['state_dict'])
    all_files = glob.glob(data_dir+'/*')[2:]
    all_files.sort()
    #test_files = all_files[100:200] + all_files[5000:5100]

    # old sampling
    # [0,13,26,39...] + [1,14,27,40..]
    #test_idx = np.concatenate([np.arange(0, len(all_files), 13), np.arange(1,len(all_files),13)])
    test_idx = np.random.choice(len(all_files), len(all_files), replace=False)
    #test_idx = np.concatenate([np.arange(0, len(all_files), 13), np.arange(1,len(all_files),13)])
    print(test_idx[:10])
    test_files = [all_files[i] for i in test_idx]
    test_files.sort(key=filename_to_idx)
    print("Test file size: " ,len(test_files))
    print(test_files[:3])

    # for debug
    #test_files = test_files[:10]
    # dQ 1-e4 1-e3

    # current sampling
    # [1,14,27,40..] + [2,15,28,...]
    #test_idx = np.concatenate([np.arange(1, len(all_files), 13), np.arange(2,len(all_files),13)]) 
    #test_files = [all_files[i] for i in test_idx]
    # dQ 1-e13 1-e14
    #print("Test file size: " ,len(test_files))

    testing_set = TimeDatasetDisk(file_names=test_files, is_train=False, noise_std=0, output_normalized=False, silent=True)
    #testing_set = TimeDataset(file_names=test_files, is_train=False, noise_std=0, output_normalized=False, silent=True)
    testloader = data.DataLoader(testing_set, shuffle=False, batch_size=1, num_workers=1)

    test_losses = AverageMeter()
    loss_name = [output_type + '_r2: {:.4e}']
    test_time_begin = time.time()
    epoch = 1
    criterion = nn.MSELoss()
    y_pred = []
    y_gt = []
    for iter, batch in enumerate(testloader):
        suffix = 'testing- epoch:{}| iters:{}/{} |'.format(epoch, iter+1, len(testloader))
        batch[0] = batch[0].reshape(-1, batch[0].shape[-1])
        batch[1] = batch[1].reshape(-1, batch[1].shape[-1])

        #print("shape x:", batch[0].shape)
        #print("shape y:", batch[1].shape)
        #if output_type == '0-29':
        #    batch[1] = get_inverse()["0_29"](batch[1][:, :30])
        #if output_type == '30-59':
        #    batch[1] = get_inverse()["30-59"](batch[1][:, 30:60])
        #if output_type == '60':
        #    batch[1] = get_inverse()["60"](batch[1][:, 60:61])
        #if output_type == '61-65':
        #    batch[1] = get_inverse()["61-65"](batch[1][:, 61:66])
        
        if output_type == '0-29':
            batch[1] = batch[1][:,:30]
        if output_type == '30-59':
            batch[1] = batch[1][:,30:60]
        if output_type == '60':
            batch[1] = batch[1][:, 60:61]
        if output_type == '61-65':
            batch[1] = batch[1][:, 61:66]
        #test_mses = tools.test_de(batch, model, criterion)
        model.eval()
        with torch.no_grad():
            points_x, points_y = batch
            points_x, points_y = (points_x.float()).cuda(), (points_y.float()).cuda()

            # compute output
            outputs_y = model(points_x)
            if output_type == '0-29':
                outputs_y = get_inverse()["0_29"](outputs_y)
            if output_type == '30-59':
                outputs_y = get_inverse()["30_59"](outputs_y)
            if output_type == '60':
                outputs_y = get_inverse()["60"](outputs_y)
            if output_type == '61-65':
                outputs_y = get_inverse()["61_65"](outputs_y)

            loss = criterion(outputs_y, points_y)
            y_pred.append(outputs_y.detach().cpu().numpy())
            y_gt.append(points_y.detach().cpu().numpy())

        test_mse = loss.item()
        test_losses.update(test_mse, batch[0].size(0))
        suffix = suffix + loss_name[0].format(test_losses.avg)
        print(suffix)


    test_time = time.time() - test_time_begin
    #### save the log and ckpt ###################################
    print(f"Output type: {output_type}, test mse: {test_losses.avg}")

    y_pred = np.concatenate(y_pred, axis=0)
    y_gt = np.concatenate(y_gt, axis=0)
    if output_type == "0-29":
        var, std, mse, rmse, mae, max_ae, bias, r2 = Regression_Metrics(y_gt*3600*24*1000, y_pred*3600*24*1000)
    if output_type == "30-59":
        var, std, mse, rmse, mae, max_ae, bias, r2 = Regression_Metrics(y_gt*3600*24/1004, y_pred*3600*24/1004)
    if output_type == "61-65":
        var, std, mse, rmse, mae, max_ae, bias, r2 = Regression_Metrics(y_gt, y_pred)
    metrics = '{}\t{:.4}\t{:.4}\t{:.4}({:.4%})\t{:.4}\t{:.4}\t{:.4}({:.4%})\t'.format(50, r2, mse, rmse, rmse/std, mae, max_ae, bias, bias/std)
    log = ""
    log += "metrics:\n"
    log += "epoch\tr2\tmse\trmse\t\tmae\tmax_ae\tbias\n"
    log += metrics
    log += "\n"

    print(log)





