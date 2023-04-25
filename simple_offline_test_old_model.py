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
#from dataloader_subset_files import Dataset
from dataloader_time_embedded import TimeDataset, get_inverse, DatasetDisk
#from dataloader_subset_files_offline_test import Dataset
from offline_test.dataloader_subset_files import Dataset
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

    output_type = args.output_type
    if args.output_type == "0-29" or args.output_type == "30-59":
        network = "resnet_output30"
    elif output_type == "61-65":
        network = "resnet_output5"
    else:
        raise Exception("output type problem")
    #network = "resnet_output30"
    #network = "resnet_output5"
    #resume = "ckpts_time/time_model029_0412/checkpoint.pth.tar"
    #resume = "ckpts_time/time_model029_0423/checkpoint_epoch10.pth.tar"
    #resume = "ckpts_longepoch_tkde/rmbaddata_wxnorm_subset_0-29_resnet_output30_nodesize512_num_blocks7_actrelu_bs1024_scheduler_coslr_lr0.001_ep100_noise0.001_wd0_dropout0/checkpoint.pth.tar"
    if args.resume == "baseline":
        if args.output_type  == "0-29":
            resume = "/cust_users/x-w19/nncam.ckpts/resmlp.2years.50epochs/0_29_nodesize512_num_blocks7_actrelu_bs1024_scheduler_coslr_lr0.001_ep50_noise0.0_wd0_dropout0/checkpoint.pth.tar"
        elif args.output_type == "30-59":
            resume = "/cust_users/x-w19/nncam.ckpts/resmlp.25GB.noise0.0/30_59_nodesize512_num_blocks7_actrelu_bs1024_scheduler_coslr_lr0.001_ep50_noise0.0_wd0_dropout0/checkpoint.pth.tar"
        elif args.output_type == "61-65":
            resume = "/cust_users/x-w19/nncam.ckpts/resmlp.newData.noise0.0/61_65_nodesize512_num_blocks7_actrelu_bs1024_scheduler_coslr_lr0.001_ep50_noise0.0_wd0_dropout0/checkpoint_epoch50.pth.tar"
    #resume = "ckpts_time/time_model3059_0423/checkpoint_epoch5.pth.tar"

    #resume = "ckpts_time/time_model6165_0423/checkpoint_epoch15.pth.tar"
    #resume = "ckpts_time/time_model029_0423/checkpoint_epoch15.pth.tar"
    #resume = "ckpts_time/time_model3059_0412/checkpoint.pth.tar"
    #resume = "ckpts_time/time_model6165_0412/checkpoint.pth.tar"
    #data_dir = "/home/users/data/nncam_data/image_testset/"
    data_dir = "/data/nncam_data/image_testset/"
    #data_dir = "/data/nncam_data/image_set/"
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
        model = models.ResNet_output30(node_size, activation, num_blocks)
    elif network == 'resnet_output5':
        model = models.ResNet_output5(node_size, activation, num_blocks)
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
    for fn in all_files:
        if "08691" in fn:
            all_files.remove(fn)
    all_files.sort()
    print("all_files len ", len(all_files))
    #test_files = all_files[100:200] + all_files[5000:5100]
    #test_idx = np.concatenate([np.arange(1, len(all_files), 400), np.arange(2,len(all_files),400)]) 
    #test_files = [all_files[i] for i in test_idx]
    #test_idx = np.concatenate([np.arange(1, len(all_files), 13), np.arange(2,len(all_files),13)]) 
    # [0,13,26,39...] + [1,14,27,40..]
    #test_idx = np.concatenate([np.arange(0, len(all_files), 13), np.arange(1,len(all_files),13)])
    #test_idx = np.arange(1,len(all_files),13)
    test_idx = np.random.choice(len(all_files), len(all_files)//2, replace=False)
    print(test_idx[:10])
    test_files = [all_files[i] for i in test_idx]
    print("Test file size: " ,len(test_files))
    print(test_files[:3])

    #testing_set = Dataset(test_files, is_train=False, noise_std=0, debug=True, )
    #testing_set = Dataset(test_files, is_train=False, noise_std=0.0, norm_type="01norm", debug=True, equator=False)
    testing_set = DatasetDisk(file_names=test_files, is_train=False, noise_std=0, output_normalized=False, silent=True)
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
        if output_type == '0-29':
            batch[1] = batch[1][:, :30]
        if output_type == '30-59':
            batch[1] = batch[1][:, 30:60]
        if output_type == '60':
            batch[1] = batch[1][:, 60:61]
        if output_type == '61-65':
            batch[1] = batch[1][:, 61:66]
       # if output_type == '0-29':
       #     batch[1] = get_inverse()["0_29"](batch[1][:, :30])
       # if output_type == '30-59':
       #     batch[1] = get_inverse()["30-59"](batch[1][:, 30:60])
       # if output_type == '60':
       #     batch[1] = get_inverse()["60"](batch[1][:, 60:61])
       # if output_type == '61-65':
       #     batch[1] = get_inverse()["61-65"](batch[1][:, 61:66])
        #test_mses = tools.test_de(batch, model, criterion)
        model.eval()
        with torch.no_grad():
            points_x, points_y = batch
            points_x, points_y = (points_x.float()).cuda(), (points_y.float()).cuda()
            #points_x, points_y = batch
            #points_x, points_y = (points_x.float()).cuda(), (points_y.float()).cuda()
            #points_x, points_y,  = batch
            #batch_x, batch_y, batch_y_raw, batch_y_label = batch
            #points_x, points_y = (points_x.float()).cuda(), (points_y.float()).cuda()
            #points_x = torch.cuda.FloatTensor(batch_x.numpy())
            if output_type == '0-29':
                batch[1] = batch[1][:,:30]
            if output_type == '30-59':
                batch[1] = batch[1][:,30:60]
            if output_type == '60':
                batch[1] = batch[1][:, 60:61]
            if output_type == '61-65':
                batch[1] = batch[1][:, 61:66]
           #if output_type == '0-29':
           #    batch_y_raw = batch_y_raw[:, :30]
           #if output_type == '30-59':
           #    batch_y_raw = batch_y_raw[:, 30:60]
           #if output_type == '60':
           #    batch_y_raw = batch_y_raw[:, 60:61]
           #if output_type == '61-65':
           #    batch_y_raw = batch_y_raw[:, 61:66]
           # points_y = (batch_y_raw.float()).cuda()

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





