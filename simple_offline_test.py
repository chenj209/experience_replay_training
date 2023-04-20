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
from dataloader_subset_files import Dataset
from dataloader_time_embedded import TimeDataset
from torch.utils import data
import models
import tools
import time
import glob

if __name__ == "__main__":
    output_type = "0-29"
    network = "resnet_output30"
    resume = "ckpts_time/time_model029_0412/checkpoint.pth.tar"
    data_dir = "/home/users/data/nncam_data/image_testset/"
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
    all_files = glob.glob(data_dir+'/*')
    all_files.sort()
    test_files = all_files[100:200] + all_files[5000:5100]

    testing_set = TimeDataset(file_names=test_files, is_train=False, noise_std=0)
    testloader = data.DataLoader(testing_set, shuffle=False, batch_size=1024, num_workers=1)

    test_losses = AverageMeter()
    loss_name = [output_type + '_r2: {:.4e}']
    test_time_begin = time.time()
    epoch = 1
    criterion = nn.MSELoss()
    for iter, batch in enumerate(testloader):
        suffix = 'testing- epoch:{}| iters:{}/{} |'.format(epoch, iter+1, len(testloader))
        if output_type == '0-29':
            batch[1] = batch[1][:, :30]
        if output_type == '30-59':
            batch[1] = batch[1][:, 30:60]
        if output_type == '60':
            batch[1] = batch[1][:, 60:61]
        if output_type == '61-65':
            batch[1] = batch[1][:, 61:66]
        test_mses = tools.test_de(batch, model, criterion)
        test_mse = test_mses[0]
        test_losses.update(test_mse, batch[0].size(0))
        suffix = suffix + loss_name[0].format(test_losses.avg)
        print(suffix)


    test_time = time.time() - test_time_begin
    #### save the log and ckpt ###################################
    print(f"Output type: {output_type}, test mse: {test_losses.avg}")



