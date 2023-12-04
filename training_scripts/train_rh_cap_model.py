import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import numpy as np
import multiprocessing as mp
from torch.utils import data
import time
import glob
import json

import sys
sys.path.append(os.path.join(sys.path[0], ".."))
from utils import Logger, AverageMeter, mkdir_p
sys.path.append(os.path.join(sys.path[0], "..", "utils"))
import argsparser
import tools
from rh import torch_get_pmid, torch_cal_qsat_water
sys.path.append(os.path.join(sys.path[0], "..", "dataloader"))
from dataloader_refactor import DatasetDisk
sys.path.append(os.path.join(sys.path[0], "..", "models"))
import models

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

class RHMaskLoss(nn.Module):
    def __init__(self):
        super(RHMaskLoss, self).__init__()
        self.hyam = np.load(os.path.join(sys.path[0],"..", "consts", "hyam.npy"))
        self.hybm = np.load(os.path.join(sys.path[0],"..", "consts", "hybm.npy"))
        self.hyam = torch.tensor(self.hyam).cuda()
        self.hybm = torch.tensor(self.hybm).cuda()


    def forward(self, x_raw, output_type, output, target):
        x = x_raw
        pmid = torch_get_pmid(x[0,:,121],self.hyam, self.hybm)
        T = x[0,:,30:60]
        Q = x[0,:,:30]
        # add old nn pred
        # Q += get_inverse()['0_29'](x[:,122:])*30*60

        qsat_water = torch_cal_qsat_water(T, pmid)
        mask = (qsat_water == 0) * 1e-16
        qsat_water += mask
        #rh_mask = (Q > qsat_water).float().cuda()
        rh_mask = (Q <= qsat_water).float().cuda()
        criterion = nn.MSELoss()
        masked_base_loss = criterion(output*rh_mask, target*rh_mask)

        return masked_base_loss

def prep_models():
    # define model
    input_dim = 122
    if args.output_type == '0-29':
        output_dim = 30
    if args.output_type == '30-59':
        output_dim = 30
    if args.output_type == '60':
        output_dim = 1
    if args.output_type == '61-65':
        output_dim = 5

    # if args.network == 'FCN':
    #     model = models.FCN(input_dim, output_dim)
    # elif args.network == 'Unet':
    #     model = models.Unet(input_dim, output_dim)
    # else:
    #     model = None
    node_size = 512
    activation = "relu"
    num_blocks = 7
    model = models.ResMLP(input_dim, output_dim, node_size, activation, num_blocks)

    print('Total params: %.2f' % (sum(p.numel() for p in model.parameters())))

    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True
    return model

def prep_dataloaders():
    all_files = glob.glob(args.data_dir+'/*')[::13]#[::7]

    print('org file num:', len(all_files))
    for i in range(17507,17530):
        for file_name in all_files:
            if str(i) in file_name:
                all_files.remove(file_name)
    print('after file num:', len(all_files))

    
    for i in ['00001', '08690', '17522', '26210']:
        for file_name in all_files:
            if i in file_name:
                all_files.remove(file_name)
    print('hahahahahah after file num:', len(all_files))

    # test_idx = np.random.choice(len(all_files),len(all_files)//10,replace=False)
    test_idx = np.arange(len(all_files))[-(len(all_files)//10):]
    test_files = [all_files[i] for i in test_idx]
    train_files = [file_name for file_name in all_files if file_name not in test_files]
    # test_files = train_files
    print('train files: {} test files: {}'.format(len(train_files), len(test_files)))

    """
    Define dataloader
    """
    training_set = DatasetDisk(train_files, is_train=True, noise_std=args.noise_std)
    trainloader = data.DataLoader(training_set, shuffle=True, batch_size=args.train_batch, num_workers=args.workers)

    validation_set = DatasetDisk(test_files, is_train=False, noise_std=args.noise_std)
    validloader = data.DataLoader(validation_set, shuffle=False, batch_size=args.train_batch, num_workers=args.workers)
    return trainloader, validloader
        

def main(args):
    model = prep_models()

    trainloader, validloader = prep_dataloaders()


    """
    Define Residual Methods and Optimizer
    """

    if True:
        criterion = RHMaskLoss()
    if False:
        criterion = nn.MSELoss()
    if args.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.weight_decay)
    else:
        optimizer = None

    # Resume
    title = ''
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Epoch', 'LR', 'train mse', args.output_type +'_r2',  'train time', 'val time'])

    lr_scheduler = {'coslr': tools.cosine_lr,
                    'constant': tools.constant}

    # test_variance = {'0-29': 0.41921, '30-59': 0.96519, '60': 0.96958, '61-65':0.54228}

    early_stopper = EarlyStopper(patience=10,min_delta=0)
    # Train and test
    current_iters = 0
    min_test_loss = np.inf
    for epoch in range(args.epoch):
        """
        training
        """
        train_losses = AverageMeter()
        train_time_begin = time.time()
        for iter, batch in enumerate(trainloader):

            lr = lr_scheduler[args.lr_strategy](optimizer, args.lr, current_iters, len(trainloader) * args.epoch)
            # unnormalized_x = batch[2]
            # pmid = get_pmid_from_ps1d(unnormalized_x[0,:,121], hyam, hybm)
            # rh = cal_rh(unnormalized_x[0,:,:30], unnormalized_x[0,:,30:60], pmid)
            #print("rh shape:", rh.shape)
            # rh_mask = torch.max(rh,1).values <= 1
            if args.output_type == '0-29':
                batch[1] = batch[1][:, :, :30]
            if args.output_type == '30-59':
                batch[1] = batch[1][:, :, 30:60]
            if args.output_type == '60':
                batch[1] = batch[1][:, :, 60:61]
            if args.output_type == '61-65':
                batch[1] = batch[1][:, :, 61:66]
#                 train_mse = tools.train_penalty(batch, model, criterion, optimizer)
#             else:
            if False:
                train_mse = tools.train(batch, model, criterion, optimizer)
            if True:
                model.train()

                points_x, points_y, x_raw = batch
                points_x, points_y = (points_x.float()).cuda(), (points_y.float()).cuda()
                x_raw = x_raw.float().cuda()
                
                
            #     print('!!!!!!!!!!!!!!!!!batch',points_x.size())  #1024 122???

                # compute output
                outputs_y = model(points_x)
                # print(outputs_y.size(), points_y.size())
                loss = criterion(x_raw, args.output_type, outputs_y, points_y)

                # print(points_y)
                # print(torch.min(points_y))
                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_mse = loss.item()
            train_losses.update(train_mse, batch[0].size(0))
            current_iters += 1
            print('training- epoch:{}/{} | iters:{}/{}| lr:{:.6f} | train mse:{:.6f}|'.format(epoch, args.epoch, iter+1, len(trainloader), lr, train_mse))
        train_time = time.time() - train_time_begin

        """
        testing
        """
        #### define the loss seperately ## we have 7 mse accordingly

        test_losses = {}
        for i in range(1):
            test_losses[i] = AverageMeter()
        loss_name = [args.output_type + '_r2: {:.5f}']
        test_time_begin = time.time()
        for iter, batch in enumerate(validloader):
            suffix = 'testing- epoch:{}| iters:{}/{} |'.format(epoch, iter+1, len(validloader))
            if args.output_type == '0-29':
                batch[1] = batch[1][:, :, :30]
            if args.output_type == '30-59':
                batch[1] = batch[1][:, :, 30:60]
            if args.output_type == '60':
                batch[1] = batch[1][:, :, 60:61]
            if args.output_type == '61-65':
                batch[1] = batch[1][:, :, 61:66]
            if False:
                test_mses = tools.test_de(batch, model, criterion)
                for i in range(1):
                    test_mse = test_mses[i]
                    test_losses[i].update(test_mse, batch[0].size(0))
                    #suffix = suffix + loss_name[i].format(1 - test_losses[i].avg/test_variance[args.output_type])
                    suffix = suffix + loss_name[i].format(test_losses[i].avg)
            if True:
                model.eval()
                with torch.no_grad():
                    points_x, points_y, x_raw = batch
                    points_x, points_y = (points_x.float()).cuda(), (points_y.float()).cuda()
                    x_raw = x_raw.float().cuda()

                    # compute output
                    outputs_y = model(points_x)
                    test_mse = criterion(x_raw, args.output_type, outputs_y, points_y).item()
                #test_mse = test_mses[i]
                test_losses[i].update(test_mse, batch[0].size(0))
                # suffix = suffix + loss_name[i].format(1 - test_losses[i].avg/test_variance[args.output_type])
                suffix = suffix + loss_name[i].format(test_losses[i].avg)
            if iter%10 == 0:
#                 print('!!!!',test_losses[i].avg,test_variance[args.output_type])
                print(suffix)


        test_time = time.time() - test_time_begin
        #### save the log and ckpt ###################################
        save_log = [epoch, lr, train_losses.avg]
        for i in range(1):
            # save_log.append(1 - test_losses[i].avg/test_variance[args.output_type])
            save_log.append(test_losses[i].avg)

        save_log.append(train_time)
        save_log.append(test_time)
        logger.append(save_log)
        if early_stopper.early_stop(test_losses[i].avg):
            tools.save_checkpoint({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, checkpoint=args.checkpoint, filename='checkpoint_epoch'+str(epoch)+'.pth.tar')
            break

        if (epoch)%5 == 0:
            tools.save_checkpoint({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, checkpoint=args.checkpoint, filename='checkpoint_epoch'+str(epoch)+'.pth.tar')
            

        if test_losses[i].avg < min_test_loss:
            tools.save_checkpoint({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, checkpoint=args.checkpoint, filename='checkpoint_best_test_loss.pth.tar')
            min_test_loss = test_losses[i].avg


    logger.close()

if __name__ == '__main__':
    parser = argsparser.get_argparser()
    args = parser.parse_args()
    print(args)

    # set the fixed seed
    if args.manualSeed is None:
        args.manualSeed = 1
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.manualSeed)

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    with open(args.checkpoint + "/configs.txt", 'w+') as f:
        for (k, v) in args._get_kwargs():
            f.write(k + ' : ' + str(v) + '\n')

    main(args)
