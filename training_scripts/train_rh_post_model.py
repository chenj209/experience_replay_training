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
#from dataloader_time_embedded import TimeDatasetDisk
from dataloader_rh_post_process import PostDatasetDisk, get_inverse, inverse_q, inverse_ps, inverse_T
from offline_test_all_models_thick import load_models_single_gpu as load_models
import multiprocessing as mp

from torch.utils import data
import models
import tools
import time
import glob
import json

HYAM = np.load("npy_outs/hyam.npy")
HYBM = np.load("npy_outs/hybm.npy")
def get_ps_lvl(ps):
    # Ensure hybm and hyam are torch tensors
    hybm = torch.tensor(HYBM).cuda()
    hyam = torch.tensor(HYAM).cuda()
    
    # Use unsqueeze to add an extra dimension to ps, equivalent to np.newaxis
    ps_level = ps.unsqueeze(1) * hybm.unsqueeze(0)
    ps_level += hyam.unsqueeze(0) * 100000
    
    return ps_level

def cal_qsat_water(t, p):
    # t: temperature (30, 96, 144)
    # p: pressure (30,96, 144)
    
    # Constants
    ps = torch.tensor([1013.246], device='cuda')
    ts = torch.tensor([373.16], device='cuda')
    e1 = 11.344 * (1.0 - t / ts)
    e2 = -3.49149 * (ts / t - 1.0)
    f1 = -7.90298 * (ts / t - 1.0)
    f2 = 5.02808 * torch.log10(ts / t)
    f3 = -1.3816 * (10.0 ** e1 - 1.0) / 10000000.0
    f4 = 8.1328 * (10.0 ** e2 - 1.0) / 1000.0
    f5 = torch.log10(ps)
    f = f1 + f2 + f3 + f4 + f5
    es = (10.0 ** f) * 100.0
    epsqs = torch.tensor([0.622], device='cuda')  # Assuming epsqs is a constant with a value of 0.622
    qsat_water = epsqs * es / (p - (1.0 - epsqs) * es)  # saturation w/respect to liquid only
    qsat_water = torch.where(qsat_water < 0.0, torch.tensor([1.0], device='cuda', dtype=qsat_water.dtype), qsat_water)
    return qsat_water

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
    def __init__(self, inverse_y):
        super(RHMaskLoss, self).__init__()
        self.inverse_y = inverse_y

    def forward(self, x, output_type, output, target):
        ps = inverse_ps(x[:,121])
        ps_lvl = get_ps_lvl(ps)
        T = inverse_T(x[:,30:60])
        Q = inverse_q(x[:,:30])
        # add old nn pred
        #Q += get_inverse()['0_29'](x[:,122:])*30*60

        qsat_water = cal_qsat_water(T, ps_lvl)
        mask = (qsat_water == 0) * 1e-16
        qsat_water += mask
        rh_mask = (Q > qsat_water).float().cuda()
        criterion = nn.MSELoss()
        masked_base_loss = criterion(output*rh_mask, target*rh_mask)

        return masked_base_loss

        

def main(args):

    print(f"Loading {args.non_rh_model_config}")
    with open(args.non_rh_model_config, "r") as f:
        non_rh_model_config = json.load(f)
    
    if args.output_type == "0-29":
        non_rh_model = load_models(
                non_rh_model_config["0-29"]["ckpt_path"],
                None,
                None,
                None
                )["0_29"]
    if args.output_type == "30-59":
        non_rh_model = load_models(
                None,
                non_rh_model_config["30-59"]["ckpt_path"],
                None,
                None
                )["30_59"]
    # define model
    if args.network == 'resnet':
        model = models.ResNet(args.node_size, args.activation)
    elif args.network == 'resnet_output30':
        model = models.ResNet_output30_RH(args.node_size, args.activation, args.num_blocks)
    elif args.network == 'resnet_output5':
        model = models.ResNet_output5(args.node_size, args.activation, args.num_blocks)
    elif args.network == 'resnet_output1':
        model = models.ResNet_output1(args.node_size, args.activation, args.num_blocks)
    elif args.network == 'mlp_output30':
        model = models.mlp_output30(args.node_size, args.activation, args.num_blocks)
    elif args.network == 'mlp_output5':
        model = models.mlp_output5(args.node_size, args.activation, args.num_blocks)
    elif args.network == 'mlp_output1':
        model = models.mlp_output1(args.node_size, args.activation, args.num_blocks)
    else:
        model = None

    print('Total params: %.2f' % (sum(p.numel() for p in model.parameters())))

    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True


    #################### 屏蔽掉一些可能存在异常的数据集 ###############################
    #all_files = glob.glob(args.data_dir+'/*')[::13]#[::7]
    all_files = glob.glob(args.data_dir+'/*')

    print('org file num:', len(all_files))
    #for i in range(17507,17530):
    for i in range(17507,17531):
        for file_name in all_files:
            if str(i) in file_name:
                all_files.remove(file_name)
    print('after file num:', len(all_files))

    
    for i in ['00001', '00002', '00003', '08690', '08691', '17522', '17523','26210','26211']:
        for file_name in all_files:
            if i in file_name:
                all_files.remove(file_name)
    print('hahahahahah after file num:', len(all_files))

    #test_idx = np.random.choice(len(all_files),len(all_files)//20,replace=False)
    #test_idx = np.random.choice(len(all_files),len(all_files)//10,replace=False)
    test_idx = np.arange(len(all_files))[-len(all_files)//10:]
    train_files = [all_files[i] for i in range(len(all_files)) if i not in test_idx]
    random.shuffle(train_files)
    test_files = [all_files[i] for i in test_idx]
    #test_file_count = len(all_files)//20
    #test_files = all_files[-test_file_count:]
    #train_files = all_files[:-test_file_count]
    #train_idx = np.concatenate([np.arange(1,len(all_files),13), np.arange(0,len(all_files),13)])
    #train_idx = np.arange(0, len(all_files))
    #print(len(all_files))
    #print(max(train_idx))
    #train_files = [all_files[i] for i in train_idx]
    #test_all_files = glob.glob("/home/users/data/nncam_data/image_testset/")
    #test_idx = np.concatenate([np.arange(1,len(test_all_files),26), np.arange(0,len(test_all_files),26)])
    #test_idx = np.random.choice(len(test_all_files), len(test_all_files
    #test_files = [test_all_files[i] for i in test_idx]
    print('train files: {} test files: {}'.format(len(train_files), len(test_files)))

    """
    Define Residual Methods and Optimizer
    """
    inverse_y = get_inverse()[args.output_type.replace("-", "_")]

    if True:
        criterion = RHMaskLoss(inverse_y)
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

    mp.set_start_method("spawn")
    training_set = PostDatasetDisk(file_names=train_files, is_train=True, noise_std=args.noise_std, silent=True, non_rh_model=non_rh_model)
    trainloader = data.DataLoader(training_set, shuffle=False, batch_size=1, num_workers=args.workers)

    testing_set = PostDatasetDisk(file_names=test_files, is_train=False, noise_std=args.noise_std, silent=True, non_rh_model=non_rh_model)
    testloader = data.DataLoader(testing_set, shuffle=False, batch_size=1, num_workers=args.workers)
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

            batch[0] = batch[0].reshape(-1, batch[0].shape[-1])
            batch[1] = batch[1].reshape(-1, batch[1].shape[-1])
            lr = lr_scheduler[args.lr_strategy](optimizer, args.lr, current_iters, len(trainloader) * args.epoch)
            if args.output_type == '0-29':
                batch[1] = batch[1][:, :30] - batch[0][:,122:]
            if args.output_type == '30-59':
                batch[1] = batch[1][:, 30:60] - batch[0][:,122:]
            if args.output_type == '60':
                batch[1] = batch[1][:, 60:61]
            if args.output_type == '61-65':
                batch[1] = batch[1][:, 61:66]
#             if args.output_type == '61-65':
#                 train_mse = tools.train_penalty(batch, model, criterion, optimizer)
#             else:
            if False:
                train_mse = tools.train(batch, model, criterion, optimizer)
            if True:
                model.train()

                points_x, points_y = batch
                points_x, points_y = (points_x.float()).cuda(), (points_y.float()).cuda()
                
                
            #     print('!!!!!!!!!!!!!!!!!batch',points_x.size())  #1024 122???

                # compute output
                outputs_y = model(points_x)
                # print(outputs_y.size(), points_y.size())
                loss = criterion(points_x, args.output_type, outputs_y, points_y)

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
        for iter, batch in enumerate(testloader):
            batch[0] = batch[0].reshape(-1, batch[0].shape[-1])
            batch[1] = batch[1].reshape(-1, batch[1].shape[-1])
            suffix = 'testing- epoch:{}| iters:{}/{} |'.format(epoch, iter+1, len(testloader))
            if args.output_type == '0-29':
                batch[1] = batch[1][:, :30] - batch[0][:,122:]

            if args.output_type == '30-59':
                batch[1] = batch[1][:, 30:60] - batch[0][:,122:]

            if args.output_type == '60':
                batch[1] = batch[1][:, 60:61]
            if args.output_type == '61-65':
                batch[1] = batch[1][:, 61:66]
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
                    points_x, points_y = batch
                    points_x, points_y = (points_x.float()).cuda(), (points_y.float()).cuda()

                    # compute output
                    outputs_y = model(points_x)
                    test_mse = criterion(points_x, args.output_type, outputs_y, points_y).item()
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
