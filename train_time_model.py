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

def main(args):
    # define model
    if args.network == 'resnet':
        model = models.ResNet(args.node_size, args.activation)
    elif args.network == 'resnet_output30':
        model = models.ResNet_output30_Time(args.node_size, args.activation, args.num_blocks)
    elif args.network == 'resnet_output5':
        model = models.ResNet_output5_Time(args.node_size, args.activation, args.num_blocks)
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

    #test_idx = np.random.choice(len(all_files),len(all_files)//10,replace=False)
    #test_idx = np.random.choice(len(all_files),len(all_files)//10,replace=False)
    #train_files = [file_name for file_name in all_files if file_name not in test_files]
    test_file_count = len(all_files)//20
    test_files = all_files[-test_file_count:]
    #train_files = all_files[:-test_file_count]
    train_files = all_files[:test_file_count*2]
    # test_files = train_files
    print('train files: {} test files: {}'.format(len(train_files), len(test_files)))

    """
    Define Residual Methods and Optimizer
    """
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

    training_set = TimeDataset(file_names=train_files, is_train=True, noise_std=args.noise_std)
    trainloader = data.DataLoader(training_set, shuffle=True, batch_size=args.train_batch, num_workers=args.workers)

    testing_set = TimeDataset(file_names=test_files, is_train=False, noise_std=args.noise_std)
    testloader = data.DataLoader(testing_set, shuffle=False, batch_size=args.train_batch, num_workers=args.workers)
    early_stopper = EarlyStopper(patience=5,min_delta=0)
    # Train and test
    current_iters = 0
    for epoch in range(args.epoch):
        """
        training
        """
        train_losses = AverageMeter()
        train_time_begin = time.time()
        for iter, batch in enumerate(trainloader):

            lr = lr_scheduler[args.lr_strategy](optimizer, args.lr, current_iters, len(trainloader) * args.epoch)
            if args.output_type == '0-29':
                batch[1] = batch[1][:, :30]
            if args.output_type == '30-59':
                batch[1] = batch[1][:, 30:60]
            if args.output_type == '60':
                batch[1] = batch[1][:, 60:61]
            if args.output_type == '61-65':
                batch[1] = batch[1][:, 61:66]
#             if args.output_type == '61-65':
#                 train_mse = tools.train_penalty(batch, model, criterion, optimizer)
#             else:
            train_mse = tools.train(batch, model, criterion, optimizer)
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
            suffix = 'testing- epoch:{}| iters:{}/{} |'.format(epoch, iter+1, len(testloader))
            if args.output_type == '0-29':
                batch[1] = batch[1][:, :30]
            if args.output_type == '30-59':
                batch[1] = batch[1][:, 30:60]
            if args.output_type == '60':
                batch[1] = batch[1][:, 60:61]
            if args.output_type == '61-65':
                batch[1] = batch[1][:, 61:66]
            test_mses = tools.test_de(batch, model, criterion)
            for i in range(1):
                test_mse = test_mses[i]
                test_losses[i].update(test_mse, batch[0].size(0))
                # suffix = suffix + loss_name[i].format(1 - test_losses[i].avg/test_variance[args.output_type])
                suffix = suffix + loss_name[i].format(test_losses[i].avg)
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
            tools.save_checkpoint({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, checkpoint=args.checkpoint, filename='checkpoint_epoch'+str(epoch+1)+'.pth.tar')
            break

        if (epoch+1)%5 == 0:
            tools.save_checkpoint({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, checkpoint=args.checkpoint, filename='checkpoint_epoch'+str(epoch+1)+'.pth.tar')
            

        tools.save_checkpoint({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, checkpoint=args.checkpoint)


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
