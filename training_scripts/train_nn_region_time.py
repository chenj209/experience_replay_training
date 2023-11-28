import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import numpy as np
from torch.utils import data
import time
import glob
import sys
# import models_v2 as models

sys.path.append(os.path.join(sys.path[0], "..", "models"))
import models
sys.path.append(os.path.join(sys.path[0], ".."))
from utils import Logger, AverageMeter, mkdir_p
sys.path.append(os.path.join(sys.path[0], "..", "dataloader"))
from dataloader_time_embedded import TimeDatasetDisk as DatasetDisk
sys.path.append(os.path.join(sys.path[0], "..", "utils"))
import argsparser
import tools

def main(args):
    # define model
    input_dim = 309
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

    print('the length of train loader (total iters): {}'.format(len(trainloader)))

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
        logger.set_names(['Epoch', 'LR', 'train mse', args.output_type +'_mse',  'chidao_mse', 'train time', 'val time'])

    lr_scheduler = {'coslr': tools.cosine_lr,
                    'constant': tools.constant}


    #test_variance = {'0-29': 0.41921, '30-59': 0.96519, '60': 0.96958, '61-65':0.54228}
    #test_variance = {'0-29': 20731.56910, '30-59': 3233.22239, '60': 0.20854, '61-65':1.010511}
    
    # Train and validate
    if args.region_mask == "all":
        region_mask = np.ones((96,144))[None, None, :, :]
    else:
        region_mask = np.load(args.region_mask)[None, None, :, :]
    region_mask = np.transpose(region_mask, (0, 2, 3, 1))
    region_mask = np.reshape(region_mask, (-1, region_mask.shape[-1]))
    region_mask = (region_mask[:,0]==1)
    current_iters = 0
    for epoch in range(args.epoch):
        """
        training
        """
        train_losses = AverageMeter()
        train_time_begin = time.time()
        for iter, batch in enumerate(trainloader):
            lr = lr_scheduler[args.lr_strategy](optimizer, args.lr, current_iters, len(trainloader) * args.epoch)
            # land_sea_flag = (args.land_sea_type == "land")
            batch[0] = batch[0][:, region_mask, :122]
            if args.output_type == '0-29':
                batch[1] = batch[1][:, region_mask, :30]
            if args.output_type == '30-59':
                batch[1] = batch[1][:, region_mask, 30:60]
            if args.output_type == '60':
                batch[1] = batch[1][:, region_mask, 60:61]
            if args.output_type == '61-65':
                batch[1] = batch[1][:, region_mask, 61:66]

            #print("debug:", batch[0].shape, batch[1].shape)
            train_mse = tools.train(batch[:2], model, criterion, optimizer)
            train_losses.update(train_mse, batch[0].size(0))
            current_iters += 1
            if iter%10 == 0:
                print('training- epoch:{}| iters:{}/{}| lr:{:.6f} | train mse:{:.6f}|'.format(epoch, iter+1, len(trainloader), lr, train_mse))
        train_time = time.time() - train_time_begin

        """
        testing
        """
        test_losses = {}
        for i in range(2):
            test_losses[i] = AverageMeter()
        loss_name = [args.output_type + '_mse: {:.5f}', 'chidao_mse: {:.5f}']
        test_time_begin = time.time()
        for iter, batch in enumerate(validloader):
            suffix = 'testing- epoch:{}| iters:{}/{} |'.format(epoch, iter+1, len(validloader))
            batch[0] = batch[0][:, region_mask, :122]
            if args.output_type == '0-29':
                batch[1] = batch[1][:, region_mask, :30]
            if args.output_type == '30-59':
                batch[1] = batch[1][:, region_mask, 30:60]
            if args.output_type == '60':
                batch[1] = batch[1][:, region_mask, 60:61]
            if args.output_type == '61-65':
                batch[1] = batch[1][:, region_mask, 61:66]

            #batch[0][:,:,121] = 0
            test_mses = tools.test_de(batch[:2], model, criterion)
            for i in range(2):
                test_mse = test_mses[i]
                test_losses[i].update(test_mse, batch[0].size(0))
                #suffix = suffix + loss_name[i].format(1 - test_losses[i].avg/test_variance[args.output_type])
                suffix = suffix + loss_name[i].format(test_losses[i].avg)
                
                
#             print('test!!!!',test_losses[i].avg,test_variance[args.output_type])
            if iter%10 == 0:
#                 print('!!!!',test_losses[i].avg,test_variance[args.output_type])
                print(suffix)
        test_time = time.time() - test_time_begin

        #### save the log and ckpt ###################################
        save_log = [epoch, lr, train_losses.avg]
        for i in range(2):
            #save_log.append(1 - test_losses[i].avg/test_variance[args.output_type])
            save_log.append(test_losses[i].avg)

        save_log.append(train_time)
        save_log.append(test_time)
        logger.append(save_log)
        if (epoch+1)%5 == 0:
            tools.save_checkpoint({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, checkpoint=args.checkpoint, filename='checkpoint_epoch'+str(epoch+1)+'.pth.tar')
        tools.save_checkpoint({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, checkpoint=args.checkpoint)

    logger.close()

if __name__ == '__main__':
    parser = argsparser.get_argparser()
    parser.add_argument('--region_mask', type=str, help='path to region mask npy file')
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
