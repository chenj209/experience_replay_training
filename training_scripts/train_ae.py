import os
import time
import json
import glob
import random
import sys
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import numpy as np
from torch.utils import data
from datetime import datetime
# from torch.utils.data.dataloader import default_collate

sys.path.append(os.path.join(sys.path[0], "..", "models"))
import autoencoder
sys.path.append(os.path.join(sys.path[0], ".."))
from utils import Logger, AverageMeter, mkdir_p
sys.path.append(os.path.join(sys.path[0], "..", "utils"))
import argsparser
import tools
from data_shape import to_inference_shape, inverse_to_inference_shape
sys.path.append(os.path.join(sys.path[0], "..", "dataloader"))
from dataloader_newformat import DatasetDisk

def prep_dataloaders(args):
    #data_dir = args.data_dir
    #if not os.path.isdir(data_dir):
        # data_dir = "./data/"
    data_dir = "/pscratch/sd/c/chenjd21/spcam_new_data/"
    col_names = np.loadtxt(data_dir + "/col_names.txt", dtype=str)
    col_names_x = ["QL", "T_nn_in", "dqvls_nn_in", "dTls_nn_in", "SOLIN", "SPPS"]
    prev_ex_vars = ["qtend_check", "stend_check", "SOLL", "SOLLD", "SOLS", "SOLSD", "FSDS"]
    col_names_y = ["qtend_check"]
    data_means = dict(np.load(data_dir + "/data_means.npz"))
    data_stds = dict(np.load(data_dir + "/data_stds.npz"))
    #################### 屏蔽掉一些可能存在异常的数据集 ###############################
    #all_files = glob.glob(args.data_dir+'/*')[::13]#[::7]
    all_files = glob.glob(data_dir+'/*.npy')
    all_files.sort()
    all_files = all_files[:35040]

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

    test_idx = np.arange(len(all_files))[-(len(all_files)//10):]
    train_files = [all_files[i] for i in range(len(all_files)) if i not in test_idx]
    valid_files = [all_files[i] for i in test_idx]
    print('train files: {} test files: {}'.format(len(train_files), len(valid_files)))
    # training_set = DatasetDisk(file_names=train_files, is_train=True, noise_std=args.noise_std, multistep=int(args.multistep))
    training_set = DatasetDisk(
        train_files,
        col_names,
        col_names_x,
        col_names_y,
        data_stds,
        data_means,
        is_train=True,
        noise_std=0,
        multistep=int(args.multistep),
        sample_rate=args.sample_rate,
        prev_ex_vars=prev_ex_vars+args.ex_input,
        image=True)
    trainloader = data.DataLoader(training_set, shuffle=True, batch_size=args.train_batch, num_workers=args.workers)

    valid_set = DatasetDisk(
        valid_files,
        col_names,
        col_names_x,
        col_names_y,
        data_stds,
        data_means,
        is_train=False,
        noise_std=0,
        multistep=int(args.multistep),
        sample_rate=args.sample_rate,
        prev_ex_vars=prev_ex_vars+args.ex_input,
        image=True)
    validloader = data.DataLoader(valid_set, shuffle=False, batch_size=args.train_batch, num_workers=args.workers)
    return trainloader, validloader

def prep_models(args):
    with open(args.ae_config, 'r') as f:
        ae_config = json.load(f)
    print(f"Model input size: {ae_config['input_size']}")
    #model = autoencoder.AutoencoderResMLP(input_size, 30, args.node_size, args.activation, args.num_blocks, args.latent_dim, region_mask=region_mask, resmlp=(args.pred_weight!=0))
    print(f"Loading model config: {json.dumps(ae_config, indent=4)}")
    model = autoencoder.Autoencoder(ae_config)
    print("Model structure:")
    print(model)
    print('Total params: %.2f' % (sum(p.numel() for p in model.parameters())))

    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True
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
        logger.set_names(['Epoch', 'LR', 'train mse', args.output_type +'_pred', args.output_type +'_rec', 'train time', 'val time'])

    lr_scheduler = {'coslr': tools.cosine_lr,
                    'constant': tools.constant}

    return model, criterion, optimizer, lr_scheduler, logger


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

def train_batch(model, criterion, optimizer, batch, args):
    points_x, points_y = batch[:2]
    #points_x, points_y = (points_x.float()).cuda(), (points_y.float()).cuda()
    points_x = (points_x.float()).cuda()

    # compute output
    x_rec = model(points_x)
    loss = criterion(x_rec, points_x)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

def validate_batch(model, batch, criterion, args):
    points_x = batch[0]
    points_x = (points_x.float()).cuda()

    # compute output
    x_rec = model(points_x)
    loss_rec = criterion(x_rec, points_x)
    return loss_rec.item()

# write a function that returns the min and max coordinates for the box
import math
def get_min_max_coords(mask, pad):
    min_x = np.min(np.where(mask)[0])
    max_x = np.max(np.where(mask)[0])
    min_y = np.min(np.where(mask)[1])
    max_y = np.max(np.where(mask)[1])
    wid_x = math.ceil((max_x - min_x)/2)*2
    wid_y = math.floor((max_y - min_y)/2)*2
    return int(min_x-pad), int(min_x+wid_x+pad), \
        int(min_y-pad), int(min_y+wid_y+pad)

def main(args):
    trainloader, validloader = prep_dataloaders(args)

    early_stopper = EarlyStopper(patience=10,min_delta=0)

    region_mask = None
    if args.region_mask is not None and args.region_mask != "all":
        region_mask = np.load(args.region_mask)[None, None, :, :]
    else:
        region_mask = np.ones((1, 1, 96, 144))
    # region_mask = to_inference_shape(region_mask).squeeze()
    min_x, max_x, min_y, max_y = get_min_max_coords(region_mask.squeeze(), 3)
    lon = np.linspace(0,357.5,144)
    lat = np.linspace(-90,90,96)
    print("Region window coordinates: ", lon[min_y], lon[max_y], lat[min_x], lat[max_x])

    # define model
    model, criterion, optimizer, lr_scheduler, logger = prep_models(args)
    # Train and test
    current_iters = 0
    for epoch in range(args.epoch):
        """
        training
        """
        train_losses = AverageMeter()
        train_time_begin = time.time()
        for iter, batch in enumerate(trainloader):
            if batch[0].size() == 1 and batch[0] == 0:
                continue
            lr = lr_scheduler[args.lr_strategy](optimizer, args.lr, current_iters, len(trainloader) * args.epoch)
            # apply the region window
            batch[0] = batch[0][:, :, min_x: max_x, min_y: max_y]
            print("input_shape:", batch[0].shape)

            model.train()
            train_mse = train_batch(model, criterion, optimizer, batch, args)

            train_losses.update(train_mse, batch[0].size(0))
            current_iters += 1
            print('training- epoch:{}/{} | iters:{}/{}| lr:{:.6f} | train mse:{:.6f}|'.format(epoch, args.epoch, iter+1, len(trainloader), lr, train_mse))
        train_time = time.time() - train_time_begin

        """
        testing
        """
        #### define the loss seperately ## we have 7 mse accordingly

        valid_losses = {}
        for i in range(1):
            valid_losses[i] = AverageMeter()
        loss_name = []
        loss_name.append(args.output_type + '_rec: {:.5f}')
        valid_time_begin = time.time()
        for iter, batch in enumerate(validloader):
            if batch[0].size() == 1 and batch[0] == 0:
                # skip empty batch due to missing data
                continue
            #batch[0] = batch[0].reshape(-1, batch[0].shape[-1])
            #batch[1] = batch[1].reshape(-1, batch[1].shape[-1])
            suffix = 'testing- epoch:{}| iters:{}/{} |'.format(epoch, iter+1, len(validloader))
            # apply region window
            batch[0] = batch[0][:, :, min_x: max_x, min_y: max_y]

            model.eval()
            valid_loss = validate_batch(model, batch, criterion, args)

            valid_losses[0].update(valid_loss, batch[0].size(0))
            suffix = suffix + loss_name[0].format(valid_losses[0].avg)
            print(suffix)


        test_time = time.time() - valid_time_begin

        current_datetime = datetime.now()
        print(f"Epoch {epoch} time: {current_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        #### save the log and ckpt ###################################
        save_log = [epoch, lr, train_losses.avg]
        for i in range(1):
            # save_log.append(1 - test_losses[i].avg/test_variance[args.output_type])
            save_log.append(valid_losses[i].avg)

        save_log.append(train_time)
        save_log.append(test_time)
        logger.append(save_log)
        if False and early_stopper.early_stop(test_losses[i].avg):
            tools.save_checkpoint({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, checkpoint=args.checkpoint, filename='checkpoint_epoch'+str(epoch)+'.pth.tar')
            break

        if (epoch)%5 == 0:
            tools.save_checkpoint({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, checkpoint=args.checkpoint, filename='checkpoint_epoch'+str(epoch)+'.pth.tar')


        tools.save_checkpoint({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, checkpoint=args.checkpoint)


    logger.close()

if __name__ == '__main__':
    current_datetime = datetime.now()

    print("Training start time:", current_datetime.strftime("%Y-%m-%d %H:%M:%S"))
    parser = argsparser.get_argparser()
    parser.add_argument("--ae_config", type=str, help="path to ae config file")
    parser.add_argument("--multistep", type=int, help="multistep", default=1)
    parser.add_argument("--sample_rate", type=int, help="sample_rate", default=12)
    parser.add_argument("--latent_dim", type=int, help="latent_dim", default=256)
    parser.add_argument("--ex_input", type=str, nargs="*", default=[])
    parser.add_argument('--region_mask', type=str, help='path to region mask npy file', default="all")
    # parser.add_argument('--rec_weight', type=float, default=0.1)
    # parser.add_argument('--pred_weight', type=float, default=0.9)

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
    current_datetime = datetime.now()
    print("Training end time:", current_datetime.strftime("%Y-%m-%d %H:%M:%S"))

