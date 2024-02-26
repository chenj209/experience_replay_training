import os
import json
import time
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
import torchvision.transforms as transforms
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
from dataloader_newformat import DatasetDisk, filter_collate
from preprocess import StandardizeTransform, \
    RectRegionMaskTransform, get_min_max_coords
from dataloader_utils import gen_multistep_col_indices, get_index_from_colnames

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
    data_dir = args.data_dir
    if not os.path.isdir(data_dir):
        data_dir = "/data/nncam_data/image_set/"
    if not os.path.isdir(data_dir):
        # data_dir = "./data/"
        data_dir = "../analysis/test_data/"
    if not os.path.isdir(data_dir):
        # data_dir = "./data/"
        data_dir = "/pscratch/sd/c/chenjd21/spcam_new_data/"
    #################### 屏蔽掉一些可能存在异常的数据集 ###############################
    #all_files = glob.glob(args.data_dir+'/*')[::13]#[::7]
    all_files = glob.glob(args.data_dir+'/*.npy')
    all_files.sort()
    all_files = all_files[:35040]

    test_idx = np.arange(len(all_files))[-(len(all_files)//10):]
    train_files = [all_files[i] for i in range(len(all_files)) if i not in test_idx]
    #random.shuffle(train_files)
    test_files = [all_files[i] for i in test_idx]
    print('train files: {} test files: {}'.format(len(train_files), len(test_files)))
    # training_set = DatasetDisk(file_names=train_files, is_train=True, noise_std=args.noise_std, multistep=int(args.multistep))

    col_names = np.loadtxt(data_dir + "/col_names.txt", dtype=str)
    col_names_x = ["QL", "T_nn_in", "dqvls_nn_in", "dTls_nn_in", "SOLIN", "SPPS"]+args.ex_input
    prev_ex_vars = ["qtend_check", "stend_check", "SOLL", "SOLLD", "SOLS", "SOLSD", "FSDS"]+args.ex_input_prev
    col_names_y = ["qtend_check"]
    data_means = dict(np.load(data_dir + "/data_means.npz"))
    data_stds = dict(np.load(data_dir + "/data_stds.npz"))

    input_indices, prev_input_indices, output_indices = gen_multistep_col_indices(
        col_names, prev_ex_vars, col_names_x, col_names_y, int(args.multistep))
    print("input_indices:", col_names[input_indices])
    print("prev_input_indices:", col_names[prev_input_indices])
    print("output_indices:", col_names[output_indices])
    with open(args.ae_config, "r") as f:
        ae_config = json.load(f)
    ae_config["input_size"][0] = len(prev_input_indices)*int(args.multistep)+len(input_indices)

    region_mask = None
    if args.region_mask is not None and args.region_mask != "all":
        region_mask = np.load(args.region_mask)
    else:
        region_mask = np.ones((96, 144))
    min_x, max_x, min_y, max_y = get_min_max_coords(region_mask, 2)
    sub_region_mask = region_mask[min_x:max_x, min_y:max_y]

    multistep_col_names_x = []
    for i in range(int(args.multistep)):
        multistep_col_names_x.extend(col_names_x)
        multistep_col_names_x.extend(prev_ex_vars)
    multistep_col_names_x.extend(col_names_x)

    transform = transforms.Compose([
        RectRegionMaskTransform(region_mask),
        StandardizeTransform(
            data_means,
            data_stds,
            multistep_col_names_x,
            col_names_y,
            col_names,
            normalize_input=True,
            normalize_output=True
            ),
    ])

    training_set = DatasetDisk(
        train_files,
        input_indices,
        prev_input_indices,
        output_indices,
        multistep=int(args.multistep),
        sample_rate=args.sample_rate,
        is_train=True,
        transform=transform,
        include_filename=True)

    trainloader = data.DataLoader(
        training_set,
        shuffle=True,
        batch_size=args.train_batch,
        num_workers=args.workers,
        collate_fn=filter_collate)

    testing_set = DatasetDisk(
        test_files,
        input_indices,
        prev_input_indices,
        output_indices,
        multistep=int(args.multistep),
        sample_rate=args.sample_rate,
        is_train=False,
        transform=transform,
        include_filename=True)
    testloader = data.DataLoader(
        testing_set,
        shuffle=False,
        batch_size=args.train_batch,
        num_workers=args.workers,
        collate_fn=filter_collate)
    #early_stopper = EarlyStopper(patience=10,min_delta=0)

    # define model
    print("Model config:", json.dumps(ae_config, indent=4))
    model = autoencoder.AutoencoderResMLP(
        config=ae_config,
        input_size=len(training_set.input_indices),
        output_size=len(training_set.output_indices),
        m=512,
        activation='relu',
        num_blocks=7,
        sub_region_mask=sub_region_mask
    )

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

    # test_variance = {'0-29': 0.41921, '30-59': 0.96519, '60': 0.96958, '61-65':0.54228}
    # def my_collate(batch):
    #     batch = list(filter (lambda x:x is not None, batch))
    #     return default_collate(batch)

    # Train and test
    current_iters = 0
    for epoch in range(args.epoch):
        """
        training
        """
        train_losses = AverageMeter()
        train_time_begin = time.time()
        for iter, batch in enumerate(trainloader):
            if batch is None:
                # skip empty batch due to missing data
                continue
            lr = lr_scheduler[args.lr_strategy](optimizer, args.lr, current_iters, len(trainloader) * args.epoch)
            if args.output_type == '0-29':
                batch[1] = batch[1][:, :, :30]
            if args.output_type == '30-59':
                batch[1] = batch[1][:, :, 30:60]
            if args.output_type == '60':
                batch[1] = batch[1][:, :, 60:61]
            if args.output_type == '61-65':
                batch[1] = batch[1][:, :, 61:66]
#             if args.output_type == '61-65':
#                 train_mse = tools.train_penalty(batch, model, criterion, optimizer)
#             else:
            # train_mse = tools.train(batch, model, criterion, optimizer)
            model.train()

            points_x, points_y = batch[:2]
            # points_y: shape (batch, features, lat, lon)
            points_y = points_y[:, :, model.module.sub_region_mask]
            # points_y: shape (batch, features, n_samples)
            points_y = points_y.permute(0, 2, 1).reshape(-1, points_y.shape[1])
            # points_y: shape (batch*n_sample, features)
            points_x, points_y = (points_x.float()).cuda(), (points_y.float()).cuda()


        #     print('!!!!!!!!!!!!!!!!!batch',points_x.size())  #1024 122???

            # compute output
            outputs_y, x_rec = model(points_x)
            # print("eval: ", outputs_y.size(), points_y.size())
            loss_pred = 0
            if outputs_y is not None:
                loss_pred = criterion(outputs_y, points_y)
            loss_rec = criterion(x_rec, points_x)
            loss = args.pred_weight*loss_pred + args.rec_weight*loss_rec
            # print(points_y)
            # print(torch.min(points_y))
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_mse = loss.item()
            train_losses.update(train_mse, batch[0].size(0))
            current_iters += 1
            print('training- epoch:{}/{} | iters:{}/{}| lr:{:.6f} | \
                  train pred mse:{:.6f}| train rec mse: {:.6f} |'.format(
                      epoch, args.epoch, iter+1, len(trainloader),
                      lr, loss_pred.item(), loss_rec.item()))
        train_time = time.time() - train_time_begin

        """
        testing
        """
        #### define the loss seperately ## we have 7 mse accordingly

        test_losses = {}
        for i in range(2):
            test_losses[i] = AverageMeter()
        loss_name = [args.output_type + '_pred: {:.5f}']
        loss_name.append(args.output_type + '_rec: {:.5f}')
        test_time_begin = time.time()
        for iter, batch in enumerate(testloader):
            if batch is None:
                # skip empty batch due to missing data
                continue
            #batch[0] = batch[0].reshape(-1, batch[0].shape[-1])
            #batch[1] = batch[1].reshape(-1, batch[1].shape[-1])
            suffix = 'testing- epoch:{}| iters:{}/{} |'.format(epoch, iter+1, len(testloader))
            if args.output_type == '0-29':
                batch[1] = batch[1][:,:,  :30]
            if args.output_type == '30-59':
                batch[1] = batch[1][:, :, 30:60]
            if args.output_type == '60':
                batch[1] = batch[1][:,:,  60:61]
            if args.output_type == '61-65':
                batch[1] = batch[1][:,:,  61:66]
            # test_mses = tools.test_de(batch, model, criterion)
            model.eval()

            points_x, points_y = batch[:2]
            points_y = points_y[:, :, model.module.sub_region_mask]
            points_y = points_y.permute(0, 2, 1).reshape(-1, points_y.shape[1])
            points_x, points_y = (points_x.float()).cuda(), (points_y.float()).cuda()


        #     print('!!!!!!!!!!!!!!!!!batch',points_x.size())  #1024 122???

            # compute output
            outputs_y, x_rec = model(points_x)
            #print("eval: ", outputs_y.size(), points_y.size())
            loss_pred = 0
            if outputs_y is not None:
                loss_pred = criterion(outputs_y, points_y).item()
            loss_rec = criterion(x_rec, points_x).item()
            test_losses[0].update(loss_pred, batch[0].size(0))
            test_losses[1].update(loss_rec, batch[0].size(0))
                # suffix = suffix + loss_name[i].format(1 - test_losses[i].avg/test_variance[args.output_type])
            suffix = suffix + loss_name[0].format(test_losses[0].avg)
            suffix = suffix + loss_name[1].format(test_losses[1].avg)
            print(suffix)


        test_time = time.time() - test_time_begin

        current_datetime = datetime.now()
        print(f"Epoch {epoch} time: {current_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        #### save the log and ckpt ###################################
        save_log = [epoch, lr, train_losses.avg]
        for i in range(2):
            # save_log.append(1 - test_losses[i].avg/test_variance[args.output_type])
            save_log.append(test_losses[i].avg)

        save_log.append(train_time)
        save_log.append(test_time)
        logger.append(save_log)
        # if False and early_stopper.early_stop(test_losses[i].avg):
        #     tools.save_checkpoint({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, checkpoint=args.checkpoint, filename='checkpoint_epoch'+str(epoch)+'.pth.tar')
        #     break

        if (epoch)%5 == 0:
            tools.save_checkpoint({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, checkpoint=args.checkpoint, filename='checkpoint_epoch'+str(epoch)+'.pth.tar')


        tools.save_checkpoint({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, checkpoint=args.checkpoint)


    logger.close()

if __name__ == '__main__':
    current_datetime = datetime.now()

    print("Training start time:", current_datetime.strftime("%Y-%m-%d %H:%M:%S"))
    parser = argsparser.get_argparser()
    parser.add_argument("--multistep", type=int, help="multistep", default=1)
    parser.add_argument("--sample_rate", type=int, help="sample_rate", default=12)
    parser.add_argument("--ex_input", type=str, nargs="*", default=[])
    parser.add_argument("--ex_input_prev", type=str, nargs="*", default=[])
    parser.add_argument('--region_mask', type=str, help='path to region mask npy file', default="all")
    parser.add_argument("--ae_config", type=str, help="path to ae config file", default=None)
    parser.add_argument('--rec_weight', type=float, default=0.1)
    parser.add_argument('--pred_weight', type=float, default=0.9)

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
