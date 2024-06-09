import os
import time
import glob
import random
import sys
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
from collections import OrderedDict
from torch.utils import data
# from torch.utils.data.dataloader import default_collate

sys.path.append(os.path.join(sys.path[0], "..", "models"))
import models
sys.path.append(os.path.join(sys.path[0], ".."))
from utils import Logger, AverageMeter, mkdir_p
sys.path.append(os.path.join(sys.path[0], "..", "utils"))
import argsparser
import tools
from data_shape import to_inference_shape, inverse_to_inference_shape
sys.path.append(os.path.join(sys.path[0], "..", "dataloader"))
#from dataloader_newformat import DatasetDisk, filter_collate
from dataloader_legacy_format import DatasetDisk, filter_collate
from preprocess import FlattenSpatialTransform, StandardizeTransform, RegionMaskTransform
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
        data_dir = "../dataloader/data/"
        # data_dir = "/pscratch/sd/c/chenjd21/spcam_new_data/"
    if not os.path.isdir(data_dir):
        data_dir = "/data/nncam_data/image_set/"
    #################### 屏蔽掉一些可能存在异常的数据集 ###############################
    #all_files = glob.glob(args.data_dir+'/*')[::13]#[::7]
    all_files = glob.glob(args.data_dir+'/*.npz')
    all_files.sort()
    all_files = all_files[:35040]

    test_idx = np.arange(len(all_files))[-(len(all_files)//10):]
    train_files = [all_files[i] for i in range(len(all_files)) if i not in test_idx]
    #random.shuffle(train_files)
    test_files = [all_files[i] for i in test_idx]
    print('train files: {} test files: {}'.format(len(train_files), len(test_files)))
    # training_set = DatasetDisk(file_names=train_files, is_train=True, noise_std=args.noise_std, multistep=int(args.multistep))
    #col_names = np.loadtxt(data_dir + "/col_names.txt", dtype=str)
    col_names = np.loadtxt(os.path.join(os.path.dirname(__file__), "..", "dataloader", "col_names.txt"), dtype=str)
    #col_names_x = ["QL", "T_nn_in", "dqvls_nn_in", "dTls_nn_in", "SOLIN", "SPPS"]+args.ex_input
    col_names_x = ["QL", "T_nn_in", "dqvls_nn_in", "dTls_nn_in", "SOLIN", "SPPS"]
    #prev_ex_vars = ["qtend_check", "stend_check", "SOLL", "SOLLD", "SOLS", "SOLSD", "FSDS"]+args.ex_input_prev
    #prev_ex_vars = ["qtend_check", "stend_check", "SOLL", "SOLLD", "SOLS", "SOLSD", "FSDS"]
    prev_ex_vars = ["qtend_check", "stend_check", "SOLL", "SOLS", "SOLSD", "SOLLD", "FSDS"]
    if args.output_type == '0-29':
        #batch[1] = batch[1][:, :, :30]
        col_names_y = ["qtend_check"]
    elif args.output_type == '30-59':
        # batch[1] = batch[1][:, :, 30:60]
        col_names_y = ["stend_check"]
    elif args.output_type == '60':
        #batch[1] = batch[1][:, :, 60:61]
        raise NotImplementedError
    elif args.output_type == '61-65':
        #batch[1] = batch[1][:, :, 61:66]
        col_names_y = ["SOLL","SOLLD","SOLS","SOLSD","FSDS"]
#             if args.output_type == '61-65':
    else:
        col_names_y = [args.output_type]
    #col_names_y = ["qtend_check"]
    data_means = dict(np.load(args.data_means))
    data_stds = dict(np.load(args.data_stds))

    input_indices = [("X", np.arange(122))]
    output_indices = [("Y", np.concatenate([np.arange(60), np.arange(61,66)]))] # index 60 is not used
    prev_input_indices = [("X", np.arange(122)), ("Y", np.concatenate([np.arange(60), np.arange(61,66)]))]
    # input_indices, prev_input_indices, output_indices = gen_multistep_col_indices(
        # col_names, prev_ex_vars, col_names_x, col_names_y, int(args.multistep))
    # print("input_indices:", col_names[input_indices])
    # print("prev_input_indices:", col_names[prev_input_indices])
    # print("output_indices:", col_names[output_indices])
    # if args.region_mask == "all":
        # region_mask = np.ones((96,144))
    # else:
        # region_mask = np.load(args.region_mask)

    multistep_col_names_x = []
    for i in range(int(args.multistep)):
        multistep_col_names_x.extend(col_names_x)
        multistep_col_names_x.extend(prev_ex_vars)
    multistep_col_names_x.extend(col_names_x)
    transform = transforms.Compose([
        #RegionMaskTransform(region_mask),
        StandardizeTransform(
            data_means,
            data_stds,
            multistep_col_names_x,
            col_names_y,
            col_names,
            normalize_input=True,
            normalize_output=True
            ),
        FlattenSpatialTransform()
        ])


    training_set = DatasetDisk(
        train_files,
        input_indices,
        prev_input_indices,
        output_indices,
        multistep=int(args.multistep),
        sample_rate=int(args.sample_rate),
        is_train=True,
        transform=transform,
        include_filename=True
    )
        # region_mask1d=None if args.region_mask=="all"
                        #    else np.load(args.region_mask))

    trainloader = data.DataLoader(training_set, shuffle=True,
                                  batch_size=args.train_batch,
                                  num_workers=args.workers,
                                  collate_fn=filter_collate,
                                  pin_memory=True)

    testing_set = DatasetDisk(
        test_files,
        input_indices,
        prev_input_indices,
        output_indices,
        multistep=int(args.multistep),
        sample_rate=int(args.sample_rate),
        is_train=False,
        transform=transform,
        include_filename=True,
        region_mask1d=None if args.region_mask=="all"
                           else np.load(args.region_mask))

    testloader = data.DataLoader(testing_set, shuffle=False,
                                 batch_size=args.train_batch,
                                 num_workers=args.workers,
                                 collate_fn=filter_collate,
                                 pin_memory=True)

    #early_stopper = EarlyStopper(patience=10,min_delta=0)
    model_load_start = time.time()

    # define model
    if args.network == 'resnet':
        model = models.ResNet(args.node_size, args.activation)
    elif args.network == 'resnet_output30':
        #model = models.ResNet_output30_Time(args.node_size, args.activation, args.num_blocks)
        #model = models.ResMLP(309, 30, args.node_size, args.activation, args.num_blocks)
        input_size = len(training_set.input_indices)\
                    +int(args.multistep)*(len(training_set.prev_input_indices))
        print(f"Model input size: {input_size}")
        model = models.ResMLP(input_size, 30, args.node_size, args.activation, args.num_blocks)
    elif args.network == 'resnet_output5':
        #model = models.ResNet_output5_Time(args.node_size, args.activation, args.num_blocks)
        input_size = len(training_set.input_indices)\
                    +int(args.multistep)*(len(training_set.prev_input_indices))
        print(f"Model input size: {input_size}")
        model = models.ResMLP(input_size, 5, args.node_size, args.activation, args.num_blocks)
    elif args.network == 'resnet_output1':
        #model = models.ResNet_output1(args.node_size, args.activation, args.num_blocks)
        input_size = len(training_set.input_indices)\
                    +int(args.multistep)*(len(training_set.prev_input_indices))
        print(f"Model input size: {input_size}")
        model = models.ResMLP(input_size, 1, args.node_size, args.activation, args.num_blocks)
    elif args.network == 'mlp_output30':
        model = models.mlp_output30(args.node_size, args.activation, args.num_blocks)
    elif args.network == 'mlp_output5':
        model = models.mlp_output5(args.node_size, args.activation, args.num_blocks)
    elif args.network == 'mlp_output1':
        model = models.mlp_output1(args.node_size, args.activation, args.num_blocks)
    else:
        model = None

    print('Total params: %.2f' % (sum(p.numel() for p in model.parameters())))
    model = model.float()
    #model = torch.nn.DataParallel(model).cuda()
    #model = model.cuda()
    model = torch.nn.DataParallel(model, device_ids=[0,1]).cuda(0)
    #print("Devices used by DataParallel:", model.device_ids)
    # Check the location of model parameters
    #for name, param in model.named_parameters():
    #    print(f"Parameter '{name}' is on device: {param.device}")

    # Check the location of model buffers (if any)
    #for name, buffer in model.named_buffers():
    #        print(f"Buffer '{name}' is on device: {buffer.device}")
    #model = model.cuda(2)
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
        try:
            model.load_state_dict(checkpoint['state_dict'])
        except Exception as e:
            print("Model loading error:", e)
            print("Retrying using by adding module prefix")
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                #name = k[7:] if k.startswith('module.') else k  # remove `module.` prefix
                name = "module."+k  # adding `module.` prefix
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Epoch', 'LR', 'train mse', args.output_type +'_r2',  'train time', 'val time'])

    lr_scheduler = {'coslr': tools.cosine_lr,
                    'constant': tools.constant}
    print("Model load time:", time.time() - model_load_start)

    # test_variance = {'0-29': 0.41921, '30-59': 0.96519, '60': 0.96958, '61-65':0.54228}
    # def my_collate(batch):
    #     batch = list(filter (lambda x:x is not None, batch))
    #     return default_collate(batch)
    total_train_time = time.time()

    # Train and test
    best_loss = 999
    current_iters = 0
    for epoch in range(args.epoch):
        print("here_start", epoch, args.epoch)
        """
        training
        """
        train_losses = AverageMeter()
        train_time_begin = time.time()
        for iter, batch in enumerate(trainloader):
            if batch is None:
                print("skip empty batch")
                # skip empty batch due to missing data
                continue
            lr = lr_scheduler[args.lr_strategy](optimizer, args.lr,
                                                current_iters, len(trainloader) * args.epoch)
#                 train_mse = tools.train_penalty(batch, model, criterion, optimizer)
#             else:
            bp_time = time.time()
            train_mse = tools.train(batch, model, criterion, optimizer)
            bp_time = time.time() - bp_time
            train_losses.update(train_mse, batch[0].size(0))
            current_iters += 1
            print('training- epoch:{}/{} | iters:{}/{}| lr:{:.6f} | train mse:{:.6f}| bp time: {:.2f} '.format(epoch, args.epoch, iter+1, len(trainloader), lr, train_mse, bp_time))
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
            if batch is None:
                # skip empty batch due to missing data
                continue
            suffix = 'testing- epoch:{}| iters:{}/{} |'.format(epoch, iter+1, len(testloader))
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
        curr_test_loss = test_losses[i].avg
        for i in range(1):
            # save_log.append(1 - test_losses[i].avg/test_variance[args.output_type])
            save_log.append(test_losses[i].avg)

        save_log.append(train_time)
        save_log.append(test_time)
        logger.append(save_log)
        # if False and early_stopper.early_stop(test_losses[i].avg):
            # tools.save_checkpoint({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, checkpoint=args.checkpoint, filename='checkpoint_epoch'+str(epoch)+'.pth.tar')
            # break

        if (epoch)%5 == 0:
            tools.save_checkpoint({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, checkpoint=args.checkpoint, filename='checkpoint_epoch'+str(epoch)+'.pth.tar')
        if curr_test_loss < best_loss:
            best_loss = curr_test_loss
            tools.save_checkpoint({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, checkpoint=args.checkpoint, filename='checkpoint_best_loss.pth.tar')

        tools.save_checkpoint({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, checkpoint=args.checkpoint)



        
        print("here", epoch, args.epoch)


    logger.close()

if __name__ == '__main__':
    parser = argsparser.get_argparser()
    parser.add_argument("--multistep", type=int, help="multistep", default=1)
    parser.add_argument("--sample_rate", type=int, help="sample_rate", default=12)
    parser.add_argument("--ex_input", type=str, nargs="*")
    parser.add_argument("--ex_input_prev", type=str, nargs="*")
    parser.add_argument("--region_mask", type=str, help="path to region mask npy file", default="all")
    parser.add_argument("--data_means", type=str)
    parser.add_argument("--data_stds", type=str)
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
