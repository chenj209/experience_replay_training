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
import numpy as np
from torch.utils import data
# from torch.utils.data.dataloader import default_collate

sys.path.append(os.path.join(sys.path[0], "..", "models"))
import autoencoder
sys.path.append(os.path.join(sys.path[0], ".."))
from utils import Logger, AverageMeter, mkdir_p
sys.path.append(os.path.join(sys.path[0], "..", "utils"))
import argsparser
import tools
from data_shape import to_inference_shape
sys.path.append(os.path.join(sys.path[0], "..", "dataloader"))
from dataloader_newformat import DatasetDisk

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
    col_names = np.loadtxt(data_dir + "/col_names.txt", dtype=str)
    col_names_x = ["QL", "T_nn_in", "dqvls_nn_in", "dTls_nn_in", "SOLIN", "SPPS"]
    prev_ex_vars = ["qtend_check", "stend_check", "SOLL", "SOLLD", "SOLS", "SOLSD", "FSDS"]
    col_names_y = ["qtend_check"]
    data_means = dict(np.load(data_dir + "/data_means.npz"))
    data_stds = dict(np.load(data_dir + "/data_stds.npz"))
    #################### 屏蔽掉一些可能存在异常的数据集 ###############################
    #all_files = glob.glob(args.data_dir+'/*')[::13]#[::7]
    all_files = glob.glob(args.data_dir+'/*.npy')
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

    #test_idx = np.random.choice(len(all_files),len(all_files)//20,replace=False)
    #test_idx = np.random.choice(len(all_files),len(all_files)//10,replace=False)
    #test_idx = np.random.choice(len(all_files),len(all_files)//10,replace=False)
    test_idx = np.arange(len(all_files))[-(len(all_files)//10):]
    train_files = [all_files[i] for i in range(len(all_files)) if i not in test_idx]
    #random.shuffle(train_files)
    test_files = [all_files[i] for i in test_idx]
    print('train files: {} test files: {}'.format(len(train_files), len(test_files)))
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
        prev_ex_vars=prev_ex_vars+args.ex_input)
    trainloader = data.DataLoader(training_set, shuffle=True, batch_size=args.train_batch, num_workers=args.workers)

    testing_set = DatasetDisk(
        test_files,
        col_names,
        col_names_x,
        col_names_y,
        data_stds,
        data_means,
        is_train=False,
        noise_std=0,
        multistep=int(args.multistep),
        sample_rate=args.sample_rate,
        prev_ex_vars=prev_ex_vars+args.ex_input)
    testloader = data.DataLoader(testing_set, shuffle=False, batch_size=args.train_batch, num_workers=args.workers)
    early_stopper = EarlyStopper(patience=10,min_delta=0)

    region_mask = None
    if args.region_mask is not None:
        region_mask = np.load(args.region_mask)[None, None, :, :]
        region_mask = to_inference_shape(region_mask)

    # define model
    input_size = len(training_set.input_indices)\
                +int(args.multistep)*(len(training_set.prev_input_indices))
    print(f"Model input size: {input_size}")
    model = autoencoder.AutoencoderResMLP(input_size, 30, args.node_size, args.activation, args.num_blocks, args.latent_size, region_mask=region_mask)

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
            if batch[0].size() == 1 and batch[0] == 0:
                # skip empty batch due to missing data
                continue
            batch[0] = batch[0].reshape(-1, batch[0].shape[-1])
            batch[1] = batch[1].reshape(-1, batch[1].shape[-1])
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
            # train_mse = tools.train(batch, model, criterion, optimizer)
            model.train()

            points_x, points_y = batch[:2]
            points_x, points_y = (points_x.float()).cuda(), (points_y.float()).cuda()
            
            
        #     print('!!!!!!!!!!!!!!!!!batch',points_x.size())  #1024 122???

            # compute output
            outputs_y, x_rec = model(points_x)
            # print(outputs_y.size(), points_y.size())
            loss_pred = criterion(outputs_y, points_y)
            loss_rec = criterion(x_rec, points_x)
            loss = 2/3*loss_pred + 1/3*loss_rec
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
        for i in range(2):
            test_losses[i] = AverageMeter()
        loss_name = [args.output_type + '_pred: {:.5f}']
        loss_name.append(args.output_type + '_rec: {:.5f}')
        test_time_begin = time.time()
        for iter, batch in enumerate(testloader):
            if batch[0].size() == 1 and batch[0] == 0:
                # skip empty batch due to missing data
                continue
            batch[0] = batch[0].reshape(-1, batch[0].shape[-1])
            batch[1] = batch[1].reshape(-1, batch[1].shape[-1])
            suffix = 'testing- epoch:{}| iters:{}/{} |'.format(epoch, iter+1, len(testloader))
            if args.output_type == '0-29':
                batch[1] = batch[1][:, :30]
            if args.output_type == '30-59':
                batch[1] = batch[1][:, 30:60]
            if args.output_type == '60':
                batch[1] = batch[1][:, 60:61]
            if args.output_type == '61-65':
                batch[1] = batch[1][:, 61:66]
            # test_mses = tools.test_de(batch, model, criterion)
            model.eval()

            points_x, points_y = batch[:2]
            points_x, points_y = (points_x.float()).cuda(), (points_y.float()).cuda()
            
            
        #     print('!!!!!!!!!!!!!!!!!batch',points_x.size())  #1024 122???

            # compute output
            outputs_y, x_rec = model(points_x)
            # print(outputs_y.size(), points_y.size())
            loss_pred = criterion(outputs_y, points_y)
            loss_rec = criterion(x_rec, points_x)
            test_losses[0].update(loss_pred.item()*2/3, batch[0].size(0))
            test_losses[1].update(loss_rec.item()*1/3, batch[0].size(0))
                # suffix = suffix + loss_name[i].format(1 - test_losses[i].avg/test_variance[args.output_type])
            suffix = suffix + loss_name[0].format(test_losses[0].avg)
            suffix = suffix + loss_name[1].format(test_losses[1].avg)
            print(suffix)


        test_time = time.time() - test_time_begin
        #### save the log and ckpt ###################################
        save_log = [epoch, lr, train_losses.avg]
        for i in range(2):
            # save_log.append(1 - test_losses[i].avg/test_variance[args.output_type])
            save_log.append(test_losses[i].avg)

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
    parser = argsparser.get_argparser()
    parser.add_argument("--multistep", type=int, help="multistep", default=1)
    parser.add_argument("--sample_rate", type=int, help="sample_rate", default=12)
    parser.add_argument("--latent_size", type=int, help="latent_size", default=4)
    parser.add_argument("--ex_input", type=str, nargs="*", default=[])
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
