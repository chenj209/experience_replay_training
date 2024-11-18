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
from sklearn.metrics import r2_score
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
from dataloader_replay_buffer import DatasetDisk, filter_collate
from dataloader_stride import DatasetDisk as TestDatasetDisk
from replay_buffer import ReplayBuffer
from preprocess import FlattenSpatialTransformNext, StandardizeTransformNext, RegionMaskTransform
from preprocess import FlattenSpatialTransform, StandardizeTransform 
from dataloader_utils import gen_multistep_col_indices, get_index_from_colnames, \
    gen_col_indices

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

def cal_input_size(col_names):
    print(col_names)
    s = 0
    for i in range(len(col_names)):
        s += len(col_names[i][1])
    return s

MODEL_TYPES = ['0_29', '30_59', '61_65']

COL_NAMES_LEGACY = {
    "X": [
        *[f"QL_lev{i}" for i in range(30)],
        *[f"T_nn_in_lev{i}" for i in range(30)],
        *[f"dqvls_nn_in_lev{i}" for i in range(30)],
        *[f"dTls_nn_in_lev{i}" for i in range(30)],
        "SOLIN",
        "SPPS"
    ],
    "Y": [
        *[f"qtend_check_lev{i}" for i in range(30)],
        *[f"stend_check_lev{i}" for i in range(30)],
        "UNKNOWN",
        "SOLL", "SOLS", "SOLSD", "SOLLD", "FSDS"
    ],
    "EX": [*[f"UL_lev{i}" for i in range(30)],
           *[f"VL_lev{i}" for i in range(30)],
           *[f"CLOUD_lev{i}" for i in range(30)], 
           "CAPE", "FLNS", "FLNT", "SPPRECC", "LWUP"]
}

VAR_DIMS = {
    "QL": 30,
    "TL": 30,
    "dqvls_nn_in": 30,
    "dTls_nn_in": 30,
    "SOLIN": 1,
    "SPPS": 1,
    "qtend_check": 30,
    "stend_check": 30,
    "SOLL": 1,
    "SOLS": 1,
    "SOLSD": 1,
    "SOLLD": 1,
    "FSDS": 1
}

def main(args):
    data_dir = args.data_dir
    if not os.path.isdir(data_dir):
        data_dir = "../dataloader/data/"
        #data_dir = "/pscratch/sd/c/chenjd21/spcam_new_data/"
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
    #col_names_x = ["QL", "T_nn_in", "dqvls_nn_in", "dTls_nn_in", "SOLIN", "SPPS"]
    col_names_x = args.input_vars
    col_names_y = args.output_vars
    col_names_prev = args.input_vars_prev
    input_indices = [
        # input indices can only come from X and EX
        # Y contains output variables which is not available at current step
        ("X", gen_col_indices(COL_NAMES_LEGACY["X"], col_names_x)),
        ("EX", gen_col_indices(COL_NAMES_LEGACY["EX"], col_names_x))
    ]
    
    prev_input_indices = [
        #  prev_input indices can come from X, Y and EX
        ("X", gen_col_indices(COL_NAMES_LEGACY["X"], col_names_x)),
        ("EX", gen_col_indices(COL_NAMES_LEGACY["EX"], col_names_x + col_names_prev)),
        ("Y", gen_col_indices(COL_NAMES_LEGACY["Y"], col_names_prev)),
    ]
    output_indices = [
        # output indices can only come from Y and EX
        ("Y", gen_col_indices(COL_NAMES_LEGACY["Y"], col_names_y)),
        ("EX", gen_col_indices(COL_NAMES_LEGACY["EX"], col_names_y)),
    ]
    data_means = dict(np.load(args.data_means))
    data_stds = dict(np.load(args.data_stds))
        
    multistep_col_names_x = []
    for i in range(int(args.multistep)):
        multistep_col_names_x.extend(col_names_x)
        multistep_col_names_x.extend(col_names_prev)
    multistep_col_names_x.extend(col_names_x)

    # compute the actual indices for variables in each batch input
    batch_indices = {}
    cur_idx = 0
    for var_name in multistep_col_names_x:
        batch_indices[var_name] = (cur_idx, cur_idx+VAR_DIMS[var_name])
        cur_idx = cur_idx + VAR_DIMS[var_name]
    print("batch indices:", batch_indices)

    traintransform = transforms.Compose([
        #RegionMaskTransform(region_mask),
        StandardizeTransformNext(
            data_means,
            data_stds,
            multistep_col_names_x,
            col_names_y,
            col_names,
            normalize_input=True,
            normalize_output=True
            ),
        FlattenSpatialTransformNext()
        ])
    testtransform = transforms.Compose([
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
        sample_stride=int(args.sample_stride),
        is_train=True,
        transform=traintransform,
        include_filename=True,
        include_idx=True,
        ex_dir=args.ex_data_dir
    )
        # region_mask1d=None if args.region_mask=="all"
                        #    else np.load(args.region_mask))

    trainloader = data.DataLoader(training_set, shuffle=True,
                                  batch_size=args.train_batch,
                                  num_workers=args.workers,
                                  collate_fn=filter_collate,
                                  pin_memory=True)

    testing_set = TestDatasetDisk(
        test_files,
        input_indices,
        prev_input_indices,
        output_indices,
        multistep=int(args.multistep),
        sample_stride=int(args.sample_stride),
        is_train=False,
        transform=testtransform,
        include_filename=True,
        ex_dir=args.ex_data_dir
    )
        # region_mask1d=None if args.region_mask=="all"
        #                   else np.load(args.region_mask))

    testloader = data.DataLoader(testing_set, shuffle=False,
                                 batch_size=args.train_batch,
                                 num_workers=args.workers,
                                 collate_fn=filter_collate,
                                 pin_memory=True)

    #early_stopper = EarlyStopper(patience=10,min_delta=0)
    model_load_start = time.time()

    # define model
    all_models = {}
    input_size = cal_input_size(input_indices)\
                +int(args.multistep)*(cal_input_size(prev_input_indices)) - 60 # remove QT
    all_models['0_29'] = models.ResMLP(input_size, 30, args.node_size, args.activation, args.num_blocks)
    all_models['30_59'] = models.ResMLP(input_size, 30, args.node_size, args.activation, args.num_blocks)
    all_models['61_65'] = models.ResMLP(input_size, 5, args.node_size, args.activation, args.num_blocks)

    print('Total params: %.2f' % (sum(p.numel() for p in all_models["0_29"].parameters())))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_cuda_devices = torch.cuda.device_count()
    print(f"Number of visible CUDA devices: {num_cuda_devices}")
    for model_type in MODEL_TYPES:
        #all_models[model_type] = all_models[model_type].float()
        #model = model.float()
        all_models[model_type] = all_models[model_type].float()
        all_models[model_type] = torch.nn.DataParallel(all_models[model_type]).to(device)
        #all_models[model_type] = all_models[model_type].to(device)
    #model = model.cuda()
    #model = torch.nn.DataParallel(model, device_ids=[0,1]).cuda(0)
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
    all_optimizers = {}
    for model_type in MODEL_TYPES:
        if args.optim == 'sgd':
            all_optimizers[model_type] = optim.SGD(all_models[model_type].parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        elif args.optim == 'adam':
            all_optimizers[model_type] = optim.Adam(all_models[model_type].parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.weight_decay)
        else:
            all_optimizers[model_type] = None

    # Resume
    all_loggers = {}
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir(args.resume), 'Error: no checkpoint directory found!'
        for model_type in MODEL_TYPES:
            checkpoint = torch.load(args.resume+f"/{model_type}_checkpoint.pth.tar")
            try:
                all_models[model_type].load_state_dict(checkpoint['state_dict'])
            except Exception as e:
                print("Model loading error:", e)
                print("Retrying using by adding module prefix")
                new_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    #name = k[7:] if k.startswith('module.') else k  # remove `module.` prefix
                    name = "module."+k  # adding `module.` prefix
                    new_state_dict[name] = v
                all_models[model_type].load_state_dict(new_state_dict)
            all_optimizers[model_type].load_state_dict(checkpoint['optimizer'])
            all_loggers[model_type] = Logger(os.path.join(args.checkpoint, model_type+'_log.txt'), title=model_type, resume=True)
    else:
        for model_type in MODEL_TYPES:
            all_loggers[model_type] = Logger(os.path.join(args.checkpoint, model_type+'_log.txt'), title=model_type)
            all_loggers[model_type].set_names(['Epoch', 'LR', 'train mse', model_type +'_r2',  'train time', 'val time'])

    lr_scheduler = {'coslr': tools.cosine_lr,
                    'constant': tools.constant}
    print("Model load time:", time.time() - model_load_start)

    # test_variance = {'0-29': 0.41921, '30-59': 0.96519, '60': 0.96958, '61-65':0.54228}
    # def my_collate(batch):
    #     batch = list(filter (lambda x:x is not None, batch))
    #     return default_collate(batch)
    total_train_time = time.time()

    # Train and test
    best_losses = {model_type: 999 for model_type in MODEL_TYPES}
    current_iters = 0+args.start_epoch*len(trainloader)
    all_lrs = {model_type: None for model_type in MODEL_TYPES}

    replay_buffer = ReplayBuffer(training_set, inp_shape=[96*144, 309], tar_shape=[96*144,65], max_size=args.buffer_size, weighted=False, workers=1)
    for epoch in range(args.start_epoch,args.epoch):
        print(f"here_start {epoch}, {args.epoch}")
        """
        training
        """
        all_train_losses = {}
        for model_type in MODEL_TYPES:
            all_train_losses[model_type] = AverageMeter()
        train_time_begin = time.time()
        data_load_start = time.time()
        for iter, batch in enumerate(trainloader):
            if batch is None:
                print("skip empty batch")
                # skip empty batch due to missing data
                continue
            print(f"Dataload time: {time.time() - data_load_start}")
            batch_start = time.time()
            prep_data_start = time.time()
            for model_type in MODEL_TYPES:
                all_lrs[model_type] = lr_scheduler[args.lr_strategy](all_optimizers[model_type], args.lr,
                                                    current_iters, len(trainloader) * args.epoch)
#                 train_mse = tools.train_penalty(batch, model, criterion, optimizer)
#             else:
            buffer_sample_size = int(trainloader.batch_size*args.mixing_ratio)
            if replay_buffer.size >= buffer_sample_size:
                sampled_replay = replay_buffer.sample(buffer_sample_size)
                sr_input, sr_target = sampled_replay[:2]
                sr_input = torch.from_numpy(sr_input)
                sr_target = torch.from_numpy(sr_target)
                batch_input = torch.cat([batch[0], sr_input], dim=0)
                #batch_input = torch.from_numpy(np.concatenate([batch[0], sr_input], axis=0)) 
                #batch_targets = {
                #    "0_29": torch.from_numpy(np.concatenate([batch[1][:,:,:30], sr_target[:,:,:30]], axis=0)),
                #    "30_59": torch.from_numpy(np.concatenate([batch[1][:,:,30:60], sr_target[:,:,30:60]], axis=0)),
                #    "61_65": torch.from_numpy(np.concatenate([batch[1][:,:,60:65], sr_target[:,:,60:65]], axis=0))
                ##}
                batch_target = torch.cat([batch[1], sr_target], dim=0)
                batch_targets = {
                    "0_29": batch_target[:,:,:30],
                    "30_59": batch_target[:,:,30:60],
                    "61_65": batch_target[:,:,60:65]
                }
            else:
                batch_input = batch[0]
                batch_targets = {
                    "0_29": batch[1][:,:,:30],
                    "30_59": batch[1][:,:,30:60],
                    "61_65": batch[1][:,:,60:65]
                }
            batch_input = batch_input[:,:,60:] # remove prevQT
            print(f"prep data: {time.time() - prep_data_start}")
            train_mses = {}
            model_preds = {}
            bp_time = time.time()
            for model_type in MODEL_TYPES:
                # train_mses[model_type] = tools.train(
                #     batches[model_type], 
                #     all_models[model_type], 
                #     criterion, 
                #     all_optimizers[model_type]
                # )
                model = all_models[model_type]
                optimizer = all_optimizers[model_type]
                model.train()

                points_x, points_y = batch_input, batch_targets[model_type]
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                points_x, points_y = (points_x.float()).to(device), (points_y.float()).to(device)
                
                
            #     print('!!!!!!!!!!!!!!!!!batch',points_x.size())  #1024 122???

                # compute output
                model_preds[model_type] = model(points_x)
                # print(outputs_y.size(), points_y.size())
                loss = criterion(model_preds[model_type], points_y)

                # print(points_y)
                # print(torch.min(points_y))
                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_mses[model_type] = loss.item()
                all_train_losses[model_type].update(train_mses[model_type], batch[0].size(0))
            print(f"bp time: {time.time() - bp_time}")
            store_start = time.time()
            # ready to save experience
            tar_idx = batch[-2]
            #if replay_buffer.ptr >= trainloader.batch_size:
                #pred_1step = model_preds["0_29"][:batch[0].size(0)].detach().cpu().numpy()
                #pred_2step = model_preds["0_29"][batch[0].size(0):].detach().cpu().numpy()
                #print(f"1 step 029 r2: {r2_score(pred_1step.flatten(), batch_targets['0_29'].detach().cpu().numpy()[:batch[0].size(0),:,:30].flatten())}")
                #print(f"2 step 029 r2: {r2_score(pred_2step.flatten(), batch_targets['0_29'].detach().cpu().numpy()[batch[0].size(0):,:,:30].flatten())}")
            next_x, next_y = batch[2:4]
            next_x = next_x.numpy()
            next_y = next_y.numpy()
            exp = np.concatenate([
                model_preds["0_29"][:batch[0].size(0)].detach().cpu().numpy(),
                model_preds["30_59"][:batch[0].size(0)].detach().cpu().numpy(),
                model_preds["61_65"][:batch[0].size(0)].detach().cpu().numpy(),
            ],axis=2)
            # replace qtend_prev and stend_prev with exp value
            next_x[:,:,122:122+65] = exp
            # compute dqls_prev and dqls_prev 
            # dqls = Q - Q_prev - qtend_prev*24*3600
            # dTls = T - T_prev - stend_prev*24*3600
            Q_prev =     next_x[:,:,  :30]
            T_prev =     next_x[:,:,30:30+30]
            qtend_prev = next_x[:,:,122:122+30]
            stend_prev = next_x[:,:,122+30:122+30+30]
            Q =      next_x[:,:,122+65:122+65+30]
            T =      next_x[:,:,122+65+30:122+65+30+30]
            dqls =   next_x[:,:,122+65+30+30:122+65+30+30+30] 
            dTls =   next_x[:,:,122+65+30+30+30:122+65+30+30+30+30]
            print(f"store start: {time.time() - store_start}")
            replay_buffer.store(next_x, next_y, tar_idx)

            current_iters += 1
            print('training- epoch:{}/{} | iters:{}/{}| lr:{:.6f} | \
                train mse: 0_29({:.6f}) 30_59({:.6f}) 61_65({:.6f}'.format(
                    epoch, args.epoch, iter+1, len(trainloader), all_lrs["0_29"], 
                    train_mses["0_29"], train_mses["30_59"], train_mses["61_65"]))
            data_load_start = time.time()
            print(f"batch time: {time.time() - batch_start}")
        train_time = time.time() - train_time_begin

        """
        testing
        """
        #### define the loss seperately ## we have 7 mse accordingly

        test_losses = {}
        for model_type in MODEL_TYPES:
            test_losses[model_type] = AverageMeter()
        loss_names = {model_type:  model_type + '_mse: {:.5f} | ' for model_type in MODEL_TYPES}
        test_time_begin = time.time()
        for iter, batch in enumerate(testloader):
            if batch is None:
                # skip empty batch due to missing data
                continue
            suffix = 'testing- epoch:{}| iters:{}/{} |'.format(epoch, iter+1, len(testloader))
            batch[0] = batch[0][:,:,60:] # remove prevQT
            batches = {
                "0_29": [batch[0], batch[1][:,:,:30]],
                "30_59": [batch[0], batch[1][:,:,30:60]],
                "61_65": [batch[0], batch[1][:,:,60:65]]
            }
            for model_type in MODEL_TYPES:
                test_mse, _  = tools.test_de(batches[model_type], all_models[model_type], criterion)
                test_losses[model_type].update(test_mse, batch[0].size(0))
                    # suffix = suffix + loss_name[i].format(1 - test_losses[i].avg/test_variance[args.output_type])
                suffix = suffix + loss_names[model_type].format(test_losses[model_type].avg)
            print(suffix)


        test_time = time.time() - test_time_begin
        #### save the log and ckpt ###################################
        for model_type in MODEL_TYPES:
            save_log = [epoch, all_lrs[model_type], all_train_losses[model_type].avg]
            curr_test_loss = test_losses[model_type].avg
            save_log.append(test_losses[model_type].avg)

            save_log.append(train_time)
            save_log.append(test_time)
            all_loggers[model_type].append(save_log)
            # if False and early_stopper.early_stop(test_losses[i].avg):
                # tools.save_checkpoint({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, checkpoint=args.checkpoint, filename='checkpoint_epoch'+str(epoch)+'.pth.tar')
                # break

            if (epoch)%5 == 0:
                tools.save_checkpoint({'state_dict': all_models[model_type].state_dict(), 'optimizer': all_optimizers[model_type].state_dict()}, checkpoint=args.checkpoint, filename=model_type+'_checkpoint_epoch'+str(epoch)+'.pth.tar')
            if curr_test_loss < best_losses[model_type]:
                best_losses[model_type] = curr_test_loss
                tools.save_checkpoint({'state_dict': all_models[model_type].state_dict(), 'optimizer': all_optimizers[model_type].state_dict()}, checkpoint=args.checkpoint, filename=model_type+'_checkpoint_best_loss.pth.tar')

            tools.save_checkpoint({'state_dict': all_models[model_type].state_dict(), 'optimizer': all_optimizers[model_type].state_dict()}, checkpoint=args.checkpoint, filename=model_type+'_checkpoint.pth.tar')



        
        print("here", epoch, args.epoch)


    for model_type in MODEL_TYPES:
        all_loggers[model_type].close()

if __name__ == '__main__':
    parser = argsparser.get_argparser()
    parser.add_argument("--multistep", type=int, help="multistep", default=1)
    parser.add_argument("--sample_stride", type=int, help="sample_stride", default=12)
    # TODO: add input/input_prev/output order here
    # parser.add_argument("--ex_input", type=str, nargs="*")
    # parser.add_argument("--ex_input_prev", type=str, nargs="*")
    parser.add_argument("--input_vars", type=str, nargs="*", default=[
        "QL", 
        "T_nn_in", 
        "dqvls_nn_in", 
        "dTls_nn_in", 
        "SOLIN", 
        "SPPS"
    ])
    parser.add_argument("--input_vars_prev", type=str, nargs="*", default=[],
                        help="additional variables to use as inputs in the previous \
                        timesteps besides vars in input_vars and output_vars")
    parser.add_argument("--output_vars", type=str, nargs="*", default=["qtend_check"])
    parser.add_argument("--ex_data_dir", type=str, help="directory to store new \
        input variables")
    parser.add_argument("--region_mask", type=str, help="path to region mask npy file", default="all")
    parser.add_argument("--data_means", type=str)
    parser.add_argument("--data_stds", type=str)
    parser.add_argument("--buffer_size", default=288, type=int)
    parser.add_argument("--mixing_ratio", default=1.0, type=float)
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
