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
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau
from datetime import datetime
import torchvision.transforms as transforms
from pytorch_msssim import ssim, ms_ssim
# from torch.utils.data.dataloader import default_collate

sys.path.append(os.path.join(sys.path[0], "..", "models"))
import autoencoder
import variational_autoencoder
sys.path.append(os.path.join(sys.path[0], ".."))
from utils import Logger, AverageMeter, mkdir_p
sys.path.append(os.path.join(sys.path[0], "..", "utils"))
import argsparser
import tools
from data_shape import to_inference_shape, inverse_to_inference_shape
sys.path.append(os.path.join(sys.path[0], "..", "dataloader"))
from dataloader_newformat import PairDatasetDisk, filter_collate
from preprocess import StandardizeTransform, \
    RectRegionMaskTransform, get_min_max_coords
from dataloader_utils import gen_multistep_col_indices, get_index_from_colnames, \
    levelwise_variable, levelwise_variable2, delevelwise_variable
from ae_consts import *
def vae_gaussian_kl_loss(mu, logvar):
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # from blog
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return KLD.mean()
    # ours
    # kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # return kl_divergence

def reconstruction_loss(x_reconstructed, x, ltype="mse"):
    if ltype == "mse":
        mse_loss = nn.MSELoss(reduction='none')
        loss = mse_loss(x_reconstructed, x)
        return torch.sum(loss, dim=[1,2,3]).mean()
    elif ltype == "bce":
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        loss = bce_loss(x_reconstructed, x)
        return torch.sum(loss, dim=[1,2,3]).mean()
    elif ltype == "msssim":
        msssim_loss = ms_ssim(x_reconstructed, x, data_range=1)
        return 1-msssim_loss
    elif ltype == "ssim":
        #ssim_loss = ssim(x_reconstructed, x, data_range=1, win_size=7, win_sigma=1)
        ssim_loss = ssim(x_reconstructed, x, data_range=1)
        return 1-ssim_loss

def compute_vae_loss_fn(beta=1, ltype="mse"):
    def _vae_loss(mu, logvar, recon_x, x_gt):
        # mu, logvar, recon_x = model(x_gt)
        recon_loss = reconstruction_loss(recon_x, x_gt, ltype=ltype)
        kld_loss = vae_gaussian_kl_loss(mu, logvar)
        # from blog
        # return 500 * recon_loss + kld_loss
        # ours
        return recon_loss + beta*kld_loss
    return _vae_loss

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


class LinearWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, warmup_start_lr, base_lr, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        self.base_lr = base_lr
        super(LinearWarmupScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Warmup phase: linearly increase lr
            lr = (self.base_lr - self.warmup_start_lr) * self.last_epoch / self.warmup_epochs + self.warmup_start_lr
            return [lr for _ in self.base_lrs]
        else:
            # Post-warmup: keep lr constant
            return [self.base_lr for _ in self.base_lrs]

def prep_batchdata(args, batch, sub_region_mask):
    x_ae, _, x_resmlp, y_resmlp, filenames = batch
    #lr = lr_scheduler[args.lr_strategy](optimizer, args.lr, current_iters, len(trainloader) * args.epoch)
    # lr = scheduler.get_lr()[-1]
    if args.output_type == '0-29':
        y_resmlp = y_resmlp[:,:30]
    if args.output_type == '30-59':
        y_resmlp = y_resmlp[:,30:60]
    if args.output_type == '60':
        y_resmlp = y_resmlp[:,60:61]
    if args.output_type == '61-65':
        y_resmlp = y_resmlp[:,61:66]
#             if args.output_type == '61-65':
#                 train_mse = tools.train_penalty(batch, model, criterion, optimizer)
#             else:
    # train_mse = tools.train(batch, model, criterion, optimizer)

    # points_x, points_y = batch[:2]
    # points_y: shape (batch, features, lat, lon)
    y_resmlp = y_resmlp[:, :, sub_region_mask]
    #points_y = points_y[:, :, model.sub_region_mask]
    # points_y: shape (batch, features, n_samples)
    y_resmlp = y_resmlp.permute(0, 2, 1).reshape(-1, y_resmlp.shape[1])
    # points_y: shape (batch*n_sample, features)
    x_ae, y_resmlp = (x_ae.float()).cuda(), (y_resmlp.float()).cuda()
    x_resmlp = x_resmlp.float().cuda()
    return x_ae, x_resmlp, y_resmlp, filenames

def main(args):
    with open(args.ae_config, "r") as f:
        ae_config = json.load(f)
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
    # all_files = all_files[:200]

    test_idx = np.arange(len(all_files))[-(len(all_files)//10):]
    train_files = [all_files[i] for i in range(len(all_files)) if i not in test_idx]
    #random.shuffle(train_files)
    test_files = [all_files[i] for i in test_idx]
    print('train files: {} test files: {}'.format(len(train_files), len(test_files)))
    # training_set = DatasetDisk(file_names=train_files, is_train=True, noise_std=args.noise_std, multistep=int(args.multistep))

    col_names = np.loadtxt(data_dir + "/col_names.txt", dtype=str)
    #col_names_x = []+args.ex_input
    # col_names_x = [
    #     "QL_lev",
    #     "T_nn_in_lev",
    #     "dqvls_nn_in_lev",
    #     "dTls_nn_in_lev",
    #     "SOLIN",
    #     "SPPS",
    #     "LWUP",
    #     "CAPE",
    #     "UL_lev",
    #     "VL_lev"
    #     ]+args.ex_input
    col_names_x_resmlp = ae_config["x_resmlp"]
    prev_ex_vars_resmlp = ae_config["x_resmlp_prev"]
    col_names_x_ae = ae_config["x_ae"]
    prev_ex_vars_ae = ae_config["x_ae_prev"]
    # prev_ex_vars = [
    #     "qtend_check_lev",
    #     "stend_check_lev",
    #     "SOLL",
    #     "SOLLD",
    #     "SOLS",
    #     "SOLSD",
    #     "FSDS",
    #     "CLOUD_lev",
    #     "SPPRECC",
    #     "FLNS",
    #     "FLNT",
    #     "SPQRL_lev",
    #     "SPQRS_lev"
    #     ]+args.ex_input_prev
    if ae_config["reduce_lvl"]:
        col_names_x_ae = levelwise_variable2(col_names_x_ae, [12,18,23,28,29])
        prev_ex_vars_ae = levelwise_variable2(prev_ex_vars_ae, [12,18,23,28,29])
    else:
        col_names_x_ae = levelwise_variable(col_names_x_ae, START_LEV, END_LEV)
        prev_ex_vars_ae = levelwise_variable(prev_ex_vars_ae, START_LEV, END_LEV)
    col_names_x_resmlp = delevelwise_variable(col_names_x_resmlp)
    prev_ex_vars_resmlp = delevelwise_variable(prev_ex_vars_resmlp)
    #prev_ex_vars = []+args.ex_input_prev
    col_names_y = ae_config["resmlp_target"]
    # data_means = dict(np.load(data_dir + "/data_means.npz"))
    # data_stds = dict(np.load(data_dir + "/data_stds.npz"))
    data_means = dict(np.load(args.data_means))
    # data_means_by_lvl = dict(np.load(data_dir + "/std_mean_by_level_means.npz"))
    # data_means.update(data_means_by_lvl)
    data_stds = dict(np.load(args.data_stds))
    # data_stds_by_lvl = dict(np.load(data_dir + "/std_mean_by_level_stds.npz"))
    # data_stds.update(data_stds_by_lvl)

    # use the same mean and std for variables except for q related
    # use the same mean and std for variables except for q related
    for k in data_means:
        if "_lev" in k and \
            ("QL" not in k and "qtend" not in k and "dqvls" not in k):
            # debug_print(f"Replacing means {k}({data_means[k]}) with\
                        #  {k.rstrip('_lev')}({data_means[k.split('_lev')[0]]})", DEBUG)
            data_means[k] = data_means[k.split("_lev")[0]]
            # debug_print(f"Replacing stds {k}({data_stds[k]}) with\
                        #  {k.rstrip('_lev')}({data_stds[k.split('_lev')[0]]})", DEBUG)
            data_stds[k] = data_stds[k.split("_lev")[0]]

    input_indices_ae, prev_input_indices_ae, output_indices_ae = gen_multistep_col_indices(
        col_names, prev_ex_vars_ae, col_names_x_ae, [], 1)
    print("input_indices_ae:", col_names[input_indices_ae])
    print("prev_input_indices_ae:", col_names[prev_input_indices_ae])
    print("output_indices_ae:", col_names[output_indices_ae])

    input_indices_resmlp, prev_input_indices_resmlp, output_indices_resmlp = gen_multistep_col_indices(
        col_names, prev_ex_vars_resmlp, col_names_x_resmlp, col_names_y, 1)
    print("input_indices_resmlp:", col_names[input_indices_resmlp])
    print("prev_input_indices_resmlp:", col_names[prev_input_indices_resmlp])
    print("output_indices_resmlp:", col_names[output_indices_resmlp])
    # resmlp input size
    ae_config["resmlp_input_size"][0] = len(prev_input_indices_resmlp)*int(args.multistep)+len(input_indices_resmlp)
    ae_config["encoder"]["input_size"][0] = len(prev_input_indices_ae)*int(args.multistep)+len(input_indices_ae)
    ae_config["decoder"]["input_size"][0] = len(prev_input_indices_ae)*int(args.multistep)+len(input_indices_ae)

    region_mask = None
    if args.region_mask is not None and args.region_mask != "all":
        region_mask = np.load(args.region_mask)
    else:
        region_mask = np.ones((96, 144))
    min_x, max_x, min_y, max_y = get_min_max_coords(region_mask, 2)
    print("sub_region:", get_min_max_coords(region_mask, 2))
    sub_region_mask = region_mask[min_x:max_x, min_y:max_y]

    multistep_col_names_x_ae = []
    for i in range(int(args.multistep)):
        multistep_col_names_x_ae.extend(col_names_x_ae)
        multistep_col_names_x_ae.extend(prev_ex_vars_ae)
    multistep_col_names_x_ae.extend(col_names_x_ae)

    multistep_col_names_x_resmlp = []
    for i in range(int(args.multistep)):
        multistep_col_names_x_resmlp.extend(col_names_x_resmlp)
        multistep_col_names_x_resmlp.extend(prev_ex_vars_resmlp)
    multistep_col_names_x_resmlp.extend(col_names_x_resmlp)

    transform_ae = transforms.Compose([
        # RectRegionMaskTransform(region_mask),
        StandardizeTransform(
            data_means,
            data_stds,
            multistep_col_names_x_ae,
            [],
            col_names,
            normalize_input=True,
            normalize_output=True,
            threshold=5
            ),
    ])
    transform_resmlp = transforms.Compose([
        # RectRegionMaskTransform(region_mask),
        StandardizeTransform(
            data_means,
            data_stds,
            multistep_col_names_x_resmlp,
            col_names_y,
            col_names,
            normalize_input=True,
            normalize_output=True
            ),
    ])

    training_set = PairDatasetDisk(
        train_files,
        curr_input_indices1=input_indices_ae,
        curr_input_indices2=input_indices_resmlp,
        prev_input_indices1=prev_input_indices_ae,
        prev_input_indices2=prev_input_indices_resmlp,
        output_indices1=output_indices_ae,
        output_indices2=output_indices_resmlp,
        multistep=int(args.multistep),
        sample_rate=int(args.sample_rate),
        is_train=True,
        transform1=transform_ae,
        transform2=transform_resmlp,
        include_filename=True,
        region_mask2d=(region_mask,2)
        )

    trainloader = data.DataLoader(
        training_set,
        shuffle=True,
        batch_size=args.train_batch,
        num_workers=args.workers,
        collate_fn=filter_collate,
        pin_memory=True)

    testing_set = PairDatasetDisk(
        test_files,
        curr_input_indices1=input_indices_ae,
        curr_input_indices2=input_indices_resmlp,
        prev_input_indices1=prev_input_indices_ae,
        prev_input_indices2=prev_input_indices_resmlp,
        output_indices1=output_indices_ae,
        output_indices2=output_indices_resmlp,
        multistep=int(args.multistep),
        sample_rate=int(args.sample_rate),
        is_train=False,
        transform1=transform_ae,
        transform2=transform_resmlp,
        include_filename=True,
        region_mask2d=(region_mask,2)
        )

    testloader = data.DataLoader(
        testing_set,
        shuffle=False,
        batch_size=args.train_batch,
        num_workers=args.workers,
        collate_fn=filter_collate,
        pin_memory=True)
    #early_stopper = EarlyStopper(patience=10,min_delta=0)

    # define model
    print("Model config:", json.dumps(ae_config, indent=4))
    if "variational" in ae_config and ae_config["variational"]:
        model_struc = variational_autoencoder.SepInputAutoencoderResMLP
        variational_flag = True
    else:
        model_struc = autoencoder.AutoencoderResMLP
        variational_flag = False
    model = model_struc(
        encoder_config=ae_config["encoder"],
        decoder_config=ae_config["decoder"],
        #input_size=len(training_set.input_indices),
        latent_size=ae_config["latent_size"],
        input_size=ae_config["resmlp_input_size"][0],
        output_size=len(training_set.output_indices2),
        m=512,
        activation='relu',
        num_blocks=7,
        sub_region_mask=sub_region_mask
    )

    print('Total params: %.2f' % (sum(p.numel() for p in model.parameters())))
    print(model)
    for name, module in model.named_modules():
        num_params = sum(p.numel() for p in module.parameters(recurse=False))
        if num_params > 0:
            layer_size_gb = (num_params * 4) / (1024**3)  # Calculating size in GB
            print(f"{name}: {type(module).__name__}, Parameters: {num_params}, Size: {layer_size_gb:.6f} GB")
    model = model.float()
    model = torch.nn.DataParallel(model).cuda()
    #model = model.cuda()
    cudnn.benchmark = True



    """
    Define Residual Methods and Optimizer
    """
    ae_warmup_epochs = ae_config["ae_warmup_epochs"]
    criterion = nn.MSELoss()

    # freeze resmlp during warmup epochs
    for param in model.module.resmlp.parameters():
        param.requires_grad = False
    param_groups = [
        # {'params': model.module.resmlp.parameters(), 'lr': 1e-3},    # Learning rate for ResMLP
        {'params': model.module.encoder.parameters(), 'lr': ae_config["ae_lr"]},  # Learning rate for encoder
        {'params': model.module.decoder.parameters(), 'lr': ae_config["ae_lr"]},  # Learning rate for decoder
    ]
    optimizer = optim.Adam(param_groups, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.weight_decay)
    reduce_on_plateau_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50, verbose=True, min_lr=1e-6)


    # Resume
    title = ''
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.resume)
        state_dict = checkpoint['state_dict']
        filtered_state_dict = {k: v for k, v in state_dict.items() if k.startswith("module.encoder") or k.startswith("module.decoder")}
        print("Loading:", filtered_state_dict.keys())
        model.load_state_dict(filtered_state_dict, strict=False)
        #model.load_state_dict(state_dict)

        #optimizer.load_state_dict(checkpoint['optimizer'])
        #logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Epoch', 'LR', 'train mse', args.output_type +'_pred', args.output_type +'_rec', 'train time', 'val time'])
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Epoch', 'LR', 'train mse', args.output_type +'_pred', args.output_type +'_rec', 'train time', 'val time'])

    #lr_scheduler = {'coslr': tools.cosine_lr,
    #                'constant': tools.constant}

    # test_variance = {'0-29': 0.41921, '30-59': 0.96519, '60': 0.96958, '61-65':0.54228}
    # def my_collate(batch):
    #     batch = list(filter (lambda x:x is not None, batch))
    #     return default_collate(batch)

    # Train and test
    current_iters = 0
    best_valid_loss = 9999
    vae_loss_fn = compute_vae_loss_fn(beta=ae_config["beta"], ltype=ae_config["ltype"])
    for epoch in range(args.epoch):
        if epoch >= ae_warmup_epochs:
            # freeze resmlp during warmup epochs
            for param in model.module.resmlp.parameters():
                param.requires_grad = True
            param_groups = [
                {'params': model.module.resmlp.parameters(), 'lr': ae_config["resmlp_lr"]},    # Learning rate for ResMLP
                {'params': model.module.encoder.parameters(), 'lr': ae_config["ae_lr"]},  # Learning rate for encoder
                {'params': model.module.decoder.parameters(), 'lr': ae_config["ae_lr"]},  # Learning rate for decoder
            ]
            optimizer = optim.Adam(param_groups, betas=(0.9, 0.999),
                                   eps=1e-8, weight_decay=args.weight_decay)
            reduce_on_plateau_scheduler = \
                ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                  patience=50, verbose=True, min_lr=1e-6)
        """
        training
        """
        train_losses = AverageMeter()
        train_time_begin = time.time()
        for iter, batch in enumerate(trainloader):
            if batch is None:
                # skip empty batch due to missing data
                continue
            lr1 = optimizer.param_groups[0]['lr']
            lr2 = optimizer.param_groups[1]['lr']
            model.train()
            x_ae, x_resmlp, y_resmlp, filenames = prep_batchdata(
                args, batch, model.module.sub_region_mask)

            # compute output
            # l1_penalty = sum(torch.abs(param).sum() for param in model.parameters())
            if variational_flag:
                outputs_y, x_rec, mu, log_var = model(x_ae, x_resmlp)
                # double check this
                # kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(),dim=1).mean()
                vae_loss = vae_loss_fn(mu, log_var, x_rec, x_ae)
                kld = vae_gaussian_kl_loss(mu, log_var)
                loss_pred = 0
                if outputs_y is not None:
                    loss_pred = criterion(outputs_y, y_resmlp)
                loss_rec = criterion(x_rec, x_ae)
                if epoch <= ae_warmup_epochs:
                    loss = vae_loss
                else:
                    loss = args.pred_weight*loss_pred + \
                        args.rec_weight*(vae_loss)
            else:
                raise NotImplementedError
            #     outputs_y, x_rec = model(points_x)
            #     loss_pred = 0
            #     if outputs_y is not None:
            #         loss_pred = criterion(outputs_y, points_y)
            #     loss_rec = criterion(x_rec, points_x)
            #     loss = args.pred_weight*loss_pred + \
            #         args.rec_weight*(loss_rec) + ae_config["l1"]*l1_penalty
            # saving big checkpoints to see the reconstruction effect
            if (epoch+1)%20 == 0 and iter < 3:
                np.savez(
                    f"{args.checkpoint}/epoch{epoch}_iter{iter}_x_rec",
                    x=x_ae.detach().cpu().numpy(),
                    x_rec=x_rec.detach().cpu().numpy())

            # print(points_y)
            # print(torch.min(points_y))
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_mse = loss.item()
            train_losses.update(train_mse, batch[0].size(0))
            current_iters += 1
            if outputs_y is not None:
                loss_pred = loss_pred.item()
            if variational_flag:
                print('training- epoch:{}/{} | iters:{}/{}| lr:{:.2e},{:.2e} | \
                    train pred mse:{:.2e}| train rec mse: {:.2e} | kl: {:.2e} |'.format(
                        epoch, args.epoch, iter+1, len(trainloader),
                        lr1, lr2, loss_pred, loss_rec.item(), kld.item()))
            else:
                raise NotImplementedError
        train_time = time.time() - train_time_begin

        """
        testing
        """
        #### define the loss seperately ## we have 7 mse accordingly

        test_losses = {}
        for i in range(2):
            test_losses[i] = AverageMeter()
        loss_name = [args.output_type + '_pred: {:.2e} | ']
        loss_name.append(args.output_type + '_rec: {:.2e} | ')
        # loss_name.append(args.output_type + '_kl: {:.2e}')
        test_time_begin = time.time()
        for iter, batch in enumerate(testloader):
            if batch is None:
                # skip empty batch due to missing data
                continue
            suffix = 'testing- epoch:{}| iters:{}/{} |'.format(epoch, iter+1, len(testloader))
            # test_mses = tools.test_de(batch, model, criterion)
            model.eval()

            x_ae, x_resmlp, y_resmlp, filenames = prep_batchdata(
                args, batch, model.module.sub_region_mask)


        #     print('!!!!!!!!!!!!!!!!!batch',points_x.size())  #1024 122???

            # compute output
            if variational_flag:
                outputs_y, x_rec, mu, log_var = model(x_ae, x_resmlp)
                # kld = vae_gaussian_kl_loss(mu, log_var)
                # kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            else:
                raise NotImplementedError
                # outputs_y, x_rec = model(x_ae)
            #print("eval: ", outputs_y.size(), points_y.size())
            loss_pred = 0
            if outputs_y is not None:
                loss_pred = criterion(outputs_y, y_resmlp).item()
            loss_rec = reconstruction_loss(x_rec, x_ae, ltype="ssim").item()
            test_losses[0].update(loss_pred, batch[0].size(0))
            test_losses[1].update(loss_rec, batch[0].size(0))
            # test_losses[2].update(kl_divergence.item(), batch[0].size(0))
                # suffix = suffix + loss_name[i].format(1 - test_losses[i].avg/test_variance[args.output_type])
            suffix = suffix + loss_name[0].format(test_losses[0].avg)
            suffix = suffix + loss_name[1].format(test_losses[1].avg)
            print(suffix)


        test_time = time.time() - test_time_begin
        #if epoch < warmup_epochs:
            #warmup_scheduler.step()
        #else:
            #pass
            # reduce_on_plateau_scheduler.step(train_losses.avg)
            # reduce lr for validating
        if args.pred_weight >= args.rec_weight:
           reduce_on_plateau_scheduler.step(loss_pred)
        else:
           reduce_on_plateau_scheduler.step(loss_rec)

        current_datetime = datetime.now()
        print(f"Epoch {epoch} time: {current_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        #### save the log and ckpt ###################################
        save_log = [epoch, f"{lr1},{lr2}", train_losses.avg]
        for i in range(2):
            # save_log.append(1 - test_losses[i].avg/test_variance[args.output_type])
            save_log.append(test_losses[i].avg)

        save_log.append(train_time)
        save_log.append(test_time)
        logger.append(save_log)
        # if False and early_stopper.early_stop(test_losses[i].avg):
        #     tools.save_checkpoint({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, checkpoint=args.checkpoint, filename='checkpoint_epoch'+str(epoch)+'.pth.tar')
        #     break

        if (epoch+1)%10 == 0:
            tools.save_checkpoint({
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                # 'scheduler': reduce_on_plateau_scheduler.state_dict(),
                # 'epoch': epoch
                }, checkpoint=args.checkpoint, filename='checkpoint_epoch'+str(epoch)+'.pth.tar')

        valid_loss = test_losses[0].avg
        if args.pred_weight < args.rec_weight:
            valid_loss = test_losses[1].avg
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            print("Saving best model: epoch ", epoch)
            tools.save_checkpoint({
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
                }, checkpoint=args.checkpoint)



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
    parser.add_argument("--data_means", type=str)
    parser.add_argument("--data_stds", type=str)
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
