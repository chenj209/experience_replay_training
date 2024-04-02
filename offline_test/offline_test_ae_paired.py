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
from torchvision.transforms import Compose
from datetime import datetime
# from torch.utils.data.dataloader import default_collate

sys.path.append(os.path.join(sys.path[0], "..", "models"))
import autoencoder
import variational_autoencoder
sys.path.append(os.path.join(sys.path[0], ".."))
from utils import Logger, AverageMeter, mkdir_p
sys.path.append(os.path.join(sys.path[0], "..", "utils"))
# import argsparser
import tools
from data_shape import to_inference_shape, inverse_to_inference_shape
sys.path.append(os.path.join(sys.path[0], "..", "dataloader"))
from preprocess import StandardizeTransform, get_min_max_coords
from dataloader_newformat import PairDatasetDisk, filter_collate
from dataloader_utils import gen_multistep_col_indices, levelwise_variable2, \
    delevelwise_variable, levelwise_variable

START_LEV = 6
END_LEV = 29

from metrics import Regression_Metrics, Regression_Metrics_axis, reverse_operations, \
    report_qtend, report_stend, report_rad_prog, report_rad_prog_individual, \
    report_qtend_vert, report_stend_vert, report_qtend_spatial, report_stend_spatial, \
    get_thickness_from_ps_1d

def prep_dataloaders(
    args,
    test_files,
    input_indices1,
    input_indices2,
    prev_input_indices1,
    prev_input_indices2,
    output_indices1,
    output_indices2,
    transform1,
    transform2,
    region_mask):

    testing_set = PairDatasetDisk(
        test_files,
        curr_input_indices1=input_indices1,
        curr_input_indices2=input_indices2,
        prev_input_indices1=prev_input_indices1,
        prev_input_indices2=prev_input_indices2,
        output_indices1=output_indices1,
        output_indices2=output_indices2,
        multistep=int(args.multistep),
        sample_rate=int(args.sample_rate),
        is_train=False,
        transform1=transform1,
        transform2=transform2,
        include_filename=True,
        region_mask2d=(region_mask,2)
        )
    testloader = data.DataLoader(testing_set, shuffle=True,
                                 batch_size=1,
                                 num_workers=4,
                                 collate_fn=filter_collate,
                                 pin_memory=True)

    return testloader

def prep_models(args, ae_config, output_size, sub_region_mask):
    print(f"Model input size: {ae_config['input_size']}")
    #model = autoencoder.AutoencoderResMLP(input_size, 30, args.node_size, \
    # args.activation, args.num_blocks, args.latent_dim, region_mask=region_mask, resmlp=(args.pred_weight!=0))
    print(f"Loading model config: {json.dumps(ae_config, indent=4)}")
    if "variational" in ae_config and ae_config["variational"] is True:
        model_struc = variational_autoencoder.SepInputAutoencoderResMLP
        variational_flag = True
    else:
        raise NotImplementedError
        model_struc = autoencoder.AutoencoderResMLP
        variational_flag = False
    model = model_struc(
        encoder_config=ae_config["encoder"],
        decoder_config=ae_config["decoder"],
        #input_size=len(training_set.input_indices),
        input_size=ae_config["input_size"][0],
        output_size=output_size,
        m=512,
        activation='relu',
        num_blocks=7,
        sub_region_mask=sub_region_mask
    )
    print("Model structure:")
    print(model)
    print('Total params: %.2f' % (sum(p.numel() for p in model.parameters())))

    model.float()
    model = torch.nn.DataParallel(model).cuda()
    #model = model.cuda()
    model.load_state_dict(torch.load(args.resume)["state_dict"])
    cudnn.benchmark = True

    return model, variational_flag

def prep_batchdata(args, batch, sub_region_mask):
    x_ae, _, x_resmlp, y_resmlp, x_raw, y_raw, filenames = batch
    #lr = lr_scheduler[args.lr_strategy](optimizer, args.lr, current_iters, len(trainloader) * args.epoch)
    # lr = scheduler.get_lr()[-1]
    # if args.output_type == '0-29':
    #     y_resmlp = y_resmlp[:, :, :30]
    # if args.output_type == '30-59':
    #     y_resmlp = y_resmlp[:, :, 30:60]
    # if args.output_type == '60':
    #     y_resmlp = y_resmlp[:, :, 60:61]
    # if args.output_type == '61-65':
    #     y_resmlp = y_resmlp[:, :, 61:66]
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
    return x_ae, x_resmlp, y_resmlp, x_raw, y_raw, filenames

def main(args):
    with open(args.ae_config, 'r') as f:
        ae_config = json.load(f)
    # region_mask = to_inference_shape(region_mask).squeeze()
    data_dir = "/pscratch/sd/c/chenjd21/spcam_new_data_32/"
    col_names = np.loadtxt(data_dir + "/col_names.txt", dtype=str)
    col_names_x = [
        "QL_lev",
        "T_nn_in_lev",
        "dqvls_nn_in_lev",
        "dTls_nn_in_lev",
        "SOLIN",
        "SPPS",
        "LWUP",
        "CAPE",
        "UL_lev",
        "VL_lev"
        ]+args.ex_input
    prev_ex_vars = [
        "qtend_check_lev",
        "stend_check_lev",
        "SOLL",
        "SOLLD",
        "SOLS",
        "SOLSD",
        "FSDS",
        "CLOUD_lev",
        "SPPRECC",
        "FLNS",
        "FLNT",
        "SPQRL_lev",
        "SPQRS_lev"
        ]+args.ex_input_prev
    if ae_config["reduce_lvl"]:
        col_names_x_ae = levelwise_variable2(col_names_x, [12,18,23,28,29])
        prev_ex_vars_ae = levelwise_variable2(prev_ex_vars, [12,18,23,28,29])
    else:
        col_names_x_ae = levelwise_variable(col_names_x, START_LEV, END_LEV)
        prev_ex_vars_ae = levelwise_variable(prev_ex_vars, START_LEV, END_LEV)
    col_names_x_resmlp = delevelwise_variable(col_names_x)
    prev_ex_vars_resmlp = delevelwise_variable(prev_ex_vars)
    col_names_y = ["qtend_check"]
    data_means = dict(np.load(data_dir + "/data_means.npz"))
    data_means_by_lvl = dict(np.load(data_dir + "/std_mean_by_level_means.npz"))
    data_means.update(data_means_by_lvl)
    data_stds = dict(np.load(data_dir + "/data_stds.npz"))
    data_stds_by_lvl = dict(np.load(data_dir + "/std_mean_by_level_stds.npz"))
    data_stds.update(data_stds_by_lvl)

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

    region_mask = None
    if args.region_mask is not None and args.region_mask != "all":
        region_mask = np.load(args.region_mask)
    else:
        region_mask = np.ones((96,144))
    min_x, max_x, min_y, max_y = get_min_max_coords(region_mask, 2)
    lon = np.linspace(0,357.5,144)
    lat = np.linspace(-90,90,96)
    print("Region window coordinates: ", lon[min_y], lon[max_y], lat[min_x], lat[max_x])
    sub_region_mask = region_mask[min_x:max_x, min_y:max_y]

    if args.thick:
        pconsts = np.load(os.path.join(sys.path[0],"..","consts","phys_consts.npz"))
        hyai = pconsts["hyai"]
        hybi = pconsts["hybi"]
        get_thickness = lambda x : get_thickness_from_ps_1d(x,hyai, hybi)
    else:
        get_thickness = None

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

    transform_ae = Compose([
        # RectRegionMaskTransform(region_mask),
        StandardizeTransform(
            data_means,
            data_stds,
            multistep_col_names_x_ae,
            [],
            col_names,
            normalize_input=True,
            normalize_output=True
            ),
    ])
    transform_resmlp = Compose([
        # RectRegionMaskTransform(region_mask),
        StandardizeTransform(
            data_means,
            data_stds,
            multistep_col_names_x_resmlp,
            col_names_y,
            col_names,
            normalize_input=True,
            normalize_output=True,
            include_raw=True
            ),
    ])

    # ae_input_size = len(prev_input_indices_ae)*int(args.multistep)+len(input_indices_ae)
    #resmlp_input_size = len(input_indices)
    # use all the inputs for better offline performance now
    #resmlp_input_size = ae_input_size
    # resmlp_input_size = len(prev_input_indices_resmlp)*int(args.multistep)+len(input_indices_resmlp)
    ae_config["input_size"][0] = len(prev_input_indices_resmlp)*int(args.multistep)+len(input_indices_resmlp)
    ae_config["encoder"]["input_size"][0] = len(prev_input_indices_ae)*int(args.multistep)+len(input_indices_ae)
    ae_config["decoder"]["input_size"][0] = len(prev_input_indices_ae)*int(args.multistep)+len(input_indices_ae)


    model, variational_flag = prep_models(args, ae_config,
                        len(output_indices_resmlp), sub_region_mask)
    all_files = glob.glob(data_dir+'/*.npy')
    all_files.sort()
    # testing data starts from 35040
    test_files = all_files[35040:]
    testloader = prep_dataloaders(args, test_files,input_indices_ae,input_indices_resmlp,\
                                  prev_input_indices_ae, prev_input_indices_resmlp,\
                                  output_indices_ae,\
                                  output_indices_resmlp, transform_ae,\
                                  transform_resmlp, region_mask)

    criterion = nn.MSELoss()
    avg_mse = 0
    avg_pred_mse = 0
    avg_mse_by_variable = np.zeros(ae_config["encoder"]["input_size"][0])
    avg_mse_by_level = np.zeros(30)
    current_iters = 0
    y_gt = []
    y_pred = []
    for iter, batch in enumerate(testloader):
        if batch is None:
            continue
        model.eval()
        # points_x, points_y, x_raw, _ = batch[:4]
        # filenames = [fname[0].split("/")[-1].rstrip(".npy") for fname in batch[-1]]

        # # reshape output prediction to shape of resmlp 1D output
        # points_y = points_y[:, :, model.module.sub_region_mask]
        # points_y = points_y.permute(0,2,1).reshape(-1, points_y.shape[1])

        # points_x = (points_x.float()).cuda()
        # points_y = (points_y.float()).cuda()
        x_ae, x_resmlp, y_resmlp, x_raw, y_raw, filenames = \
            prep_batchdata(args, batch, model.module.sub_region_mask)
        if get_thickness is not None:
            x_raw = x_raw[:, :, model.module.sub_region_mask]
            x_raw = x_raw.permute(0,2,1).reshape(-1, x_raw.shape[1])
            thickness = get_thickness(x_raw[:,-1].numpy())

        if variational_flag:
            outputs_y, x_rec, mu, log_var = model(x_ae, x_resmlp)
        else:
            raise NotImplementedError
            outputs_y, x_rec = model(points_x)

        loss_rec = criterion(x_rec, x_ae).item()
        loss_pred = criterion(outputs_y, y_resmlp).item()
        avg_mse += loss_rec
        avg_pred_mse += loss_pred
        avg_mse_by_variable += np.mean((x_rec - x_ae).cpu().detach().numpy()**2, axis=(0,2,3))
        avg_mse_by_level += np.mean((points_y - y_resmlp).cpu().detach().numpy()**2, axis=0)
        x_ae = x_ae.cpu().detach().numpy()
        y_resmlp = y_resmlp.cpu().detach().numpy()
        x_rec = x_rec.cpu().detach().numpy()
        outputs_y = outputs_y.cpu().detach().numpy()
        if filenames[0] in ['37621', '41029', '42601', '44018', '44270', '47930', '53919', '53823', '50666']:
            np.save(f"{args.save_path}/offline_test_ae_{'-'.join(filenames)}_x.npy", x_ae)
            np.save(f"{args.save_path}/offline_test_ae_{'-'.join(filenames)}_x_rec.npy", x_rec)
            np.save(f"{args.save_path}/offline_test_ae_{'-'.join(filenames)}_y.npy", y_resmlp)
            np.save(f"{args.save_path}/offline_test_ae_{'-'.join(filenames)}_y_pred.npy", outputs_y)
            if variational_flag:
                np.save(f"{args.save_path}/offline_test_ae_{'-'.join(filenames)}_mu.npy", mu.cpu().detach().numpy())
                np.save(f"{args.save_path}/offline_test_ae_{'-'.join(filenames)}_log_var.npy", log_var.cpu().detach().numpy())
        if get_thickness is not None:
            outputs_y = outputs_y * thickness
            points_y = points_y * thickness
        y_gt.append(points_y)
        y_pred.append(outputs_y)

        current_iters += 1
        print('testing: iters:{}/{}| pred mse:{:.2e} | rec mse:{:.2e} |'\
              .format(iter+1, len(testloader), loss_pred, loss_rec))
    avg_mse /= current_iters
    avg_mse_by_variable /= current_iters
    avg_mse_by_level /= current_iters
    avg_pred_mse /= current_iters
    np.save(f"{args.save_path}/offline_test_ae_avg_recmse_by_var.npy", avg_mse_by_variable)
    np.save(f"{args.save_path}/offline_test_ae_avg_predmse_by_level.npy", avg_mse_by_level)
    print("Average pred MSE by level:")
    for i in range(30):
        print(f"Level {i}: {avg_mse_by_level[i]}")
    print(f"Average rec MSE: {avg_mse}")
    print(f"Average pred MSE: {avg_pred_mse}")
    np.save(f"{args.save_path}/offline_test_ae_subregion_mask.npy", sub_region_mask)
    y_gt = np.concatenate(y_gt, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    qtend_log = report_qtend(y_gt, y_pred)
    qtend_log_lvl = report_qtend_vert(y_gt, y_pred)
    print(qtend_log_lvl)

if __name__ == "__main__":
    import argparse
    import numpy as np
    import torch
    parser = argparse.ArgumentParser(description="offline test for autoencoder")
    parser.add_argument("--ae_config", type=str, help="path to ae config file")
    parser.add_argument("--multistep", type=int, help="multistep", default=1)
    parser.add_argument("--resume", type=str, help="path to ae model file")
    parser.add_argument("--save_path", type=str, help="path to save results", default="offline_test")
    parser.add_argument("--ex_input", type=str, nargs="*", default=[])
    parser.add_argument("--ex_input_prev", type=str, nargs="*", default=[])
    parser.add_argument("--sample_rate", type=int, help="sample_rate", default=12)
    #parser.add_argument("--latent_dim", type=int, help="latent_dim", default=256)
    #parser.add_argument("--ex_input", type=str, nargs="*", default=[])
    parser.add_argument('--region_mask', type=str, help='path to region mask npy file', default="all")
    parser.add_argument("--thick", action="store_true", help="use thick mask")
    args = parser.parse_args()
    torch.multiprocessing.set_sharing_strategy('file_system')
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    main(args)
