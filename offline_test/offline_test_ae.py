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
import argsparser
import tools
from data_shape import to_inference_shape, inverse_to_inference_shape
sys.path.append(os.path.join(sys.path[0], "..", "dataloader"))
from preprocess import StandardizeTransform, get_min_max_coords
from dataloader_newformat import DatasetDisk, filter_collate
from dataloader_utils import gen_multistep_col_indices

def prep_dataloaders(
    args, 
    test_files,
    input_indices, 
    prev_input_indices, 
    output_indices,
    region_mask):

    testing_set = DatasetDisk(
        test_files,
        input_indices,
        prev_input_indices,
        output_indices,
        is_train=False,
        multistep=int(args.multistep),
        sample_rate=int(args.sample_rate),
        include_filename=True,
        region_mask2d=(region_mask,2)
        )
    testloader = data.DataLoader(testing_set, shuffle=True, 
                                 batch_size=args.train_batch, 
                                 num_workers=args.workers,
                                 collate_fn=filter_collate,
                                 pin_memory=True)

    return testloader

def prep_models(args, resmlp_input_size, ae_input_size, output_size, sub_region_mask):
    with open(args.ae_config, 'r') as f:
        ae_config = json.load(f)
    ae_config['input_size'] = ae_input_size
    print(f"Model input size: {ae_config['input_size']}")
    #model = autoencoder.AutoencoderResMLP(input_size, 30, args.node_size, args.activation, args.num_blocks, args.latent_dim, region_mask=region_mask, resmlp=(args.pred_weight!=0))
    print(f"Loading model config: {json.dumps(ae_config, indent=4)}")
    if "variational" in ae_config and ae_config["variational"] is True:
        model_struc = variational_autoencoder.AutoencoderResMLP
    else:
        model_struc = autoencoder.AutoencoderResMLP
    model = model_struc(
        config=ae_config,
        input_size=resmlp_input_size,
        output_size=output_size,
        m=512,
        activation='relu',
        num_blocks=7,
        sub_region_mask=sub_region_mask
    )
    print("Model structure:")
    print(model)
    print('Total params: %.2f' % (sum(p.numel() for p in model.parameters())))


    model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(torch.load(args.resume))
    cudnn.benchmark = True

    return model

def main(args):
    # region_mask = to_inference_shape(region_mask).squeeze()
    data_dir = "/pscratch/sd/c/chenjd21/spcam_new_data/"
    col_names = np.loadtxt(data_dir + "/col_names.txt", dtype=str)
    col_names_x = ["QL", "T_nn_in", "dqvls_nn_in", "dTls_nn_in", "SOLIN", "SPPS"]+args.ex_input
    prev_ex_vars = ["qtend_check", "stend_check", "SOLL", "SOLLD", "SOLS", "SOLSD", "FSDS"]+args.ex_input_prev
    col_names_y = ["qtend_check"]
    data_means = dict(np.load(data_dir + "/data_means.npz"))
    data_stds = dict(np.load(data_dir + "/data_stds.npz"))

    all_files = glob.glob(data_dir+'/*.npy')
    all_files.sort()
    # testing data starts from 35040
    test_files = all_files[35040:]

    input_indices, prev_input_indices, output_indices = gen_multistep_col_indices(
        col_names, prev_ex_vars, col_names_x, col_names_y, int(args.multistep)
    )
    print("Input indices: ", col_names[input_indices])
    print("Prev input indices: ", col_names[prev_input_indices])
    print("Output indices: ", col_names[output_indices])


    testloader = prep_dataloaders(args, test_files, input_indices, 
                                  prev_input_indices, output_indices)

    ae_input_size = len(prev_input_indices)*int(args.multistep)+len(input_indices)
    resmlp_input_size = len(input_indices)
    model = prep_models(args, resmlp_input_size, ae_input_size, 
                        len(output_indices), region_mask)

    region_mask = np.load(args.region_mask)
    min_x, max_x, min_y, max_y = get_min_max_coords(region_mask.squeeze(), 3)
    lon = np.linspace(0,357.5,144)
    lat = np.linspace(-90,90,96)
    print("Region window coordinates: ", lon[min_y], lon[max_y], lat[min_x], lat[max_x])
    criterion = nn.MSELoss()
    avg_mse = 0
    avg_mse_by_variable = np.zeros(309)
    for iter, batch in enumerate(testloader):
            if batch[0].size() == 1 and batch[0] == 0:
                continue
            batch[0] = batch[0][:, :, min_x: max_x, min_y: max_y]

            model.eval()
            points_x = batch[0]
            points_x = (points_x.float()).cuda()
            x_rec = model(points_x)
            loss_rec = criterion(x_rec, points_x).item()
            avg_mse += loss_rec
            avg_mse_by_variable += np.mean((x_rec - points_x).cpu().detach().numpy()**2, axis=(0,2,3))
            if iter < 10:
                np.save(f"{args.save_path}/offline_test_ae_{iter}_x.npy", points_x.cpu().detach().numpy())
                np.save(f"{args.save_path}/offline_test_ae_{iter}_x_rec.npy", x_rec.cpu().detach().numpy())

            current_iters += 1
            print('testing: iters:{}/{}| mse:{:.6f}|'.format(iter+1, len(testloader), loss_rec))
    avg_mse /= current_iters
    avg_mse_by_variable /= current_iters
    np.save(f"{args.save_path}/offline_test_ae_avg_mse.npy", avg_mse)
    print(f"Average MSE: {avg_mse}")

if __name__ == "__main__":
    args = argsparser.argsparser()
    parser = argsparser.prepare_parser()
    parser.add_argument("--ae_config", type=str, help="path to ae config file")
    parser.add_argument("--multistep", type=int, help="multistep", default=1)
    parser.add_argument("--resume", type=str, help="path to ae model file")
    parser.add_argument("--save_path", type=str, help="path to save results", default="offline_test")
    # parser.add_argument("--sample_rate", type=int, help="sample_rate", default=12)
    #parser.add_argument("--latent_dim", type=int, help="latent_dim", default=256)
    #parser.add_argument("--ex_input", type=str, nargs="*", default=[])
    parser.add_argument('--region_mask', type=str, help='path to region mask npy file', default="all")
    main(args)