import sys
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import numpy as np
import json
import time
import glob
from torch.utils import data

sys.path.append(os.path.join(sys.path[0], '..', 'consts'))
sys.path.append(os.path.join(sys.path[0], '..', 'dataloader'))
sys.path.append(os.path.join(sys.path[0], '..', 'models'))
import phys_consts
from dataloader_refactor import DatasetDisk
from load_models import load_models, get_inverse
from metrics import Regression_Metrics, Regression_Metrics_axis, reverse_operations, \
    report_qtend, report_stend, report_rad_prog, report_rad_prog_individual, \
    report_qtend_vert, report_stend_vert, report_qtend_spatial, report_stend_spatial, \
    get_thickness_from_ps_1d


def offline_test(args, all_models, testloader, get_thickness, silent=False, save=False):

    #test_losses = AverageMeter()
    #loss_name = [output_type + '_r2: {:.4e}']
    test_time_begin = time.time()
    epoch = 1
    criterion = nn.MSELoss()
    y_pred = []
    y_1 = []
    y_2 = []
    y_3 = []
    y_4 = []
    y_gt = []
    if args.region_mask == "all":
        region_mask = np.ones((1,1,96,144))
    else:
        region_mask = np.load(args.region_mask)[None, None, :, :]
    region_mask = np.transpose(region_mask, (0, 2, 3, 1))
    region_mask = np.reshape(region_mask, (-1, region_mask.shape[-1]))
    region_mask = (region_mask[:,0]==1)
    for iter, batch in enumerate(testloader):
        # allow empty batch
        print(f"testing {iter}/{len(testloader)}", end='\r')
        if batch[0].shape[0] == 0:
            continue
        suffix = 'testing- epoch:{}| iters:{}/{} |'.format(epoch, iter+1, len(testloader))
        batch[0] = batch[0].reshape(-1, batch[0].shape[-1])
        batch[1] = batch[1].reshape(-1, batch[1].shape[-1])
        batch[2] = batch[2].reshape(-1, batch[2].shape[-1])
        batch[0] = batch[0][region_mask, :]
        batch[1] = batch[1][region_mask, :]
        batch[2] = batch[2][region_mask, :]
        #model.eval()
        with torch.no_grad():
            points_x, points_y, x_raw = batch
            if get_thickness is not None:
                thickness = get_thickness(x_raw[:,121].numpy())
            #points_x, points_y = (points_x.float()).cuda(), (points_y.float()).cuda()
            points_x = (points_x.float()).cuda()
            y1 = get_inverse()['0_29'](all_models['0_29'](points_x).detach()
                                                        .cpu().numpy())
            if get_thickness is not None:
                y1 *= thickness * phys_consts.LATVAP
            y_1.append(y1)
            # y2 = get_inverse()['30_59'](all_models['30_59'](points_x).detach()
            #                                  .cpu().numpy())
            # if get_thickness is not None:
            #     y2 *= thickness
            # y_2.append(y2)
            #y_4.append(get_inverse()['61_65'](all_models['61_65'](points_x).detach()
                                             #.cpu().numpy()))
            if get_thickness is not None:
                points_y[:,:30] *= thickness*phys_consts.LATVAP
                points_y[:,30:60] *= thickness
            points_y = points_y.numpy()
            y_gt.append(points_y)


    y_1 = np.concatenate(y_1, axis=0)
    # y_2 = np.concatenate(y_2, axis=0)
    #y_4 = np.concatenate(y_4, axis=0)
    y_gt = np.concatenate(y_gt, axis=0)
    if save:
        if get_thickness is not None:
            np.save("qtend_pred_thickness.npy", y_1)
            np.save("stend_pred_thickness.npy", y_2)
            np.save("spcam_gt_thickness.npy", y_gt)
        else:
            np.save("qtend_pred.npy", y_1)
            np.save("stend_pred.npy", y_2)
            np.save("spcam_gt.npy", y_gt)



    test_time = time.time() - test_time_begin

    qtend_log = report_qtend(y_gt, y_1)
    qtend_log_lvl = report_qtend_vert(y_gt, y_1)
    if args.region_mask == "all":
        qtend_log_spatial = report_qtend_spatial(y_gt, y_1)
    #print(json.dumps(qtend_log_lvl, indent=4))
    del y_1

    # stend_log = report_stend(y_gt, y_2)
    # stend_log_lvl = report_stend_vert(y_gt, y_2)
    # stend_log_spatial = report_stend_spatial(y_gt, y_2)
    # del y_2

    #rad_log = report_rad_prog(y_gt, y_4)

    #rad_log_individual = report_rad_prog_individual(y_gt, y_4)

    res = {
        "qtend_log": qtend_log,
        # "stend_log": stend_log,
        #"rad_log": rad_log,
        #"rad_log_individual": rad_log_individual,
        "qtend_log_lvl": qtend_log_lvl,
        # "stend_log_lvl": stend_log_lvl,
        # "stend_log_spatial": stend_log_spatial
    }
    if args.region_mask == "all":
        res["qtend_log_spatial"] = qtend_log_spatial
    return res

if __name__ == "__main__":
    import argparse
    import random
    #dh_settings.get_weight()
    #dh_settings.get_hyai_hybi()
    torch.multiprocessing.set_sharing_strategy('file_system')
    random.seed(0)
    np.random.seed(0)
    parser = argparse.ArgumentParser()
    #parser.add_argument("--output_type", "-ot", help="choose from 0-29, 30-59, 61-65")
    #parser.add_argument("--resume", "-re", help="path to selected model")
    parser.add_argument("config", help="path to configuration file")
    parser.add_argument("out_json", help="path to output json file")
    parser.add_argument("--sample", type=int, help="sample frequency to use", default=1)
    parser.add_argument("--thick", action="store_true")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--region_mask", type=str)
    args = parser.parse_args()
    print(args.config)
    with open(args.config, "r") as f:
        config = json.load(f)
    all_models = load_models(
        config["0-29"]["ckpt_path"],
        # config["30-59"]["ckpt_path"],
        None,
        None,
        #config["61-65"]["ckpt_path"]
        None
    )
    np.random.seed(0)
    #data_dir = "/data/nncam_data/image_testset/"
    data_dir = "/home/users/data/nncam_data/image_testset/"
    print("Test set path: ", data_dir)

    cudnn.benchmark = True

    all_files = glob.glob(data_dir+'/*')[2::args.sample]
    for fn in all_files:
        if "08691" in fn:
            all_files.remove(fn)
    all_files.sort()
    print("all_files len ", len(all_files))
    test_idx = np.random.choice(len(all_files), len(all_files), replace=False)
    print(test_idx[:10])
    test_files = [all_files[i] for i in test_idx]
    print("Test file size: " ,len(test_files))
    print(test_files[:3])

    testing_set = DatasetDisk(file_names=test_files, is_train=False, noise_std=0, output_normalized=False, silent=True)
    testloader = data.DataLoader(testing_set, shuffle=False, batch_size=1, num_workers=1)
    if args.thick:
        pconsts = np.load(os.path.join(sys.path[0],"..","consts","phys_consts.npz"))
        hyai = pconsts["hyai"]
        hybi = pconsts["hybi"]
        get_thickness = lambda x : get_thickness_from_ps_1d(x,hyai, hybi) 
    else:
        get_thickness = None

    logs = offline_test(args, all_models, testloader, get_thickness, save=args.save)

    with open(args.out_json, "w") as f:
        json.dump({
            "dq/dt": logs["qtend_log"],
            #"dT/dt": logs["stend_log"],
            #"radiation": rad_log,
            #"dqdt_lvl": qtend_log_lvl,
            #"dTdt_lvl": stend_log_lvl,
            #**logs["rad_log_individual"]
            }, f, indent=4)
    # with open("ex_"+args.out_json, "w") as f:
    #     json.dump({
    #         #"dq/dt": qtend_log,
    #         #"dT/dt": stend_log,
    #         "radiation": logs["rad_log"],
    #         "dqdt_lvl": logs["qtend_log_lvl"],
    #         "dTdt_lvl": logs["stend_log_lvl"],
    #         #**rad_log_individual
    #         }, f, indent=4)

    if args.region_mask == "all":
        np.savez(f"ex_qtend_{args.out_json.rstrip('.json')}_spatial.npz", **logs["qtend_log_spatial"])
    #np.savez(f"ex_stend_{args.out_json.rstrip('.json')}_spatial.npz", **logs["stend_log_spatial"])
    np.savez(f"ex_qtend_{args.out_json.rstrip('.json')}_vert.npz", **logs["qtend_log_lvl"])
    #np.savez(f"ex_stend_{args.out_json.rstrip('.json')}_vert.npz", **logs["stend_log_lvl"])





