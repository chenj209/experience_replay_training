import matplotlib.pyplot as plt
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'consts'))
import phys_consts
def get_precip_hist(precip):
    """
    precip: (N, 96, 144)
    precip unit: mm/day
    """
    max_precip = 200
    bins = np.arange(0, max_precip+1, 1)
    hist, _ = np.histogram(precip, bins=bins, density=True)
    return hist

def compute_scores(precip_pred, precip_gt, threshold=[0.1,10,25,50,100]):
    """
    qtend_pred: (N, 96, 144)
    qtend_gt: (N, 96, 144)
    """
    # compute ETS score for each threshold
    ets_scores = []
    far_scores = []
    mar_scores = []
    for th in threshold:
        # Calculate contingency table elements
        hits = np.sum((precip_pred > th) & (precip_gt > th))  # True positives
        false_alarms = np.sum((precip_pred > th) & (precip_gt <= th))  # False positives 
        misses = np.sum((precip_pred <= th) & (precip_gt > th))  # False negatives
        correct_negatives = np.sum((precip_pred <= th) & (precip_gt <= th))  # True negatives
        
        # Calculate total number of forecasts
        total = hits + false_alarms + misses + correct_negatives
        
        # Calculate random hits
        hits_random = ((hits + misses) * (hits + false_alarms)) / total
        
        # Calculate ETS score
        ets = (hits - hits_random) / (hits + misses + false_alarms - hits_random)
        ets_scores.append(ets)

        # Calculate FAR score (False Alarm Rate)
        far = false_alarms / (hits + false_alarms)
        far_scores.append(far)

        # Calculate MAR score (Miss Alarm Rate)
        mar = misses / (hits + misses)
        mar_scores.append(mar)

    return ets_scores, far_scores, mar_scores

def qtend_to_precip(qtend, thickness):
    return np.sum(qtend*thickness, axis=1) / 1000.0 * 24 * 3600 *1000 * (-1)

def precip_analysis(precip_pred, precip_gt, thickness):
    """
    qtend_pred: (N, 30 ,96, 144)
    qtend_gt: (N, 30, 96, 144)
    qtend unit: kg/kg/s
    """
    # print(qtend_pred.shape, qtend_gt.shape)
    # print(thickness.shape)
    # precip_pred = np.sum(qtend_pred*thickness, axis=1) / 1000.0 * 24 * 3600 *1000 * (-1)
    # precip_gt = np.sum(qtend_gt*thickness, axis=1) / 1000.0 * 24 * 3600  * 1000*(-1)
    print("precip shape:", precip_pred.shape, precip_gt.shape)
    print(precip_pred.mean(), precip_gt.mean())
    print(precip_pred.max(), precip_gt.max())
    print(precip_pred.min(), precip_gt.min())
    ets_scores, far_scores, mar_scores = compute_scores(precip_pred, precip_gt)
    hist_pred = get_precip_hist(precip_pred)
    hist_gt = get_precip_hist(precip_gt)
    return ets_scores, far_scores, mar_scores, hist_pred, hist_gt

def plot_precip_hist(hist_preds, hist_gt):
    # Set up the plot
    plt.figure(figsize=(10, 6))
    
    # Create x-axis values (precipitation bins)
    bins = np.arange(len(hist_preds[list(hist_preds.keys())[0]]))
    for model_name, hist_pred in hist_preds.items():
        plt.semilogy(bins, hist_pred, label=model_name, linewidth=2)
    plt.semilogy(bins, hist_gt, label="SPCAM", color='red', linewidth=2)

    # Set y-axis limits and ticks
    plt.ylim(1e-5, 100)
    plt.yticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100])

    # Customize x-axis
    plt.xlabel('Precipitation (mm/day)')
    plt.ylabel('Frequency (%)')
    
    # Add legend
    plt.legend()
    
    # Add grid for better readability
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    plt.tight_layout()
    return plt.gcf()

def plot_verification_scores(ets_dict, far_dict, mar_dict, thresholds):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot ETS scores
    for model_name, scores in ets_dict.items():
        ax1.plot(thresholds, scores, label=model_name, marker='o')
    ax1.set_xlabel('Threshold (mm/day)')
    ax1.set_ylabel('ETS Score')
    ax1.set_title('Equitable Threat Score')
    ax1.grid(True, alpha=0.2)
    ax1.legend()
    
    # Plot FAR scores
    for model_name, scores in far_dict.items():
        ax2.plot(thresholds, scores, label=model_name, marker='o')
    ax2.set_xlabel('Threshold (mm/day)')
    ax2.set_ylabel('FAR Score')
    ax2.set_title('False Alarm Rate')
    ax2.grid(True, alpha=0.2)
    ax2.legend()
    
    # Plot MAR scores
    for model_name, scores in mar_dict.items():
        ax3.plot(thresholds, scores, label=model_name, marker='o')
    ax3.set_xlabel('Threshold (mm/day)')
    ax3.set_ylabel('MAR Score')
    ax3.set_title('Miss Alarm Rate')
    ax3.grid(True, alpha=0.2)
    ax3.legend()
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    # test plot_precip_hist
    import torch
    from glob import glob
    from tqdm.autonotebook import tqdm
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'dataloader'))
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
    from configs import *
    from dataloader_legacy_format import DatasetDisk, filter_collate
    from dataloader_utils import gen_multistep_col_indices, gen_col_indices
    from torchvision.transforms import Compose
    from preprocess import MinMaxTransformLegacy, StandardizeTransform, \
    FlattenSpatialTransform, get_min_max_coords
    from torch.utils import data
    from load_models import load_resmlp_newformat2
    from offline_test_newformat import legacy_inverse
    from data_shape import inverse_to_inference_shape, to_inference_shape
    from metrics import get_thickness_from_ps_2d

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
    col_names_x = [
        "QL", 
        "T_nn_in", 
        "dqvls_nn_in", 
        "dTls_nn_in", 
        "SOLIN", 
        "SPPS"
        ]
    # col_names_y = ["qtend_check", "SPPRECC"]
    col_names_y = ["qtend_check"]
    col_names_prev = ["qtend_check", "stend_check", "SOLL", "SOLS", "SOLSD", "SOLLD", "FSDS"]

    pconsts = np.load(os.path.join(sys.path[0],"..","consts","phys_consts.npz"))
    hyai = pconsts["hyai"]
    hybi = pconsts["hybi"]
    get_thickness = lambda x : get_thickness_from_ps_2d(x,hyai, hybi)

    data_dir = DATA_DIR
    multistep = 1
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
    col_names = np.loadtxt(data_dir + "/col_names.txt", dtype=str)
    data_means = dict(np.load(os.path.join(sys.path[0],"..","consts","all_means.npz")))
    data_stds = dict(np.load(os.path.join(sys.path[0],"..","consts","all_stds.npz")))
    # col_names_x = ["QL", "T_nn_in", "dqvls_nn_in", "dTls_nn_in", "SOLIN", "SPPS"]
    # prev_ex_vars = ["qtend_check", "stend_check", "SOLL", "SOLLD", "SOLS", "SOLSD", "FSDS"]
    # col_names_y = ["qtend_check"]
    # input_indices, prev_input_indices, output_indices = gen_multistep_col_indices(
    #     col_names, prev_ex_vars, col_names_x, col_names_y, multistep
    # )
    multistep_col_names_x = []
    for i in range(int(multistep)):
        multistep_col_names_x.extend(col_names_x)
        multistep_col_names_x.extend(col_names_prev)
    multistep_col_names_x.extend(col_names_x)
    print("multistep: ", multistep_col_names_x)
    transform = Compose([
        StandardizeTransform(
                data_means,
                data_stds,
                multistep_col_names_x,
                col_names_y,
                col_names,
                normalize_input=True,
                normalize_output=True,
                include_raw=True,
                threshold=1e10
                    ),
        FlattenSpatialTransform(),
        ])
    testing_set = DatasetDisk(
        glob("/data/nncam_data/image_testset/" + "*.npz")[2:1460],
        input_indices,
        prev_input_indices,
        output_indices,
        multistep=int(multistep),
        sample_rate=int(1),
        is_train=True,
        transform=transform,
        include_filename=True,
        ex_dir="/zz/share3/chenj209/spcam_new_data_32/"
    )
    testloader = data.DataLoader(testing_set, shuffle=False,
                                 batch_size=1,
                                 num_workers=1,
                                 collate_fn=filter_collate,
                                 pin_memory=True)
    # model_ckpt_path = "/cust_users/x-w19/nncam.ckpts/resmlp.2years.50epochs/0_29_nodesize512_num_blocks7_actrelu_bs1024_scheduler_coslr_lr0.001_ep50_noise0.0_wd0_dropout0/checkpoint.pth.tar"
    #model_ckpt_path = "/share3/chenj209/ckpts_sampled12/conv_mem/replay_buffer309_v2_seed1117_full/0_29_checkpoint_best_loss.pth.tar"
    models = {}
    model_ckpt_path = "/share3/chenj209/ckpts_sampled12/conv_mem/noreplay_buffer309_v2_seed1117_full_noprevQT/0_29_checkpoint_best_loss.pth.tar"
    models["PM+No ER"] = load_resmlp_newformat2(model_ckpt_path, 249, 30)
    models["PM+No ER"].eval()
    model_ckpt_path = "/share3/chenj209/ckpts_sampled12/conv_mem/replay_buffer309_v2_seed1117_full_noprevQT/0_29_checkpoint_best_loss.pth.tar"
    models["PM+ER"] = load_resmlp_newformat2(model_ckpt_path, 249, 30)
    models["PM+ER"].eval()
    model_ckpt_path = "/cust_users/x-w19/nncam.ckpts/resmlp.2years.50epochs/0_29_nodesize512_num_blocks7_actrelu_bs1024_scheduler_coslr_lr0.001_ep50_noise0.0_wd0_dropout0/checkpoint.pth.tar"
    models["No PM+No ER"] = load_resmlp_newformat2(model_ckpt_path, 122, 30)
    models["No PM+No ER"].eval()
    # gt_list = []
    raw_list = []
    pred_list = { model_name: [] for model_name in models.keys() }
    thickness_list = []
    with torch.no_grad():
        for i, batch in tqdm(enumerate(testloader), total=len(testloader)):
            points_x, points_y, x_raw, y_raw = batch[:4]
            # gt_precc = y_raw[:,30].cpu().numpy()
            y_raw = y_raw[:,:30]
            points_y = points_y[:,:,:30]
            inputs = points_x.cuda()
            thickness = get_thickness(x_raw[:,-1].cpu().numpy())
            thickness_list.append(thickness)
            for model_name, model in models.items():
                if model_name == "No PM+No ER":
                    inputs_legacy, _ = MinMaxTransformLegacy(include_raw=False)((x_raw[0,-122:].cpu().numpy(), y_raw[0].cpu().numpy()))
                    inputs_legacy = to_inference_shape(inputs_legacy[None,])
                    preds = model(torch.from_numpy(inputs_legacy)).cpu().numpy()
                    preds = inverse_to_inference_shape(legacy_inverse['0_29'](preds))
                    preds = qtend_to_precip(preds, thickness)
                else:
                    # print("inputs shape:", inputs.shape)
                    preds = model(inputs[:,:,60:]).cpu().numpy()
                    preds = inverse_to_inference_shape((preds)*data_stds["qtend_check"]+data_means["qtend_check"])
                    preds = qtend_to_precip(preds, thickness)
                pred_list[model_name].append(preds)
            gt = points_y.cpu().numpy()
            raw = y_raw.cpu().numpy()
            gt = inverse_to_inference_shape((gt)*data_stds["qtend_check"]+data_means["qtend_check"])
            gt = qtend_to_precip(gt, thickness)
            # gt_list.append(gt)
            raw_list.append(gt)
    # gt_list = np.concatenate(gt_list, axis=0)
    for model_name in models.keys():
        pred_list[model_name] = np.concatenate(pred_list[model_name], axis=0)
    raw_list = np.concatenate(raw_list, axis=0)
    thickness_list = np.concatenate(thickness_list, axis=0)
    hist_preds = { model_name: [] for model_name in models.keys() }
    thresholds = [0.1, 10, 25, 50, 100]
    ets_dict = {}
    far_dict = {}
    mar_dict = {}
    
    for model_name in models.keys():
        print(model_name)
        ets_scores, far_scores, mar_scores, hist_pred, hist_gt = precip_analysis(pred_list[model_name], raw_list, thickness_list)
        hist_preds[model_name] = hist_pred*100
        ets_dict[model_name] = ets_scores
        far_dict[model_name] = far_scores
        mar_dict[model_name] = mar_scores
        print("ETS score: ", ets_scores)
        print("FAR score: ", far_scores)
        print("MAR score: ", mar_scores)
    
    # Plot and save verification scores
    fig_scores = plot_verification_scores(ets_dict, far_dict, mar_dict, thresholds)
    fig_scores.savefig("verification_scores_noprevQT.png")
    plt.close(fig_scores)
    
    # Original histogram plotting
    plot_precip_hist(hist_preds, hist_gt*100)
    plt.savefig("precip_hist_all_noprevQT_1998Jan.png")
    plt.show()
