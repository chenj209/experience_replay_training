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
    max_precip = 160
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

def precip_analysis(qtend_pred, qtend_gt, thickness):
    """
    qtend_pred: (N, 30 ,96, 144)
    qtend_gt: (N, 30, 96, 144)
    qtend unit: kg/kg/s
    """
    print("qtend pred:", qtend_pred.mean(), qtend_pred.min(), qtend_pred.max())
    print(qtend_pred[0,:,0,0])
    print(qtend_pred.shape, qtend_gt.shape)
    print("thickenss:", thickness.mean(), thickness.min(), thickness.max())
    print(thickness[0,:,0,0])
    print(thickness.shape)
    precip_pred = np.sum(qtend_pred*thickness, axis=1) / 1000.0 * 24 * 3600 *1000 * (-1)
    precip_gt = np.sum(qtend_gt*thickness, axis=1) / 1000.0 * 24 * 3600  * 1000*(-1)
    print("precip shape:", precip_pred.shape, precip_gt.shape)
    print(precip_pred[0].mean(), precip_gt[0].mean())
    print(precip_pred.mean(), precip_gt.mean())
    print(precip_pred.max(), precip_gt.max())
    print(precip_pred.min(), precip_gt.min())
    ets_scores, far_scores, mar_scores = compute_scores(precip_pred, precip_gt)
    hist_pred = get_precip_hist(precip_pred)
    hist_gt = get_precip_hist(precip_gt)
    return ets_scores, far_scores, mar_scores, hist_pred, hist_gt

def plot_precip_hist(hist_pred, hist_gt):

    # Set up the plot
    plt.figure(figsize=(10, 6))
    
    # Plot histograms
    width = 0.35  # Width of the bars
    bins = np.arange(len(hist_pred))
    
    plt.bar(bins - width/2, hist_pred, width, label='Prediction', color='blue', alpha=0.6)
    plt.bar(bins + width/2, hist_gt, width, label='Ground Truth', color='red', alpha=0.6)

    # Set y-axis to log scale with custom ticks
    plt.yscale('log')
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
    from data_shape import inverse_to_inference_shape
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
    col_names_y = ["qtend_check", "SPPRECC"]
    col_names_prev = []

    pconsts = np.load(os.path.join(sys.path[0],"..","consts","phys_consts.npz"))
    hyai = pconsts["hyai"]
    hybi = pconsts["hybi"]
    get_thickness = lambda x : get_thickness_from_ps_2d(x,hyai, hybi)

    data_dir = DATA_DIR
    multistep = 0
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
    # col_names = np.loadtxt(data_dir + "/col_names.txt", dtype=str)
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
    transform = Compose([
        MinMaxTransformLegacy(include_raw=True),
        FlattenSpatialTransform(),
        ])
    testing_set = DatasetDisk(
        glob("/data/nncam_data/image_testset/" + "*.npz")[2:17520],
        input_indices,
        prev_input_indices,
        output_indices,
        multistep=int(multistep),
        sample_rate=int(144),
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
    model_ckpt_path = "/cust_users/x-w19/nncam.ckpts/resmlp.2years.50epochs/0_29_nodesize512_num_blocks7_actrelu_bs1024_scheduler_coslr_lr0.001_ep50_noise0.0_wd0_dropout0/checkpoint.pth.tar"
    model = load_resmlp_newformat2(model_ckpt_path, 122, 30)
    model.eval()
    gt_list = []
    raw_list = []
    pred_list = []
    thickness_list = []
    gt_precc_list = []
    with torch.no_grad():
        for i, batch in tqdm(enumerate(testloader), total=len(testloader)):
            points_x, points_y, x_raw, y_raw = batch[:4]
            gt_precc = y_raw[:,30].cpu().numpy()
            y_raw = y_raw[:,:30]
            points_y = points_y[:,:,:30]
            inputs = points_x.cuda()
            preds = model(inputs).cpu().numpy()
            thickness = get_thickness(x_raw[:,-1].cpu().numpy())
            thickness_list.append(thickness)
            gt = points_y.cpu().numpy()
            raw = y_raw.cpu().numpy()
            preds = inverse_to_inference_shape(legacy_inverse['0_29'](preds))
            gt = inverse_to_inference_shape(legacy_inverse['0_29'](gt))
            gt_list.append(gt)
            raw_list.append(raw)
            pred_list.append(preds)
            gt_precc_list.append(gt_precc)
    gt_list = np.concatenate(gt_list, axis=0)
    pred_list = np.concatenate(pred_list, axis=0)
    raw_list = np.concatenate(raw_list, axis=0)
    thickness_list = np.concatenate(thickness_list, axis=0)
    gt_precc_list = np.concatenate(gt_precc_list, axis=0)
    print(gt_precc_list.shape)
    print("mean precc: ", np.mean(gt_precc_list))
    print("mean precc: ", np.mean(gt_precc_list)*24*3600*1000)
    ets_scores, far_scores, mar_scores, hist_pred, hist_gt = precip_analysis(pred_list, raw_list, thickness_list)
    print("ETS score: ", ets_scores)
    print("FAR score: ", far_scores)
    print("MAR score: ", mar_scores)
    plot_precip_hist(hist_pred, hist_gt)
    plt.savefig("precip_hist.png")
    plt.show()
