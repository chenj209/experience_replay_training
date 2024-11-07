import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import torch.utils.data as data
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

def plot_precip_hist(hist_preds, model_names):
    # Set up the plot
    plt.figure(figsize=(10, 6))
    
    # Create x-axis values (precipitation bins)
    bins = np.arange(len(hist_preds[0]))
    for hist_pred, model_name in zip(hist_preds, model_names):
        plt.semilogy(bins, hist_pred, label=model_name, linewidth=2)

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

class PrecipDataset(data.Dataset):
    def __init__(self, ep_path, no_ep_path, start_idx=3, end_idx=1460):
        self.ep_path = ep_path
        self.no_ep_path = no_ep_path
        self.indices = list(range(start_idx, end_idx))
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        i = self.indices[idx]
        ep_data = np.load(self.ep_path + f"coupled_{i}.npz")['x']
        no_ep_data = np.load(self.no_ep_path + f"coupled_{i}.npz")['x']
        return ep_data, no_ep_data

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
    from offline_test_newformat import legacy_inverse, inverse_output
    from data_shape import inverse_to_inference_shape
    from metrics import get_thickness_from_ps_2d, get_thickness_from_ps_1d

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
    get_thickness = lambda x : get_thickness_from_ps_1d(x,hyai, hybi)

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
    data_means = np.load(os.path.join(sys.path[0],"..","consts","all_means.npz"))
    data_stds = np.load(os.path.join(sys.path[0],"..","consts","all_stds.npz"))
    # transform = Compose([
    #     MinMaxTransformLegacy(include_raw=True),
    #     FlattenSpatialTransform(),
    #     ])
    # testing_set = DatasetDisk(
    #     glob("/data/nncam_data/image_testset/" + "*.npz")[2:17520],
    #     input_indices,
    #     prev_input_indices,
    #     output_indices,
    #     multistep=int(multistep),
    #     sample_rate=int(144),
    #     is_train=True,
    #     transform=transform,
    #     include_filename=True,
    #     ex_dir="/zz/share3/chenj209/spcam_new_data_32/"
    # )
    # testloader = data.DataLoader(testing_set, shuffle=False,
    #                              batch_size=1,
    #                              num_workers=1,
    #                              collate_fn=filter_collate,
    #                              pin_memory=True)
    ep_model_ckpt_path = "/share3/chenj209/ckpts_sampled12/conv_mem/replay_buffer309_v2_seed1117_full_noprevQT/0_29_checkpoint_best_loss.pth.tar"
    noep_model_ckpt_path = "/share3/chenj209/ckpts_sampled12/conv_mem/noreplay_buffer309_v2_seed1117_full_noprevQT/0_29_checkpoint_best_loss.pth.tar"
    ep_model = load_resmlp_newformat2(ep_model_ckpt_path, 249, 30)
    noep_model = load_resmlp_newformat2(noep_model_ckpt_path, 249, 30)
    ep_model.eval()
    noep_model.eval()
    gt_list = []
    raw_list = []
    pred_list = []
    thickness_list = []
    gt_precc_list = []

    no_ep_path = "/zz/share3/chenj209/noreplay_buffer_spinup5_seed1117_full_noprevQT/"
    ep_path = "/share3/chenj209/online_data_ep_noprevQT_rerun/"
    variables = ["Q", "T", "dqls_prev", "dTls_prev", "solin_prev", "ps_prev", "qtend_prev", "stend_prev", "rad_prev", "Q", "T", "dqls", "dTls", "solin", "ps"]
    dims = [30,30,30,30,1,1,30,30,5,30,30,30,30,1,1]
    idx_range = []
    cur_idx = 0
    for v,d in zip(variables, dims):
        idx_range.append([cur_idx, cur_idx+d])
        cur_idx = cur_idx+d
    # no_ep_vars = ["dqls_prev", "qtend_prev", "Q", "dqls" ]
    vars_source = {
        "dqls_prev": "ep",
        "dTls_prev": "ep",
        "solin_prev": "ep",
        "ps_prev": "ep",
        "qtend_prev": "ep",
        "stend_prev": "ep",
        "rad_prev": "ep",
        "Q": "noep",
        "T": "ep",
        "dqls": "ep",
        "dTls": "ep",
        "solin": "ep",
        "ps": "ep",
    }
    # ep_vars = ["dqls_prev", "dTls_prev", "solin_prev", "ps_prev", "qtend_prev", "stend_prev", "rad_prev", "Q", "T", "dqls", "dTls", "solin", "ps"]
    # ep_vars = ["dqls_prev", "dTls_prev", "qtend_prev", "stend_prev", "rad_prev", "Q", "T", "dqls", "dTls", "solin", "ps"]
    # ep_vars = ["dqls_prev", "solin_prev", "ps_prev", "qtend_prev", "rad_prev", "Q", "T", "dqls", "dTls", "solin", "ps"]
    # no_ep_vars = ["dqls_prev", "dTls_prev", "solin_prev", "ps_prev", "qtend_prev", "stend_prev", "rad_prev", "Q", "T", "dqls", "dTls", "solin", "ps"]
    # no_ep_vars = ["solin_prev", "ps_prev"]
    # no_ep_vars = []
    ep_base_list = []
    noep_base_list = []
    ep_list = []
    noep_list = []
    dataset = PrecipDataset(ep_path, no_ep_path)
    dataloader = data.DataLoader(dataset, batch_size=32, num_workers=4, shuffle=False)
    with torch.no_grad():
        for ep_data, no_ep_data in tqdm(dataloader):
            # Reshape batch dimension
            ep_data = ep_data.reshape(-1, ep_data.shape[-1])
            no_ep_data = no_ep_data.reshape(-1, no_ep_data.shape[-1])
            
            recon_data = []
            for v, d in zip(variables, idx_range):
                if vars_source[v] == "ep":
                    recon_data.append(ep_data[:,d[0]:d[1]])
                elif vars_source[v] == "noep":
                    recon_data.append(no_ep_data[:,d[0]:d[1]])
            recon_data = torch.cat(recon_data, dim=1)[:,60:]
            
            inputs = recon_data.float().cuda()
            ep_preds = ep_model(inputs).cpu()
            noep_preds = noep_model(inputs).cpu()
            ep_base_preds = ep_model((ep_data.float().cuda())[:,60:]).cpu()
            noep_base_preds = noep_model((no_ep_data.float().cuda())[:,60:]).cpu()
            if vars_source["ps"] == "ep":
                ps = ep_data[:,-1]
            else:
                ps = no_ep_data[:,-1]
            ps = ps * 9495.39130101 + 96529.54020537
            thickness = get_thickness(ps.numpy())
            thickness_list.append(thickness)
            
            ep_preds = ep_preds * data_stds["qtend_check"] + data_means["qtend_check"]
            noep_preds = noep_preds * data_stds["qtend_check"] + data_means["qtend_check"]
            ep_base_preds = ep_base_preds * data_stds["qtend_check"] + data_means["qtend_check"]
            noep_base_preds = noep_base_preds * data_stds["qtend_check"] + data_means["qtend_check"]
            ep_list.append(np.sum(ep_preds.numpy() * thickness, axis=1) * 24 * 3600 * (-1))
            noep_list.append(np.sum(noep_preds.numpy() * thickness, axis=1) * 24 * 3600 * (-1))
            ep_base_list.append(np.sum(ep_base_preds.numpy() * thickness, axis=1) * 24 * 3600 * (-1))
            noep_base_list.append(np.sum(noep_base_preds.numpy() * thickness, axis=1) * 24 * 3600 * (-1))
    ep_list = np.concatenate(ep_list, axis=0)
    noep_list = np.concatenate(noep_list, axis=0)
    ep_base_list = np.concatenate(ep_base_list, axis=0)
    noep_base_list = np.concatenate(noep_base_list, axis=0)
    ep_hist = get_precip_hist(ep_list)
    noep_hist = get_precip_hist(noep_list)
    ep_base_hist = get_precip_hist(ep_base_list)
    noep_base_hist = get_precip_hist(noep_base_list)
    plot_precip_hist([noep_base_hist*100, ep_base_hist*100, noep_hist*100, ep_hist*100], ["noep_base", "ep_base", "noep", "ep"])
    plt.savefig("precip_no_ep_input_Q.png")
    plt.show()
