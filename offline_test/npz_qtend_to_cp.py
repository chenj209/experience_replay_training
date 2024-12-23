import numpy as np
import glob
import os
import datetime
import sys
import pandas as pd
from tqdm.autonotebook import tqdm
from metrics import get_thickness_from_ps_2d, get_thickness_from_ps_1d, report_metric, report_metric_vert, report_metric_spatial

def filename_to_datetime(filename):
    ts = int(filename.split("/")[-1].split(".")[0])
    start_date = datetime.datetime(1998, 1, 1, 0, 0, 0)
    return start_date + datetime.timedelta(hours=(ts-1)*0.5)

def qtend_to_precip(qtend, thickness):
    """
    qtend: (N, 30, 96, 144)
    thickness: (N, 30, 96, 144)
    """
    return np.sum(qtend*thickness, axis=1) / 1000.0 * 24 * 3600 *1000 * (-1)

if __name__ == "__main__":
    #npz_path = "/data/nncam_data/image_testset/"
    npz_path = "/share3/chenj209/online_data/noreplay_buffer_spinup5_seed1117_full_noprevQT/"
    pconsts = np.load(os.path.join(sys.path[0],"..","consts","phys_consts.npz"))
    hyai = pconsts["hyai"]
    hybi = pconsts["hybi"]
    precip_df = pd.DataFrame(columns=range(96*144))
    files = glob.glob(os.path.join(npz_path, "*.npz"))
    files.sort()
    for i in tqdm(range(3, 6641)):
        input_file = f"coupled_{i}.npz"
        output_file = f"coupled_{i}_pred.npz"
        input_npz_file = np.load(os.path.join(npz_path, input_file))
        output_npz_file = np.load(os.path.join(npz_path, output_file))
        qtend = output_npz_file["pred"][:,:30]
        ps = input_npz_file["x"][:,-1]*9495.39130101 + 96529.54020537
        thickness = get_thickness_from_ps_1d(ps, hyai, hybi)
        precip = qtend_to_precip(qtend, thickness)
        #date = filename_to_datetime(file)
        date = filename_to_datetime(f"{i}.npz")
        # print(date)
        # print(precip.shape)
        # print(precip.mean())
        if precip.mean() < 0 or precip.mean() > 1000:
            continue
        #precip_df.loc[pd.to_datetime(date)] = precip[0].flatten()
        precip_df.loc[pd.to_datetime(date)] = precip.flatten()
    # for file in tqdm(files):
    #     npz_file = np.load(os.path.join(npz_path, file))
    #     qtend = npz_file["data_y"][:,:30]
    #     ps = npz_file["data_x"][:,-1]
    #     thickness = get_thickness_from_ps_2d(ps, hyai, hybi)
    #     precip = qtend_to_precip(qtend, thickness)
    #     date = filename_to_datetime(file)
    #     if precip.mean() < 0 or precip.mean() > 1000:
    #         continue
    #     precip_df.loc[pd.to_datetime(date)] = precip[0].flatten()
    # precip_df.to_csv("spcam_precip.csv")
    precip_df.to_csv("noep_precip.csv")
    print(precip_df.head())
    print("Global mean precipitation (mm/day):", precip_df.mean())