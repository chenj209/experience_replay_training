import re
def col_name_cmp(var_name, col_name):
    pattern = f"^{var_name}(_lev\d+)?$"
    # print(pattern)
    return re.match(pattern, col_name) is not None

def get_index_from_colnames(col_names, var_name):
    start_idx = -1
    end_idx = -1
    for i, col_name in enumerate(col_names):
        if col_name_cmp(var_name, col_name) and start_idx == -1:
            start_idx = i
        if col_name_cmp(var_name, col_name):
            end_idx = i
        if end_idx != -1 and not col_name_cmp(var_name, col_name):
            return start_idx, end_idx+1
    return start_idx, end_idx+1

if __name__ == "__main__":
    import numpy as np
    import sys
    import glob
    from tqdm.autonotebook import tqdm
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--start_ts", type=int, default=0)
    args = parser.parse_args()
    #filename = f"/pscratch/sd/c/chenjd21/spcam_new_data/{str(sys.argv[1]).zfill(5)}.npy"
    #print(np.load(filename).mean())
    #print(np.load(filename)[683:683+30].mean())
    data_dir = "/pscratch/sd/c/chenjd21/spcam_new_data/"
    means = np.load(data_dir + "data_means.npz")
    stds = np.load(data_dir + "data_stds.npz")
    col_names = np.loadtxt(data_dir + "col_names.txt", dtype=str)
    pattern = f"^(.*?)(?=_lev\d+|$)"
    var_names = []
    for col_name in col_names:
        match = re.match(pattern, col_name)
        if match and match.group(1) not in var_names:
            var_names.append(match.group(1))
    files = glob.glob(data_dir + "*.npy")
    files.sort()
    files = files[args.start_ts:]
    print(files[:10])
    for file in tqdm(files):
        data = np.load(file)
        for name in var_names:
            s,e = get_index_from_colnames(col_names, name)
            var_data = data[s:e].reshape(-1)
            if name not in ["time", "datesec"] and np.all(var_data == 0):
                print(f"{file} contains invalid {name} data: all zeros")
            if np.mean(var_data) > means[name]+stds[name]*10 or np.mean(var_data) < means[name]-stds[name]*10:
                print(f"{file} contains invalid {name} data: outlier mean value {np.mean(var_data)} from {means[name]} +/- {stds[name]}")
