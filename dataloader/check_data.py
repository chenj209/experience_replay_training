import re
import numpy as np
from multiprocessing import Pool, Manager
from tqdm import tqdm

def process_file(args):
    file, var_names, col_names, means, stds = args
    results = []
    problem_files = []
    try:
        data = np.load(file)
        err_flag = 0
        for name in var_names:
            s, e = get_index_from_colnames(col_names, name)
            var_data = data[s:e].reshape(-1)
            if name not in ["time", "datesec"] and np.all(var_data == 0):
                results.append(f"{file} contains invalid {name} data: all zeros")
            if np.mean(var_data) > means[name]+stds[name]*10 or np.mean(var_data) < means[name]-stds[name]*10:
                mean_val = np.mean(var_data)
                results.append(f"{file} contains invalid {name} data: outlier mean value {mean_val} from {means[name]} +/- {stds[name]}")
        if len(results) > 0:
            problem_files = [file]
    except Exception as e:
        results.append(f"Error processing {file}: {str(e)}")

    return results, problem_files

def main(files, var_names, col_names, means, stds, num_processes=4):
    pool_args = [(file, var_names, col_names, means, stds) for file in files]
    problem_files = []
    err_results = []

    with Pool(processes=num_processes) as pool:
        with tqdm(total=len(files)) as pbar:
            for res in pool.imap_unordered(process_file, pool_args):
                result, problem_file = res
                pbar.update(1)
                for message in result:
                    print(message)
                problem_files.extend(problem_file)
                err_results.extend(result)
    problem_files.sort()
    print(problem_files)
    with open("problem_files.txt", "w") as f:
        for file in problem_files:
            f.write(file)
            f.write("\n")
    with open("err_results.txt", "w") as f:
        for res in err_results:
            f.write(res)
            f.write("\n")

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
    means = {key:means[key] for key in means.files}
    stds = np.load(data_dir + "data_stds.npz")
    stds = {key:stds[key] for key in stds.files}
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
    #for file in tqdm(files):
    #    data = np.load(file)
    #    for name in var_names:
    #        s,e = get_index_from_colnames(col_names, name)
    #        var_data = data[s:e].reshape(-1)
    #        if name not in ["time", "datesec"] and np.all(var_data == 0):
    #            print(f"{file} contains invalid {name} data: all zeros")
    #        if np.mean(var_data) > means[name]+stds[name]*10 or np.mean(var_data) < means[name]-stds[name]*10:
    #            print(f"{file} contains invalid {name} data: outlier mean value {np.mean(var_data)} from {means[name]} +/- {stds[name]}")


    main(files, var_names, col_names, means, stds)

