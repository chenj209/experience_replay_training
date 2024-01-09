import concurrent.futures
import numpy as np
import glob
import argparse
import re
from dataloader_newformat import get_index_from_colnames
from tqdm.autonotebook import tqdm
def process_file(fn, var_names, col_names, region_mask, debug):
    data = np.load(fn)
    data_sums = {var_name: 0 for var_name in var_names}
    all_data = {var_name: [] for var_name in var_names} if debug else None

    for var_name in var_names:
        start, end = get_index_from_colnames(col_names, var_name)
        idx = list(range(start, end))
        cur_data = data[idx][:, region_mask].astype(np.float64)
        data_sums[var_name] += cur_data.sum()

        if debug:
            all_data[var_name].append(cur_data)

    return data_sums, all_data
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, default="./data/")
    parser.add_argument("--col_names", type=str, default="./data/col_names.txt")
    parser.add_argument("--region_mask", type=str, default="all")
    parser.add_argument("--out_name", type=str, default="test_data")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    if args.region_mask == "all":
        region_mask = np.ones((96,144)).astype(bool)
    else:
        region_mask = np.load(args.region_mask).astype(bool)

    col_names = np.loadtxt(args.col_names, dtype=str)
    # pattern = f"^(().*)(_lev\d+)?$"
    pattern = f"^(.*?)(?=_lev\d+|$)"
    var_names = []
    for col_name in col_names:
        match = re.match(pattern, col_name)
        if match and match.group(1) not in var_names:
            var_names.append(match.group(1))
    print(var_names)
    all_files = glob.glob(args.datapath + "/*.npy")

    # remove bad files
    for i in range(17507,17531):
        for file_name in all_files:
            if str(i) in file_name:
                all_files.remove(file_name)

    
    for i in ['00001', '00002', '00003', '08690', '08691', '17522', '17523','26210','26211']:
        for file_name in all_files:
            if i in file_name:
                all_files.remove(file_name)


    sample_file = np.load(all_files[0])
    # data_sums = {var_name: 0 for var_name in var_names}
    # print(region_mask.shape)
    # if args.debug:
    #     all_data = {var_name: [] for var_name in var_names}
    # for fn in tqdm(all_files):
    #     data = np.load(fn)
    #     # import ipdb; ipdb.set_trace()
    #     for var_name in var_names:
    #         start, end = get_index_from_colnames(col_names, var_name)
    #         idx = list(range(start,end))
    #         cur_data = data[idx][:,region_mask].astype(np.float64)
    #         data_sums[var_name] += cur_data.sum()
    #         if args.debug:
    #             all_data[var_name].append(cur_data)
    
    data_sums = {var_name: 0 for var_name in var_names}
    all_data = {var_name: [] for var_name in var_names} if args.debug else None

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_file, fn, var_names, col_names, region_mask, args.debug) for fn in all_files]
        
        # for future in concurrent.futures.as_completed(futures):
        #     file_data_sums, file_all_data = future.result()
        #     for var_name in var_names:
        #         data_sums[var_name] += file_data_sums[var_name]
        #         if args.debug:
        #             all_data[var_name].extend(file_all_data[var_name])
        with tqdm(total=len(all_files)) as progress:
            for future in concurrent.futures.as_completed(futures):
                file_data_sums, file_all_data = future.result()
                for var_name in var_names:
                    data_sums[var_name] += file_data_sums[var_name]
                    if args.debug:
                        all_data[var_name].extend(file_all_data[var_name])

                # Update the progress bar
                progress.update(1)
    data_means = data_sums
    for var_name in var_names:
        start, end = get_index_from_colnames(col_names, var_name)
        var_lvl = end-start
        data_means[var_name] = data_sums[var_name] / (len(all_files)*np.sum(region_mask)*var_lvl)
    # compute std
    data_std_sums = {var_name: 0 for var_name in var_names}
    for fn in tqdm(all_files):
        data = np.load(fn)
        for var_name in var_names:
            start, end = get_index_from_colnames(col_names, var_name)
            idx = list(range(start,end))
            cur_data = data[idx][:,region_mask].astype(np.float64)
            data_std_sums[var_name] += ((cur_data - data_means[var_name])**2).sum()

    data_stds = data_std_sums
    for var_name in var_names:
        start, end = get_index_from_colnames(col_names, var_name)
        var_lvl = end-start
        data_stds[var_name] = np.sqrt(data_std_sums[var_name] / (len(all_files)*np.sum(region_mask)*var_lvl))

    if args.debug:
        # check std and mean by comparing it with the all_data_x and all_data_y
        for var_name in var_names:
            all_data[var_name] = np.concatenate(all_data[var_name], axis=1)
            assert(np.allclose(all_data[var_name].mean(), data_means[var_name]))
            assert(np.allclose(all_data[var_name].std(), data_stds[var_name]))

    np.savez(args.out_name+"_means", **data_means)
    np.savez(args.out_name+"_stds", **data_stds)


