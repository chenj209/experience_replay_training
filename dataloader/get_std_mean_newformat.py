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

def process_file_std(fn, var_names, col_names, region_mask, data_means):
    data = np.load(fn)
    data_std_sums = {var_name: 0 for var_name in var_names}

    for var_name in var_names:
        start, end = get_index_from_colnames(col_names, var_name)
        idx = list(range(start, end))
        cur_data = data[idx][:, region_mask].astype(np.float64)
        data_std_sums[var_name] += ((cur_data - data_means[var_name])**2).sum()

    return data_std_sums

def region_slice2d(region_mask2d):
    mask, pad = region_mask2d
    min_x = np.min(np.where(mask)[0])
    max_x = np.max(np.where(mask)[0])
    min_y = np.min(np.where(mask)[1])
    max_y = np.max(np.where(mask)[1])
    return min_x-pad, max_x+pad, min_y-pad, max_y+pad 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, default="./data/")
    parser.add_argument("--region_mask", type=str, default="all")
    parser.add_argument("--out_name", type=str)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--col_names", type=str, default="./data/col_names.txt")

    args = parser.parse_args()
    if args.region_mask == "all":
        region_mask = np.ones((96,144)).astype(bool)
    else:
        region_mask = np.load(args.region_mask).astype(bool)
        # min_x, max_x, min_y, max_y = region_slice2d((region_mask, 2))
        # region_mask = np.zeros((96,144))
        # region_mask[min_x:max_x, min_y:max_y] = 1
        # region_mask = region_mask.astype(bool)

    col_names = np.loadtxt(args.col_names, dtype=str)
    # pattern = f"^(().*)(_lev\d+)?$"
    pattern = f"^(.*?)(?=_lev\d+|$)"
    # var_names = col_names

    # for each col_name, compute mean and std
    var_names = list(col_names)
    for col_name in col_names:
        match = re.match(pattern, col_name)
        if match and match.group(1) not in var_names:
            var_names.append(match.group(1))
    # remove duplicates from var_names
    var_names = list(set(var_names))
    var_names.sort()
    print(var_names)
    all_files = glob.glob(args.datapath + "/*.npy")
    all_files.sort()
    all_files = all_files[:35040]
    print(all_files[:10], "...", all_files[-10:])


    # remove bad files
    for i in range(17507,17531):
        for file_name in all_files:
            if str(i) in file_name:
                all_files.remove(file_name)
    for i in ['00001', '08690', '17522', '26210']:
        for file_name in all_files:
            if i in file_name:
                all_files.remove(file_name)
    for i in ['35042', '43730', '52562']:
        for file_name in all_files:
            if i in file_name:
                all_files.remove(file_name)
    for i in range(55015,55056):
        for file_name in all_files:
            if str(i) in file_name:
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
    # for fn in tqdm(all_files):
    #     data = np.load(fn)
    #     for var_name in var_names:
    #         start, end = get_index_from_colnames(col_names, var_name)
    #         idx = list(range(start,end))
    #         cur_data = data[idx][:,region_mask].astype(np.float64)
    #         data_std_sums[var_name] += ((cur_data - data_means[var_name])**2).sum()
    # data_std_sums = {var_name: 0 for var_name in var_names}

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_file_std, fn, var_names, col_names, region_mask, data_means): fn for fn in all_files}

        with tqdm(total=len(all_files)) as progress:
            for future in concurrent.futures.as_completed(futures):
                file_data_std_sums = future.result()
                for var_name in var_names:
                    data_std_sums[var_name] += file_data_std_sums[var_name]

                progress.update(1)


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


