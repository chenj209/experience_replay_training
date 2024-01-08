if __name__ == '__main__':
    import numpy as np
    import glob
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, default="./data/")
    parser.add_argument("--region_mask", type=str, default="all")
    parser.add_argument("--out_name", type=str)
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    if args.region_mask == "all":
        region_mask = np.ones((96,144)).astype(bool)
    else:
        region_mask = np.load(args.region_mask).astype(bool)

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
    data_sum = np.zeros(sample_file.shape[0])
    print(region_mask.shape)
    if args.debug:
        all_data = []
    for fn in all_files:
        data = np.load(fn)
        # import ipdb; ipdb.set_trace()
        cur_sum = data.astype(np.float64)[:, region_mask].sum(axis=1)
        data_sum += cur_sum
        if args.debug:
            all_data.append(data[:, region_mask].astype(np.float64))
    data_mean = data_sum / (len(all_files)*np.sum(region_mask))
    # compute std
    data_std_sum = np.zeros(sample_file.shape[0])
    for fn in all_files:
        data = np.load(fn)
        data_std_sum += ((data[:, region_mask] - data_mean[:,None])**2).sum(axis=1)

    data_std = np.sqrt(data_std_sum / (len(all_files)*np.sum(region_mask)))

    if args.debug:
        # check std and mean by comparing it with the all_data_x and all_data_y
        all_data = np.concatenate(all_data, axis=1)
        print(all_data.shape)
        assert(np.allclose(all_data.mean(axis=1), data_mean))
        assert(np.allclose(all_data.std(axis=1), data_std))

    np.savez(args.out_name, data_mean=data_mean, data_std=data_std)


