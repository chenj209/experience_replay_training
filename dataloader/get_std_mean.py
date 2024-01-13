if __name__ == '__main__':
    import numpy as np
    import glob
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, default="./data/")
    parser.add_argument("--region_mask", type=str)
    parser.add_argument("--out_name", type=str)
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    region_mask = np.load(args.region_mask).astype(bool)

    all_files = glob.glob(args.datapath + "/*.npz")

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
    data_x_sum = np.zeros(sample_file['data_x'].shape[1])
    data_y_sum = np.zeros(sample_file['data_y'].shape[1])
    print(region_mask.shape)
    if args.debug:
        all_data_x = []
        all_data_y = []
    for fn in all_files:
        data = np.load(fn)
        data_x_sum += data['data_x'].astype(np.float64)[0,:, region_mask].sum(axis=0)
        data_y_sum += data['data_y'].astype(np.float64)[0,:, region_mask].sum(axis=0)
        if args.debug:
            all_data_x.append(data['data_x'][0,:, region_mask].astype(np.float64))
            all_data_y.append(data['data_y'][0,:, region_mask].astype(np.float64))
    data_x_mean = data_x_sum / (len(all_files)*np.sum(region_mask))
    data_y_mean = data_y_sum / (len(all_files)*np.sum(region_mask))
    # compute std
    data_x_std_sum = np.zeros(sample_file['data_x'].shape[1])
    data_y_std_sum = np.zeros(sample_file['data_y'].shape[1])
    for fn in all_files:
        data = np.load(fn)
        data_x_std_sum += ((data['data_x'][0,:, region_mask] - data_x_mean)**2).sum(axis=0)
        data_y_std_sum += ((data['data_y'][0,:, region_mask] - data_y_mean)**2).sum(axis=0)

    data_x_std = np.sqrt(data_x_std_sum / (len(all_files)*np.sum(region_mask)))
    data_y_std = np.sqrt(data_y_std_sum / (len(all_files)*np.sum(region_mask)))

    if args.debug:
        # check std and mean by comparing it with the all_data_x and all_data_y
        all_data_x = np.concatenate(all_data_x, axis=0)
        all_data_y = np.concatenate(all_data_y, axis=0)
        assert(np.allclose(all_data_x.mean(axis=0), data_x_mean))
        assert(np.allclose(all_data_y.mean(axis=0), data_y_mean))
        assert(np.allclose(all_data_x.std(axis=0), data_x_std))
        assert(np.allclose(all_data_y.std(axis=0), data_y_std))

    np.savez(args.out_name, data_x_mean=data_x_mean, data_y_mean=data_y_mean, data_x_std=data_x_std, data_y_std=data_y_std)


