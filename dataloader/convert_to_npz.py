import netCDF4 as nc
import pandas as pd
import numpy as np

out_dir = "/share3/chenj209/spcam_new_data_32/"

def to_npz(fn):
    print(f"Processing {fn}")
    try:
        dataset = nc.Dataset(fn)
    except:
        return None
    time_dim = dataset.dimensions['time'].size
    lev_dim = dataset.dimensions['lev'].size if 'lev' in dataset.dimensions else None
    lat_dim = dataset.dimensions['lat'].size
    lon_dim = dataset.dimensions['lon'].size

    # List of variables to process
    consts = []
    for variable in dataset.variables.keys():
        # print(variable, dataset.variables[variable].size)
        if dataset.variables[variable].size <= 48*8:
            consts.append(variable)
    variables_to_process = [variable for variable in dataset.variables.keys()
                            if variable not in dataset.dimensions and variable not in consts]  # Add all variables here
    #variables_to_process.remove("qtend_check")
    #variables_to_process.remove("stend_check")
    #print(variables_to_process)
    #print(consts)
    # for const in consts:
    #     print(dataset.variables[const])
    final_data = []

    # Process each variable
    for var in variables_to_process:
        # Extract the variable data
        var_data = dataset.variables[var][:]

        if 'lev' in dataset.variables[var].dimensions:
            # Flatten along the 'lev' dimension and reshape
            reshaped_data = var_data.reshape(time_dim, lev_dim, lat_dim, lon_dim)
            reshaped_data = reshaped_data.transpose(0, 2, 3, 1).reshape(-1, lev_dim)
        else:
            # Reshape to match the time, lat, lon structure
            reshaped_data = var_data.reshape(time_dim, lat_dim, lon_dim).reshape(-1, 1)
        # Create a DataFrame for the reshaped data
        final_data.append(reshaped_data)


    # # Create Multi-Index from time, lat, and lon
    time = np.repeat(dataset.variables['time'][:], lat_dim * lon_dim)[:,None]
    datesec = np.repeat(dataset.variables['datesec'][:], lat_dim * lon_dim)[:,None]
    lat = np.tile(np.repeat(dataset.variables['lat'][:], lon_dim), time_dim)[:,None]
    lon = np.tile(dataset.variables['lon'][:], time_dim * lat_dim)[:,None]
    final_data_np = np.concatenate([time, datesec, lat, lon, *final_data], axis=1)
    starting_idx = int(dataset.variables['time'][0]*48)+1
    for i in range(48):
        reshaped_data = final_data_np[i*96*144:(i+1)*96*144,:].reshape(96,144,-1).transpose(2,0,1)
        data = reshaped_data.data.astype(np.float32)
        np.save(out_dir + "/" + str(i+starting_idx).zfill(5)+".npy", data)

if __name__ == "__main__":
    import glob
    import multiprocessing

    # Open the NetCDF file
    #dataset = nc.Dataset('path_to_your_file.nc')
    #dataset = nc.Dataset('/global/cfs/cdirs/m4359/zhangtao/nncam/raw-spcam-data//spcam_std_rad.cam.h1.1997-01-01-00000.nc')
    dataset = nc.Dataset('/share1/x-w19/raw-spcam-data/spcam_std_rad.cam.h1.1997-01-01-00000.nc')

    # generate col_names

    # Read the dimensions
    time_dim = dataset.dimensions['time'].size
    lev_dim = dataset.dimensions['lev'].size if 'lev' in dataset.dimensions else None
    lat_dim = dataset.dimensions['lat'].size
    lon_dim = dataset.dimensions['lon'].size

    # List of variables to process
    consts = []
    for variable in dataset.variables.keys():
        # print(variable, dataset.variables[variable].size)
        if dataset.variables[variable].size <= 48*8:
            consts.append(variable)
    variables_to_process = [variable for variable in dataset.variables.keys()
                            if variable not in dataset.dimensions and variable not in consts]  # Add all variables here
    #variables_to_process.remove("qtend_check")
    #variables_to_process.remove("stend_check")
    print(variables_to_process)
    print(consts)
    # for const in consts:
    #     print(dataset.variables[const])
    final_data = []
    col_names_all = []

    # Process each variable
    for var in variables_to_process:
        if 'lev' in dataset.variables[var].dimensions:
            col_names = [f'{var}_lev{lev_idx}' for lev_idx in range(lev_dim)]
            col_names_all.extend(col_names)
        else:
            col_names = [var]
            col_names_all.extend(col_names)

    col_names_all = ["time", "datesec", "lat", "lon"] + col_names_all

    np.savetxt("col_names.txt", col_names_all, fmt="%s")

    #all_files = glob.glob("/global/cfs/cdirs/m4359/zhangtao/nncam/raw-spcam-data/*.nc")
    all_files = glob.glob("/share1/x-w19/raw-spcam-data/*.nc")
    all_files.sort()
    from multiprocessing import Pool
    with Pool(8) as p:
        print(p.map(to_npz, all_files))


