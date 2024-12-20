from logger import debug_print
import sys
import os
import numpy as np
sys.path.append(os.path.join(sys.path[0], "..", "dataloader"))
from dataloader_utils import get_index_from_colnames

norm_err_file = open("norm_err_log.txt", 'w')  # flush print output immediately

DEBUG = False
NORM_THRES = 4

#def inverse_61_65(x):
#    return x * (1412 - 0)
inverse_legacy = {}
inverse_legacy["Q"] = lambda x: (x+1.0)*np.float64(0.0119)
inverse_legacy["T"] = lambda x: (x+1.0)*np.float64(82)+159
inverse_legacy["dqls"] = lambda x: (x)*np.float64(2.13e-6)
inverse_legacy["dTls"] = lambda x: (x)*np.float64(3.89e-3)
inverse_legacy["solin"] = lambda x: x*np.float64(1412-0)
inverse_legacy["ps"] = lambda x: x*np.float64(105782-59928) + np.float64(59928)
inverse_legacy['qtend']  = lambda x: x*np.float64(3.11e-6)
inverse_legacy['stend'] = lambda x: x*np.float64(3.63)
inverse_legacy['radiation'] = lambda x: x*np.float64(1412-0)

#inverse_legacy = {}
#inverse_legacy["Q"] = lambda x: (x+1.0)/2.0*(0.0238)+0
#inverse_legacy["T"] = lambda x: (x+1.0)/2.0*(323-159)+159
#inverse_legacy["dqls"] = lambda x: (x+1.0)/2.0*(2.13e-6*2)-2.13e-6
#inverse_legacy["dTls"] = lambda x: (x+1.0)/2.0*(3.89e-3*2)-3.89e-3
#inverse_legacy["solin"] = lambda x: x*(1412-0)
#inverse_legacy["ps"] = lambda x: x*(105782-59928) + 59928
#inverse_legacy['qtend']  = lambda x: (x+1.0)/2.0*(3.11e-6*2)-3.11e-6
#inverse_legacy['stend'] = lambda x: (x+1.0)/2.0*(3.63*2)-3.63
#inverse_legacy['radiation'] = lambda x: inverse_61_65(x)
def normalize_data_var_names2(data, var_names, col_names, data_mean, data_std, err_header="", threshold=NORM_THRES):
    x = data.copy()
    cur_idx = 0
    err_flag = False
    for var_name in var_names:
        start, end = get_index_from_colnames(col_names, var_name)
        data_range = end - start
        cur_data = x[cur_idx:cur_idx+data_range]
        debug_print(f"Normalizing {var_name}, idx ({cur_idx}:{cur_idx+data_range})", DEBUG)
        debug_print(f"Current data mean {cur_data.mean()}", DEBUG)
        cur_data = (cur_data - data_mean[var_name]) / (data_std[var_name]+1e-16)
        debug_print(f"Stored data mean/std {data_mean[var_name]}/{data_std[var_name]}", DEBUG)
        debug_print(f"Normalized data mean {cur_data.mean()}", DEBUG)
        if cur_data.mean() > threshold or cur_data.mean() < -threshold:
            print(f"Warning: {err_header} {var_name} mean out of range {cur_data.mean()}", file=norm_err_file, flush=True)
            print(f"Warning: {err_header} {var_name} mean out of range {cur_data.mean()}")
            err_flag = True
        x[cur_idx:cur_idx+data_range] = cur_data
        cur_idx += data_range
    data_norm = x
    if err_flag:
        return None
    return data_norm

def normalize_data_var_names(data, var_names, col_names, data_mean, data_std):
    x = data.copy()
    cur_idx = 0
    for var_name in var_names:
        debug_print(f"Normalizing {var_name}", DEBUG)
        start, end = get_index_from_colnames(col_names, var_name)
        data_range = end - start
        cur_data = x[:,cur_idx:cur_idx+data_range]
        debug_print(f"Current data mean {cur_data.mean()}", DEBUG)
        cur_data = (cur_data - data_mean[var_name]) / data_std[var_name]
        debug_print(f"Normalized data mean {cur_data.mean()}", DEBUG)
        x[:,cur_idx:cur_idx+data_range] = cur_data
        cur_idx += data_range
    data_norm = x
    return data_norm

def inverse_data_var_names(data, var_names, col_names, data_mean, data_std):
    x = data.copy()
    cur_idx = 0
    for var_name in var_names:
        start, end = get_index_from_colnames(col_names, var_name)
        data_range = end - start
        cur_data = x[:,cur_idx:cur_idx+data_range]
        cur_data = cur_data * data_std[var_name] + data_mean[var_name]
        x[:,cur_idx:cur_idx+data_range] = cur_data
        cur_idx += data_range
    data_inv = x
    return data_inv

def normalize_x(data_x):
    x = data_x.copy()

    x[:, 0:30,:,:]  = (x[:, 0:30,:,:] - 0) /(0.0238) * 2 - 1
    x[:,30:60,:,:]  = (x[:,30:60,:,:] - 159) / (323 - 159) * 2 - 1
    x[:,60:90,:,: ] = (x[:,60:90,:,:] + 2.13e-6) / (2.13e-6*2) * 2 - 1
    x[:,90:120,:,:] = (x[:,90:120,:,:] + 3.89e-3) / (3.89e-3*2) * 2 - 1
    x[:,120,:,:]    = (x[:,120,:,:] - 0)/ (1412 - 0)
    x[:,121,:,:]    = (x[:,121,:,:] - 59928) / (105782 - 59928)

    data_x_norm = x

    return data_x_norm

def normalize_y(data_y):
    y = data_y.copy()
    # output data (target)
    y[:, 0:30,:,:] = (y[:, 0:30,:,:] + 3.11e-6) / (3.11e-6*2) * 2 - 1
    y[:,30:60,:,:] = (y[:,30:60,:,:] + 3.63) / (3.63*2) * 2 - 1
    y[:,60:61,:,:]    =  y[:,60:61,:,:] / (2.12e-6) * 2 - 1

    y[:,61:62,:,:]    = (y[:,61:62,:,:] - 0) / (1412 - 0)
    y[:,62:63,:,:]    = (y[:,62:63,:,:] - 0) / (1412 - 0)
    y[:,63:64,:,:]    = (y[:,63:64,:,:] - 0) / (1412 - 0)
    y[:,64:65,:,:]    = (y[:,64:65,:,:] - 0) / (1412 - 0)
    y[:,65:66,:,:]    = (y[:,65:66,:,:] - 0) / (1412 - 0)


    data_y_norm = y

    return data_y_norm

def inverse_61_65(y):
    y[:,0] = (y[:,0]) * (1412 - 0)
    y[:,1] = (y[:,1]) * (1412 - 0)
    y[:,2] = (y[:,2]) * (1412 - 0)
    y[:,3] = (y[:,3]) * (1412 - 0)
    y[:,4] = (y[:,4]) * (1412 - 0)
    return y

def inverse_61_64(y):
    y[:,0] = (y[:,0]+1)/2*(332+53)-53       # flns
    y[:,1] = (y[:,1]+1)/2*(419-83)+83       # flnt
    y[:,2] = (y[:,2]+1)/2*(1063+2.13)-2.13  # fsns
    y[:,3] = (y[:,3]+1)/2*(1299)+0          # fsnt
    return y

def get_inverse():
    inverse = {}
    inverse[ '0_29'] = lambda y: (y+1)/2*(3.11e-6*2)-3.11e-6
    inverse['30_59'] = lambda y: (y+1)/2*(3.63*2)-3.63
    inverse['60']    = lambda y: (y+1)/2*(2.12e-6)
    inverse['61_64'] = lambda y: inverse_61_64(y)
    inverse['61_65'] = lambda y: inverse_61_65(y)

    return inverse

def get_inverse_newformat(col_names, data_mean, data_std):
    inverse = {}
    inverse['0_29'] = lambda y: inverse_data_var_names(y, "qtend_check", col_names, data_mean, data_std)
    inverse['30_59'] = lambda y: inverse_data_var_names(y, "stend_check", col_names, data_mean, data_std)
    return inverse

def normalization(data_x, data_y):
    x = data_x
    y = data_y
#     print('!!!!!!!!!',x.shape,y.shape)

    x[:, 0:30,:,:]  = (x[:, 0:30,:,:] - 0) /(0.0238) * 2 - 1
    x[:,30:60,:,:]  = (x[:,30:60,:,:] - 159) / (323 - 159) * 2 - 1
    x[:,60:90,:,: ] = (x[:,60:90,:,:] + 2.13e-6) / (2.13e-6*2) * 2 - 1
    x[:,90:120,:,:] = (x[:,90:120,:,:] + 3.89e-3) / (3.89e-3*2) * 2 - 1
    x[:,120,:,:]    = (x[:,120,:,:] - 0)/ (1412 - 0)
    x[:,121,:,:]    = (x[:,121,:,:] - 59928) / (105782 - 59928)

    # output data (target)
    y[:, 0:30,:,:] = (y[:, 0:30,:,:] + 3.11e-6) / (3.11e-6*2) * 2 - 1
    y[:,30:60,:,:] = (y[:,30:60,:,:] + 3.63) / (3.63*2) * 2 - 1
    y[:,60:61,:,:]    =  y[:,60:61,:,:] / (2.12e-6) * 2 - 1

    y[:,61:62,:,:]    = (y[:,61:62,:,:] - 0) / (1412 - 0)
    y[:,62:63,:,:]    = (y[:,62:63,:,:] - 0) / (1412 - 0)
    y[:,63:64,:,:]    = (y[:,63:64,:,:] - 0) / (1412 - 0)
    y[:,64:65,:,:]    = (y[:,64:65,:,:] - 0) / (1412 - 0)
    y[:,65:66,:,:]    = (y[:,65:66,:,:] - 0) / (1412 - 0)


    data_x_norm, data_y_norm = x,y

    return data_x_norm, data_y_norm
