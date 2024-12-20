import numpy as np
from normalization import normalize_x, normalize_y, inverse_legacy
def normalize_x2(data_x):
    x = data_x.copy()

    x[:, 0:30,:,:]  = (x[:, 0:30,:,:] - 0) /np.float64(0.0119) - 1
    x[:,30:60,:,:]  = (x[:,30:60,:,:] - np.float64(159)) / np.float64(82) - 1
    x[:,60:90,:,: ] = (x[:,60:90,:,:]) / np.float64(2.13e-6)
    x[:,90:120,:,:] = x[:,90:120,:,:]/np.float64(3.89e-3)
    x[:,120,:,:]    = (x[:,120,:,:] - 0)/ np.float64(1412 - 0)
    x[:,121,:,:]    = (x[:,121,:,:] - np.float64(59928)) / np.float64(105782 - 59928)

    data_x_norm = x

    return data_x_norm

def normalize_y2(data_y):
    y = data_y.copy()
    # output data (target)
    #y[:, 0:30,:,:] = (y[:, 0:30,:,:] + 3.11e-6) / (3.11e-6*2) * 2 - 1
    y[:, 0:30,:,:] = y[:,0:30,:,:]/np.float64(3.11e-6)
    #y[:,30:60,:,:] = (y[:,30:60,:,:] + np.float64(3.63)) / (np.float64(3.63)*2) * np.float64(2) - np.float64(1)
    y[:,30:60,:,:] = y[:,30:60,:,:]/np.float64(3.63)
    y[:,60:61,:,:]    =  y[:,60:61,:,:] / (2.12e-6) * 2 - 1

    y[:,61:62,:,:]    = (y[:,61:62,:,:] - 0) / np.float64(1412 - 0)
    y[:,62:63,:,:]    = (y[:,62:63,:,:] - 0) / np.float64(1412 - 0)
    y[:,63:64,:,:]    = (y[:,63:64,:,:] - 0) / np.float64(1412 - 0)
    y[:,64:65,:,:]    = (y[:,64:65,:,:] - 0) / np.float64(1412 - 0)
    y[:,65:66,:,:]    = (y[:,65:66,:,:] - 0) / np.float64(1412 - 0)


    data_y_norm = y

    return data_y_norm

inverse_legacy2 = {}
inverse_legacy2["Q"] = lambda x: (x+1.0)*np.float64(0.0119)
inverse_legacy2["T"] = lambda x: (x+1.0)*np.float64(82)+159
inverse_legacy2["dqls"] = lambda x: (x)*np.float64(2.13e-6)
inverse_legacy2["dTls"] = lambda x: (x)*np.float64(3.89e-3)
inverse_legacy2["solin"] = lambda x: x*np.float64(1412-0)
inverse_legacy2["ps"] = lambda x: x*np.float64(105782-59928) + np.float64(59928)
inverse_legacy2['qtend']  = lambda x: x*np.float64(3.11e-6)
#inverse_legacy['stend'] = lambda x: (x+np.float64(1.0))/np.float64(2.0)*(np.float64(3.63*2))-np.float64(3.63)
inverse_legacy2['stend'] = lambda x: x*np.float64(3.63)
inverse_legacy2['radiation'] = lambda x: x*np.float64(1412-0)

if __name__ == "__main__":
    data = np.load("/data/nncam_data/image_set/00100.npz")
    #data_x = data["data_x"].astype(np.float64)
    data_x = data["data_x"]
    norm_x = normalize_x(data_x)
    norm_x2 = normalize_x2(data_x)
    assert(np.allclose(norm_x, norm_x2,atol=1e-6, rtol=1e-5))
    #data_y = data["data_y"].astype(np.float64)
    data_y = data["data_y"]
    norm_y = normalize_y(data_y)
    norm_y2 = normalize_y2(data_y)
    assert(np.allclose(norm_y, norm_y2,atol=1e-6, rtol=1e-5))
    index = {
        "Q": [0, 30],
        "T": [30, 60],
        "dqls": [60, 90],
        "dTls": [90,120],
        "solin": [120,121],
        "ps": [121,122],
        "qtend": [0,30],
        "stend": [30,60],
        "radiation": [61,66]
    }
    for var_name in ["Q", "T", "dqls", "dTls", "solin", "ps"]:
        print(var_name)
        idx = index[var_name]
        raw_var = data_x[:,idx[0]:idx[1]]
        norm_var = norm_x2[:,idx[0]:idx[1]]
        inv_var = inverse_legacy2[var_name](norm_var)
        #assert(np.allclose(raw_var, inv_var, atol=1e-6, rtol=1e-5))
        assert(np.allclose(raw_var, inv_var))
    for var_name in ["qtend", "stend", "radiation"]:
        print(var_name)
        idx = index[var_name]
        raw_var = data_y[:,idx[0]:idx[1]]
        norm_var = norm_y2[:,idx[0]:idx[1]]
        inv_var = inverse_legacy2[var_name](norm_var)
        #assert(np.allclose(raw_var, inv_var, atol=1e-6, rtol=1e-5))
        assert(np.allclose(raw_var, inv_var))

