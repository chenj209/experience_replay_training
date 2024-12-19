from normalization import normalize_x, normalize_y, inverse_legacy
import numpy as np

if __name__ == "__main__":
    data = np.load("/data/nncam_data/image_set/00100.npz")
    data_x = data["data_x"].astype(np.float64)
    norm_x = normalize_x(data_x)
    data_y = data["data_y"].astype(np.float64)
    norm_y = normalize_y(data_y)
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
        idx = index[var_name]
        raw_var = data_x[:,idx[0]:idx[1]]
        norm_var = norm_x[:,idx[0]:idx[1]]
        inv_var = inverse_legacy[var_name](norm_var)
        assert(np.allclose(raw_var, inv_var))
    for var_name in ["qtend", "stend", "radiation"]:
        idx = index[var_name]
        raw_var = data_y[:,idx[0]:idx[1]]
        norm_var = norm_y[:,idx[0]:idx[1]]
        inv_var = inverse_legacy[var_name](norm_var)
        assert(np.allclose(raw_var, inv_var))

