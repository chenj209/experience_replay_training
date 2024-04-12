import numpy as np
import os
import sys
sys.path.append(os.path.join(sys.path[0], "..", "consts"))
import phys_consts

PHYS_CONST_FILENAME = "phys_consts.npz"
NC_SAMPLE_PATH = "/Users/jiandachen/Projects/NNCAM/nn_metrics/10year_manual_rh.cam.h0.2009-12.nc"

def get_phys_consts(nc_sample_path):
    if os.path.exists(PHYS_CONST_FILENAME):
        nc_sample = np.load(PHYS_CONST_FILENAME)
    else:
        nc_sample = nc.Dataset(nc_sample_path)
    hybi = nc_sample["hybi"]
    hyai = nc_sample["hyai"]
    hybm = nc_sample["hybm"]
    hyam = nc_sample["hyam"]

    pconsts = {"hybi": hybi, "hyai": hyai, "hybm": hybm, "hyam": hyam}
    if not os.path.exists(PHYS_CONST_FILENAME):
        np.savez(PHYS_CONST_FILENAME, **pconsts)
    return pconsts

def get_thickness_from_ps_1d(ps, hybi, hyai):
    """
    ps: (N, lon, lat): shape N samples
    """
    hybi_diff = np.diff(hybi)
    hyai_diff = np.diff(hyai)
    thick = ps[:, np.newaxis] \
        * hybi_diff[np.newaxis, :]
    thick += hyai_diff[np.newaxis, :] * 100000
    thick /= phys_consts.GRAVIT
    #t_shape = thick.shape
    #return thick.reshape(t_shape)
    return thick

def get_thickness_from_ps_2d(ps, hybi, hyai):
    """
    ps: (N, lon, lat): shape N samples
    """
    hybi_diff = np.diff(hybi)
    hyai_diff = np.diff(hyai)
    thick = ps[:, np.newaxis, :, :] \
        * hybi_diff[np.newaxis, :, np.newaxis, np.newaxis]
    thick += hyai_diff[np.newaxis, :, np.newaxis, np.newaxis] * 100000
    thick /= phys_consts.GRAVIT
    #t_shape = thick.shape
    #return thick.reshape(t_shape)
    return thick

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

def Regression_Metrics(y_true, y_pred):
    
    var  = np.var(y_true)
    std  = np.std(y_true)
    
    mse  = np.mean((y_true-y_pred)**2)
    rmse = mse**0.5
       
    mae    = np.mean(np.absolute(y_pred-y_true))  # Mean absolute error
    me    = np.mean(y_pred-y_true)  # Mean absolute error
    max_ae = np.max(np.absolute(y_pred-y_true))   # Max absolute error
    
    bias = np.mean(y_pred-y_true)
    r2   = 1 - mse/var 
    
    return {
        "var": float(var), 
        "std": float(std), 
        "mse": float(mse), 
        "rmse": float(rmse), 
        "mae": float(mae), 
        "max_ae": float(max_ae), 
        "bias": float(bias), 
        "r2": float(r2), 
        "me": float(me)
    }

def Regression_Metrics_axis(y_true, y_pred, axis):
    
    var  = np.var(y_true,axis=axis)
    var_pred  = np.var(y_pred,axis=axis)
    std  = np.std(y_true,axis=axis)
    std_pred = np.std(y_pred,axis=axis)
    mean = np.mean(y_true,axis=axis)
    mean_pred = np.mean(y_pred,axis=axis)
    
    mse  = np.mean((y_true-y_pred)**2, axis=axis)
    rmse = np.sqrt(mse)
       
    mae    = np.mean(np.absolute(y_pred-y_true),axis=axis)  # Mean absolute error
    me    = np.mean(y_pred-y_true, axis=axis)  # Mean absolute error
    max_ae = np.max(np.absolute(y_pred-y_true), axis=axis)   # Max absolute error
    
    bias = np.mean(y_pred-y_true, axis=axis)
    r2   = 1 - mse/var 
    return {
        "var": var, 
        "var_pred": var_pred, 
        "std": std, 
        "std_pred": std_pred, 
        "mean": mean, 
        "mean_pred": mean_pred, 
        "mse": mse, 
        "rmse": rmse, 
        "mae": mae, 
        "max_ae": max_ae, 
        "bias": bias, 
        "r2": r2, 
        "me": me 
    }

    

def transform_datashape(tx):
    x = tx.copy()
    x = np.transpose(x, (0, 2, 3, 1))
    x = x.reshape(-1, x.shape[-1])
    return x

def reverse_operations(tx_reshaped, original_shape):
    """
    Reverses the operations of transpose and reshape on a tensor.
    
    Args:
    - tx_reshaped: The reshaped tensor.
    - original_shape: The original shape of the tensor before the operations were applied.
    
    Returns:
    - tx_original: The tensor in its original shape and order.
    """
    
    # Calculate the number of samples from the reshaped tensor
    N_sample = tx_reshaped.shape[0] // (original_shape[1] * original_shape[2])
    
    # Reshape the tensor to its shape after the transpose operation but before the reshape operation
    tx_transposed = tx_reshaped.reshape(N_sample, original_shape[1], original_shape[2], original_shape[0])
    
    # Transpose the tensor to its original shape and order
    tx_original = np.transpose(tx_transposed, (0, 3, 1, 2))
    
    return tx_original

def report_stend_spatial(y_gt, stend):
    y_gt = y_gt[:,30:60]
    original_shape = (30, 96, 144)
    y_gt = reverse_operations(y_gt, original_shape)
    stend = reverse_operations(stend, original_shape)
    return Regression_Metrics_axis(y_gt, stend, axis=0)

def report_stend_vert(y_gt, stend):
    y_gt = y_gt[:,30:60]
    original_shape = (30, 96, 144)
    y_gt = reverse_operations(y_gt, original_shape)
    stend = reverse_operations(stend, original_shape)
    return Regression_Metrics_axis(y_gt, stend, axis=(0,2,3))

def report_stend(y_gt, stend, mask=None):
    if mask is not None:
        out = Regression_Metrics(y_gt[:,30:60][mask], stend[mask])
    else:
        out = Regression_Metrics(y_gt[:,30:60], stend)
    stend_metrics = '{}\t{:.4}\t{:.4}\t{:.4}({:.4%})\t{:.4}\t{:.4} \
    \t{:.4}({:.4%})\t'.format(
        50, out["r2"], out["mse"], out["rmse"], out["rmse"]/out["std"], 
        out["mae"], out["max_ae"], out["bias"], out["bias"]/out["std"])
    stend_log = ""
    stend_log += "stend metrics:\n"
    stend_log += "epoch\tr2\tmse\trmse\t\tmae\tmax_ae\tbias\n"
    stend_log += stend_metrics
    stend_log += "\n"
    print(stend_log)
    return out

def report_qtend_spatial(y_gt, qtend):
    y_gt = y_gt[:,0:30]
    original_shape = (30, 96, 144)
    y_gt = reverse_operations(y_gt, original_shape)
    qtend = reverse_operations(qtend, original_shape)
    return Regression_Metrics_axis(y_gt, qtend, axis=0)

def report_qtend_vert_quantile(y_gt, qtend, quantile):
    tail = (1-quantile)/2
    results = []
    for i in range(y_gt.shape[1]):
        # qtend_qt = np.quantile(qtend[:,i], [tail, 1-tail])
        y_gt_qt = np.quantile(y_gt[:,i], [tail, 1-tail])
        qtend_qt_data = qtend[(qtend[:,i] >= y_gt_qt[0]) & (qtend[:,i] <= y_gt_qt[1])]
        y_gt_qt_data = y_gt[(y_gt[:,i] >= y_gt_qt[0]) & (y_gt[:,i] <= y_gt_qt[1])]
        results.append(Regression_Metrics(y_gt_qt_data, qtend_qt_data))
    merged_results = {}
    for key in results[0].keys():
        merged_results[key] = [result[key] for result in results]
    return merged_results

def report_qtend_vert(y_gt, qtend):
    y_gt = y_gt[:,0:30]
    #original_shape = (30, 96, 144)
    #y_gt = reverse_operations(y_gt, original_shape)
    #qtend = reverse_operations(qtend, original_shape)
    #print(qtend.shape)
    return Regression_Metrics_axis(y_gt, qtend, axis=0)

def report_qtend(y_gt, qtend, mask=None):
    if mask is not None:
        out = Regression_Metrics(y_gt[:,0:30][mask], qtend[mask])
    else:
        out = Regression_Metrics(y_gt[:,0:30], qtend)
    qtend_metrics = '{}\t{:.4}\t{:.4}\t{:.4}({:.4%})\t{:.4}\t{:.4} \
    \t{:.4}({:.4%})\t'.format(
        50, out["r2"], out["mse"], out["rmse"], out["rmse"]/out["std"], 
        out["mae"], out["max_ae"], out["bias"], out["bias"]/out["std"])
    qtend_log = ""
    qtend_log += "qtend metrics:\n"
    qtend_log += "epoch\tr2\tmse\trmse\t\tmae\tmax_ae\tbias\n"
    qtend_log += qtend_metrics
    qtend_log += "\n"
    print(qtend_log)
    return out

def report_rad_prog(y_gt, rad_prog):
    out = Regression_Metrics(y_gt[:,61:66], rad_prog)
    rad_prog = '{}\t{:.4}\t{:.4}\t{:.4}({:.4%})\t{:.4}\t{:.4} \
    \t{:.4}({:.4%})\t'.format(
        50, out["r2"], out["mse"], out["rmse"], out["rmse"]/out["std"], 
        out["mae"], out["max_ae"], out["bias"], out["bias"]/out["std"])
    rad_log = ""
    rad_log += "rad metrics:\n"
    rad_log += "epoch\tr2\tmse\trmse\t\tmae\tmax_ae\tbias\n"
    rad_log += rad_prog
    rad_log += "\n"
    print(rad_log)
    return out

def report_rad_prog_individual(y_gt, rad_pred):
    rad_vars = [
     "soll",
     "sols", 
     "solsd",
     "solld",
     "fsds"
    ]
    out = {}
    for i in range(len(rad_vars)):
        pout = Regression_Metrics(y_gt[:,61+i:61+i+1], rad_pred[:,i:i+1])
        rad_prog = '{}\t{:.4}\t{:.4}\t{:.4}({:.4%})\t{:.4}\t{:.4} \
        \t{:.4}({:.4%})\t'.format(
            50, pout["r2"], pout["mse"], pout["rmse"], pout["rmse"]/pout["std"], 
            pout["mae"], pout["max_ae"], pout["bias"], pout["bias"]/pout["std"])
        rad_log = ""
        rad_log += f"rad {rad_vars[i]} metrics:\n"
        rad_log += "epoch\tr2\tmse\trmse\t\tmae\tmax_ae\tbias\n"
        rad_log += rad_prog
        rad_log += "\n"
        #print(rad_log)
        out[rad_vars[i]] = pout
    return out

if __name__ == "__main__":
    import netCDF4 as nc
    get_phys_consts(NC_SAMPLE_PATH)
