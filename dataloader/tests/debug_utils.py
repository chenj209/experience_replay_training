import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from dataloader_utils import get_index_from_colnames

def print_mean_std_by_var(data, target_vars, col_names, delevelwise=False, reduce_lvl=0):
    """
    Print the mean and standard deviation of the data for each variable in target_vars.

    data: shape (n_samples, n_features, ...)
    target_vars: list of variable names
    col_names: list of column names in order of appearance in the raw data
    """
    if delevelwise:
        _target_vars = []
        prev_col = None
        for col in target_vars:
            delevelwise_var = col.split("_lev")[0]
            if delevelwise_var != prev_col:
                _target_vars.append(delevelwise_var)
            prev_col = delevelwise_var
        target_vars = _target_vars
    cur_idx = 0
    for col in target_vars:
        start_idx, end_idx = get_index_from_colnames(col_names, col)
        data_range = end_idx - start_idx
        if data_range == 30:
            data_range = data_range - reduce_lvl
        print(col, data[:,cur_idx:cur_idx+data_range].mean(), \
              data[:,cur_idx:cur_idx+data_range].std())
        cur_idx += data_range