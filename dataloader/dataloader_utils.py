import re
def idx_to_filename(idx):
  return str(idx).rjust(5,'0') + '.npy'

def filename_to_idx(filename):
    data_pattern = ".*(\d{5})\.npy"
    m = re.search(data_pattern, filename)
    if m is None:
        return -1
    else:
        return int(m.group(1))

def col_name_cmp(var_name, col_name):
    pattern = f"^{var_name}(_lev\d+)?$"
    # print(pattern)
    return re.match(pattern, col_name) is not None

def get_index_from_colnames(col_names, var_name):
    start_idx = -1
    end_idx = -1
    for i, col_name in enumerate(col_names):
        if col_name_cmp(var_name, col_name) and start_idx == -1:
            start_idx = i
        if col_name_cmp(var_name, col_name):
            end_idx = i
        if end_idx != -1 and not col_name_cmp(var_name, col_name):
            return start_idx, end_idx+1
    return start_idx, end_idx+1


def gen_multistep_col_indices(col_names, prev_ex_vars, col_names_x, col_names_y, multistep):
    prev_input_indices = []
    curr_input_indices = []
    for cn in col_names_x:
        start_idx, end_idx = get_index_from_colnames(col_names, cn)
        prev_input_indices.extend(list(range(start_idx, end_idx)))
        curr_input_indices.extend(list(range(start_idx, end_idx)))
    if prev_ex_vars is not None:
        for cn in prev_ex_vars:
            start_idx, end_idx = get_index_from_colnames(col_names, cn)
            prev_input_indices.extend(list(range(start_idx, end_idx)))
    output_indices = []
    for cn in col_names_y:
        start_idx, end_idx = get_index_from_colnames(col_names, cn)
        output_indices.extend(list(range(start_idx, end_idx)))
    input_indices = curr_input_indices
    return input_indices, prev_input_indices, output_indices