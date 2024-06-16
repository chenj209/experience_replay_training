# import numpy as np
# from dataloader_utils import gen_col_indices
# import glob
# from tqdm.autonotebook import tqdm
# DATA_PATH = "../analysis/test_data/"
# OUTPUT_PATH = "./ex_data/"
# col_names = np.loadtxt(DATA_PATH + "col_names.txt", dtype=str)

# EX_VARS = ["UL", "VL", "CLOUD", "CAPE", "FLNS", "FLNT", "SPPRECC", "LWUP"]

# col_indices = gen_col_indices(col_names, EX_VARS)
# all_files = glob.glob(DATA_PATH + "*.npy") 

# for fn in tqdm(all_files):
#     sample_data = np.load(fn, mmap_mode="r")
#     sampled_ex_data = sample_data[col_indices]
#     out_name = fn.split("/")[-1] 
#     np.save(OUTPUT_PATH + out_name, sampled_ex_data)

import numpy as np
from dataloader_utils import gen_col_indices
import glob
from tqdm.autonotebook import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

DATA_PATH = "../analysis/test_data"
OUTPUT_PATH = "./ex_data/"
col_names = np.loadtxt(DATA_PATH + "/col_names.txt", dtype=str)

EX_VARS = ["UL", "VL", "CLOUD", "CAPE", "FLNS", "FLNT", "SPPRECC", "LWUP"]

col_indices = gen_col_indices(col_names, EX_VARS)
all_files = glob.glob(DATA_PATH + "/*.npy")

def process_file(fn):
    sample_data = np.load(fn, mmap_mode="r")
    sampled_ex_data = sample_data[col_indices]
    out_name = fn.split("/")[-1]
    np.save(OUTPUT_PATH + out_name, sampled_ex_data)

# Ensure output directory exists
os.makedirs(OUTPUT_PATH, exist_ok=True)
num_workers = 4
with ThreadPoolExecutor(max_workers=num_workers) as executor:
    futures = {executor.submit(process_file, fn): fn for fn in all_files}
    for future in tqdm(as_completed(futures), total=len(futures)):
        fn = futures[future]
        try:
            future.result()
        except Exception as e:
            print(f"Error processing file {fn}: {e}")