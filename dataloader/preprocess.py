import sys
import os
import math
import numpy as np

sys.path.append(os.path.join(sys.path[0], "..", "utils"))
from normalization import normalize_data_var_names2
from data_shape import to_inference_shape2
from logger import debug_print
from dataloader_utils import get_index_from_colnames

DEBUG = True

class FlattenSpatialTransform:
    def __call__(self, sample):
        """
        Takes in data_x and data_y and flattens the spatial dimensions.

        data_x: input data, shape (n_features, lat, lon)
        data_y: output data, shape (n_features, lat, lon)

        returns:
            data_x: input data, shape (lat*lon, n_features)
        """
        data_x, data_y = sample[:2]
        data_x = data_x.reshape(data_x.shape[0], -1).T
        data_y = data_y.reshape(data_y.shape[0], -1).T
        return data_x, data_y, *sample[2:]

class RegionMaskTransform:
    def __init__(self, mask, include_raw=False):
        """
        Takes a mask and applies it to the data.

        mask: mask to apply to the data, shape (lat, lon)
        include_raw: whether to include the raw input data in the output as the third element
        """
        self.mask = mask
        self.include_raw = include_raw

    def __call__(self, sample):
        """
        Takes in data_x and data_y and masks the data to the region of interest.

        data_x: input data, shape (n_features, lat, lon)
        data_y: output data, shape (n_features, lat, lon)
        """
        # debug_print(f"RegionMaskTransform: sample {len(sample)}", DEBUG)
        data_x, data_y = sample[:2]
        if self.include_raw:
            return data_x[:, (self.mask).astype(bool)], \
                data_y[:, (self.mask).astype(bool)], \
                data_x[:, (self.mask).astype(bool)], *sample[2:]

        return data_x[:, (self.mask).astype(bool)], \
            data_y[:, (self.mask).astype(bool)], *sample[2:]

def get_min_max_coords(mask, pad):
    min_x = np.min(np.where(mask)[0])
    max_x = np.max(np.where(mask)[0])
    min_y = np.min(np.where(mask)[1])
    max_y = np.max(np.where(mask)[1])
    # wid_x = math.ceil((max_x - min_x)/2)*2
    # wid_y = math.floor((max_y - min_y)/2)*2
    # return int(min_x-pad), int(min_x+wid_x+pad), \
    #     int(min_y-pad), int(min_y+wid_y+pad)
    return min_x-pad, max_x+pad, min_y-pad, max_y+pad

class RectRegionMaskTransform:
    def __init__(self, mask, pad=2, include_raw=False):
        """
        Takes a mask, find a rectangle region box for the mask and applies it to the data.

        mask: mask to apply to the data
        include_raw: whether to include the raw input data in the output as the third element
        """
        self.mask = mask
        self.min_x, self.max_x, self.min_y, self.max_y = get_min_max_coords(mask, pad)
        debug_print(f"min_x: {self.min_x}, max_x: {self.max_x}, \
                    min_y: {self.min_y}, max_y: {self.max_y}", DEBUG)
        self.include_raw = include_raw


    def __call__(self, sample):
        """
        Takes in data_x and data_y and masks the data to the region of interest.

        data_x: input data, shape (n_features, lat, lon)
        data_y: output data, shape (n_features, lat, lon)
        """
        data_x, data_y = sample[:2]
        if self.include_raw:
            return data_x[:, self.min_x:self.max_x, self.min_y:self.max_y], \
                data_y[:, self.min_x:self.max_x, self.min_y:self.max_y], \
                data_x[:, self.min_x:self.max_x, self.min_y:self.max_y], *sample[2:]

        return data_x[:, self.min_x:self.max_x, self.min_y:self.max_y], \
            data_y[:, self.min_x:self.max_x, self.min_y:self.max_y], *sample[2:]


class StandardizeTransform:
    def __init__(
            self, 
            data_mean, 
            data_std, 
            data_cols_x,
            data_cols_y,
            col_names,
            multistep=0,
            normalize_input=True,
            normalize_output=True
            ):
        """
        Takes a set of input and output from SPCAM data and standardizes it.

        data_mean: dictionary of mean values for each column
        data_std: dictionary of standard deviation values for each column
        data_cols_x: list of columns to standardize in order of appearance in the 
        selected data x
        data_cols_y: list of columns to standardize in order of appearance in the 
        selected data x
        col_names: raw column names of the data in order of appearance in the raw data
                    (loaded from col_names.txt)
        input_normalized: boolean, whether to normalize the input data
        normalize_input: boolean, whether to normalize the input data
        normalize_output: boolean, whether to normalize the output data
        """
        self.data_mean = data_mean
        self.data_std = data_std
        self.data_cols_x = data_cols_x
        self.data_cols_y = data_cols_y
        self.col_names = col_names
        self.multistep = multistep
        self.normalize_input = normalize_input
        self.normalize_output = normalize_output

    def __call__(self, sample):
        """
        Takes in data_x and data_y and standardizes it.

        data_x: input data, shape (n_samples, n_features, lat, lon)
        data_y: output data, shape (n_samples, n_features, lat, lon)
        """
        # debug_print("StandardizeTransform: sample {}".format(len(sample)), DEBUG)
        err_header = "StandardizeTransform:"
        if len(sample) > 2:
            # the last element is the filenames
            filenames = sample[-1]
            # check if the filenames are in the correct format
            if type(filenames[0]) == str:
                err_header = f"{err_header} {filenames}: "

        data_x, data_y = sample[:2]
        x = data_x.copy()
        # check multistep shape here
        target_shape = 0
        for col in self.data_cols_x:
            start_idx, end_idx = get_index_from_colnames(self.col_names, col)
            target_shape += end_idx - start_idx
        # debug_print("StandardizeTransform: x shape: {}".format(x.shape), DEBUG)
        assert x.shape[0] == target_shape, f"Input data shape does not match \
            the expected shape {target_shape}"

        if self.normalize_input:
            x = normalize_data_var_names2(x, self.data_cols_x, self.col_names, 
                                 self.data_mean, self.data_std, err_header=err_header)
            for col in self.data_cols_x:
                start_idx, end_idx = get_index_from_colnames(self.col_names, col)

        y = data_y.copy()
        if self.normalize_output:
            y = normalize_data_var_names2(y, self.data_cols_y, self.col_names, 
                                    self.data_mean, self.data_std, err_header=err_header)
        return x, y, *sample[2:]
