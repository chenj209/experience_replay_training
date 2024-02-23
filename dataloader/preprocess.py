import sys
import os

sys.path.append(os.path.join(sys.path[0], "..", "utils"))
from normalization import normalize_data_var_names
from data_shape import to_inference_shape2
from logger import debug_print

DEBUG = True

class FlattenSpatialTransform:
    def __call__(self, sample):
        """
        Takes in data_x and data_y and flattens the spatial dimensions.

        data_x: input data, shape (n_samples, n_features, lat, lon)
        data_y: output data, shape (n_samples, n_features, lat, lon)
        """
        data_x, data_y = sample[:2]
        return to_inference_shape2(data_x), to_inference_shape2(data_y), *sample[2:]

class StandardizeTransform:
    def __init__(
            self, 
            data_mean, 
            data_std, 
            data_cols_x,
            data_cols_y,
            col_names,
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
        self.normalize_input = normalize_input
        self.normalize_output = normalize_output

    def __call__(self, sample):
        """
        Takes in data_x and data_y and standardizes it.

        data_x: input data, shape (n_samples, n_features, lat, lon)
        data_y: output data, shape (n_samples, n_features, lat, lon)
        """
        data_x, data_y = sample[:2]
        x = data_x.copy()
        debug_print("StandardizeTransform: x shape: {}".format(x.shape), DEBUG)
        if self.normalize_input:
            x = normalize_data_var_names(x, self.data_cols_x, self.col_names, 
                                 self.data_mean, self.data_std)
            x_mean = x.mean(axis=(1,2))
            if x_mean.max() > 2 or x_mean.min() < -2:
                print("Warning: Input data is not normalized correctly")
        y = data_y.copy()
        if self.normalize_output:
            y = normalize_data_var_names(y, self.data_cols_y, self.col_names, 
                                    self.data_mean, self.data_std)
            y_mean = y.mean(axis=(1,2))
            if y_mean.max() > 2 or y_mean.min() < -2:
                print("Warning: Output data is not normalized correctly")
        return x, y, *sample[2:]
