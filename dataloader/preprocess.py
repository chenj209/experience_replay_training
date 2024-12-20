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
        if sample is None:
            return None
        data_x, data_y = sample[:2]
        # print("DEBUG1210: data_x.shape", data_x.shape)
        data_x = data_x.reshape(data_x.shape[0], -1).T
        data_y = data_y.reshape(data_y.shape[0], -1).T
        res = [data_x, data_y]
        res.extend(sample[2:])
        return res

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
        if sample is None:
            return None
        # debug_print(f"RegionMaskTransform: sample {len(sample)}", DEBUG)
        data_x, data_y = sample[:2]
        if self.include_raw:
            res = [ 
            data_x[:, (self.mask).astype(bool)], \
            data_y[:, (self.mask).astype(bool)], \
            data_x[:, (self.mask).astype(bool)]]
        else:
            res = [data_x[:, (self.mask).astype(bool)], \
            data_y[:, (self.mask).astype(bool)]] 
        res.extend(sample[2:])
        return res


def get_min_max_coords(mask, pad):
    min_x = np.min(np.where(mask)[0])
    max_x = np.max(np.where(mask)[0])
    min_y = np.min(np.where(mask)[1])
    max_y = np.max(np.where(mask)[1])
    # wid_x = math.ceil((max_x - min_x)/2)*2
    # wid_y = math.floor((max_y - min_y)/2)*2
    # return int(min_x-pad), int(min_x+wid_x+pad), \
    #     int(min_y-pad), int(min_y+wid_y+pad)
    min_x, max_x, min_y, max_y =  min_x-pad, max_x+pad+1, min_y-pad, max_y+pad+1
    min_x = max(0, min_x)
    max_x = min(mask.shape[0], max_x)
    min_y = max(0, min_y)
    max_y = min(mask.shape[1], max_y)
    return int(min_x), int(max_x), int(min_y), int(max_y)

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
        if sample is None:
            return None
        data_x, data_y = sample[:2]
        if self.include_raw:
            res = [data_x[:, self.min_x:self.max_x, self.min_y:self.max_y], \
                data_y[:, self.min_x:self.max_x, self.min_y:self.max_y], \
                data_x[:, self.min_x:self.max_x, self.min_y:self.max_y]]
        else:
            res = [data_x[:, self.min_x:self.max_x, self.min_y:self.max_y], \
            data_y[:, self.min_x:self.max_x, self.min_y:self.max_y]]
        res.extend(sample[2:])
        return res

class MinMaxTransformLegacy2stepNoFSDS:
    def __init__(self, include_raw=False):
        """
        Takes a mask and applies it to the data.

        mask: mask to apply to the data, shape (lat, lon)
        include_raw: whether to include the raw input data in the output as the third element
        """
        self.include_raw = include_raw
    
    def __call__(self, sample):
        data_x_raw, data_y_raw = sample[:2]
        data_x = data_x_raw.copy()
        data_y = data_y_raw.copy()
        assert(data_x.shape[0] == 308)
        assert(data_y.shape[0] == 64)
        #data_x[0:30] = (data_x[0:30] - 0)   / (0.0238) * 2 - 1 # Q
        #data_x[30:60] = (data_x[30:60] - 159) / (323 - 159) * 2 - 1 # T
        #data_x[60:90] = (data_x[60:90] + 2.13e-6) / (2.13e-6*2) * 2 - 1 # dqls
        #data_x[90:120] = (data_x[90:120] + 3.89e-3) / (3.89e-3*2) * 2 - 1 # dTls
        #data_x[120]    = (data_x[120] - 0)     / (1412 - 0) # solin
        #data_x[121]    = (data_x[121] - 59928) / (105782 - 59928) # ps
        data_x[:, 0:30,:,:]  = (data_x[:, 0:30,:,:] - 0) /np.float64(0.0119) - 1
        data_x[:,30:60,:,:]  = (data_x[:,30:60,:,:] - np.float64(159)) / np.float64(82) - 1
        data_x[:,60:90,:,: ] = (data_x[:,60:90,:,:]) / np.float64(2.13e-6)
        data_x[:,90:120,:,:] = data_x[:,90:120,:,:]/np.float64(3.89e-3)
        data_x[:,120,:,:]    = (data_x[:,120,:,:] - 0)/ np.float64(1412 - 0)
        data_x[:,121,:,:]    = (data_x[:,121,:,:] - np.float64(59928)) / np.float64(105782 - 59928)
        data_x[122+0:122+30] = data_x[122+0:122+30] / np.float64(3.11e-6)
        data_x[122+30:122+60] = data_x[122+30:122+60] / np.float64(3.63)
        # data_x[60:61]    =  data_x[60:61,:,:] / (2.12e-6) * 2 - 1
        
        data_x[122+60:122+61]    = (data_x[122+60:122+61] - 0) / np.float64(1412 - 0)
        data_x[122+61:122+62]    = (data_x[122+61:122+62] - 0) / np.float64(1412 - 0)
        data_x[122+62:122+63]    = (data_x[122+62:122+63] - 0) / np.float64(1412 - 0)
        data_x[122+63:122+64]    = (data_x[122+63:122+64] - 0) / np.float64(1412 - 0)
        data_x[186:186+30] = data_x[186:186+30] / np.float64(0.0119) -1  # Q
        data_x[186+30:186+60] = (data_x[186+30:186+60] - np.float64(159)) / np.float64(82) - 1 # T
        data_x[186+60:186+90] = data_x[186+60:186+90] / np.float64(2.13e-6) # dqls
        data_x[186+90:186+120] = data_x[186+90:186+120] / np.float64(3.89e-3) # dTls
        data_x[186+120]    = (data_x[186+120] - 0)     / np.float64(1412 - 0) # solin
        data_x[186+121]    = (data_x[186+121] - np.float64(59928)) / np.float64(105782 - 59928) # ps


        data_y[0:30] = data_y[0:30] / np.float64(3.11e-6)
        data_y[30:60] = data_y[30:60] / np.float64(3.63)
        # data_y[60:61]    =  data_y[60:61,:,:] / (2.12e-6) * 2 - 1
        
        data_y[60:61]    = (data_y[60:61] - 0) / np.float64(1412 - 0)
        data_y[61:62]    = (data_y[61:62] - 0) / np.float64(1412 - 0)
        data_y[62:63]    = (data_y[62:63] - 0) / np.float64(1412 - 0)
        data_y[63:64]    = (data_y[63:64] - 0) / np.float64(1412 - 0)
        # data_y[64:65]    = (data_y[64:65] - 0) / (1412 - 0)    
        

        if self.include_raw:
            res = [data_x, data_y, data_x_raw, data_y_raw]
        else:
            res = [data_x, data_y]
        res.extend(sample[2:])
        return res

class MinMaxTransformLegacy2stepNext:
    def __init__(self, include_raw=False):
        """
        Takes a mask and applies it to the data.

        mask: mask to apply to the data, shape (lat, lon)
        include_raw: whether to include the raw input data in the output as the third element
        """
        self.include_raw = include_raw
    
    def __call__(self, sample):
        data_sets = [[*sample[:2]], [*sample[2:4]]] # current inputs and next inputs
        res = []

        #data_x_raw, data_y_raw = sample[:2]
        for data_x_raw, data_y_raw in data_sets:
            data_x = data_x_raw.copy()
            data_y = data_y_raw.copy()
            #if data_x.shape[0] != 309 or data_y.shape[0] != 65:
            if data_x.shape[0] != 309:
                raise ValueError(f"Unexpected shapes: data_x {data_x.shape}, data_y {data_y.shape}")
            data_x[ 0:30]  = (data_x[ 0:30] - 0) /np.float64(0.0119) - 1
            data_x[30:60]  = (data_x[30:60] - np.float64(159)) / np.float64(82) - 1
            data_x[60:90 ] = (data_x[60:90]) / np.float64(2.13e-6)
            data_x[90:120] = data_x[90:120]/np.float64(3.89e-3)
            data_x[120]    = (data_x[120] - 0)/ np.float64(1412 - 0)
            data_x[121]    = (data_x[121] - np.float64(59928)) / np.float64(105782 - 59928)
            data_x[122+0:122+30] = data_x[122+0:122+30] / np.float64(3.11e-6)
            data_x[122+30:122+60] = data_x[122+30:122+60] / np.float64(3.63)
            # data_x[60:61]    =  data_x[60:61,:,:] / (2.12e-6) * 2 - 1
            
            data_x[122+60:122+61]    = (data_x[122+60:122+61] - 0) / np.float64(1412 - 0)
            data_x[122+61:122+62]    = (data_x[122+61:122+62] - 0) / np.float64(1412 - 0)
            data_x[122+62:122+63]    = (data_x[122+62:122+63] - 0) / np.float64(1412 - 0)
            data_x[122+63:122+64]    = (data_x[122+63:122+64] - 0) / np.float64(1412 - 0)
            data_x[122+64:122+65]    = (data_x[122+64:122+65] - 0) / np.float64(1412 - 0)
            data_x[187:187+30] = data_x[187:187+30] / np.float64(0.0119) -1  # Q
            data_x[187+30:187+60] = (data_x[187+30:187+60] - np.float64(159)) / np.float64(82) - 1 # T
            data_x[187+60:187+90] = data_x[187+60:187+90] / np.float64(2.13e-6) # dqls
            data_x[187+90:187+120] = data_x[187+90:187+120] / np.float64(3.89e-3) # dTls
            data_x[187+120]    = (data_x[187+120] - 0)     / np.float64(1412 - 0) # solin
            data_x[187+121]    = (data_x[187+121] - np.float64(59928)) / np.float64(105782 - 59928) # ps


            data_y[0:30] = data_y[0:30] / np.float64(3.11e-6)
            data_y[30:60] = data_y[30:60] / np.float64(3.63)
            # data_y[60:61]    =  data_y[60:61,:,:] / (2.12e-6) * 2 - 1
            
            data_y[60:61]    = (data_y[60:61] - 0) / np.float64(1412 - 0)
            data_y[61:62]    = (data_y[61:62] - 0) / np.float64(1412 - 0)
            data_y[62:63]    = (data_y[62:63] - 0) / np.float64(1412 - 0)
            data_y[63:64]    = (data_y[63:64] - 0) / np.float64(1412 - 0)
            data_y[64:65]    = (data_y[64:65] - 0) / np.float64(1412 - 0)    
            res.extend([x,y])
        

        res.extend(sample[4:])
        return res

class MinMaxTransformLegacy2step:
    def __init__(self, include_raw=False):
        """
        Takes a mask and applies it to the data.

        mask: mask to apply to the data, shape (lat, lon)
        include_raw: whether to include the raw input data in the output as the third element
        """
        self.include_raw = include_raw
    
    def __call__(self, sample):
        data_x_raw, data_y_raw = sample[:2]
        data_x = data_x_raw.copy()
        data_y = data_y_raw.copy()
        #if data_x.shape[0] != 309 or data_y.shape[0] != 65:
        if data_x.shape[0] != 309:
            raise ValueError(f"Unexpected shapes: data_x {data_x.shape}, data_y {data_y.shape}")
        data_x[ 0:30]  = (data_x[ 0:30] - 0) /np.float64(0.0119) - 1
        data_x[30:60]  = (data_x[30:60] - np.float64(159)) / np.float64(82) - 1
        data_x[60:90 ] = (data_x[60:90]) / np.float64(2.13e-6)
        data_x[90:120] = data_x[90:120]/np.float64(3.89e-3)
        data_x[120]    = (data_x[120] - 0)/ np.float64(1412 - 0)
        data_x[121]    = (data_x[121] - np.float64(59928)) / np.float64(105782 - 59928)
        data_x[122+0:122+30] = data_x[122+0:122+30] / np.float64(3.11e-6)
        data_x[122+30:122+60] = data_x[122+30:122+60] / np.float64(3.63)
        # data_x[60:61]    =  data_x[60:61,:,:] / (2.12e-6) * 2 - 1
        
        data_x[122+60:122+61]    = (data_x[122+60:122+61] - 0) / np.float64(1412 - 0)
        data_x[122+61:122+62]    = (data_x[122+61:122+62] - 0) / np.float64(1412 - 0)
        data_x[122+62:122+63]    = (data_x[122+62:122+63] - 0) / np.float64(1412 - 0)
        data_x[122+63:122+64]    = (data_x[122+63:122+64] - 0) / np.float64(1412 - 0)
        data_x[122+64:122+65]    = (data_x[122+64:122+65] - 0) / np.float64(1412 - 0)
        data_x[187:187+30] = data_x[187:187+30] / np.float64(0.0119) -1  # Q
        data_x[187+30:187+60] = (data_x[187+30:187+60] - np.float64(159)) / np.float64(82) - 1 # T
        data_x[187+60:187+90] = data_x[187+60:187+90] / np.float64(2.13e-6) # dqls
        data_x[187+90:187+120] = data_x[187+90:187+120] / np.float64(3.89e-3) # dTls
        data_x[187+120]    = (data_x[187+120] - 0)     / np.float64(1412 - 0) # solin
        data_x[187+121]    = (data_x[187+121] - np.float64(59928)) / np.float64(105782 - 59928) # ps


        data_y[0:30] = data_y[0:30] / np.float64(3.11e-6)
        data_y[30:60] = data_y[30:60] / np.float64(3.63)
        # data_y[60:61]    =  data_y[60:61,:,:] / (2.12e-6) * 2 - 1
        
        data_y[60:61]    = (data_y[60:61] - 0) / np.float64(1412 - 0)
        data_y[61:62]    = (data_y[61:62] - 0) / np.float64(1412 - 0)
        data_y[62:63]    = (data_y[62:63] - 0) / np.float64(1412 - 0)
        data_y[63:64]    = (data_y[63:64] - 0) / np.float64(1412 - 0)
        data_y[64:65]    = (data_y[64:65] - 0) / np.float64(1412 - 0)    
        

        if self.include_raw:
            res = [data_x, data_y, data_x_raw, data_y_raw]
        else:
            res = [data_x, data_y]
        res.extend(sample[2:])
        return res


class MinMaxTransformLegacy:
    def __init__(self, include_raw=False):
        """
        Takes a mask and applies it to the data.

        mask: mask to apply to the data, shape (lat, lon)
        include_raw: whether to include the raw input data in the output as the third element
        """
        self.include_raw = include_raw
    
    def __call__(self, sample):
        data_x_raw, data_y_raw = sample[:2]
        data_x = data_x_raw.copy()
        data_y = data_y_raw.copy()
        data_x[0:30] = (data_x[0:30] - 0)   / (0.0238) * 2 - 1
        data_x[30:60] = (data_x[30:60] - 159) / (323 - 159) * 2 - 1
        data_x[60:90] = (data_x[60:90] + 2.13e-6) / (2.13e-6*2) * 2 - 1
        data_x[90:120] = (data_x[90:120] + 3.89e-3) / (3.89e-3*2) * 2 - 1
        data_x[120]    = (data_x[120] - 0)     / (1412 - 0)
        data_x[121]    = (data_x[121] - 59928) / (105782 - 59928)

        data_y[0:30] = (data_y[0:30] + 3.11e-6) / (3.11e-6*2) * 2 - 1
        data_y[30:60] = (data_y[30:60] + 3.63)    / (3.63*2) * 2    - 1
        # data_y[60:61]    =  data_y[60:61,:,:] / (2.12e-6) * 2 - 1
        
        data_y[60:61]    = (data_y[60:61] - 0) / (1412 - 0)
        data_y[61:62]    = (data_y[61:62] - 0) / (1412 - 0)
        data_y[62:63]    = (data_y[62:63] - 0) / (1412 - 0)
        data_y[63:64]    = (data_y[63:64] - 0) / (1412 - 0)
        data_y[64:65]    = (data_y[64:65] - 0) / (1412 - 0)    

        if self.include_raw:
            res = [data_x, data_y, data_x_raw, data_y_raw]
        else:
            res = [data_x, data_y]
        res.extend(sample[2:])
        return res

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
            normalize_output=True,
            include_raw=False,
            threshold=4
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
        self.include_raw = include_raw
        self.threshold = threshold

    def __call__(self, sample):
        """
        Takes in data_x and data_y and standardizes it.

        data_x: input data, shape (n_samples, n_features, lat, lon)
        data_y: output data, shape (n_samples, n_features, lat, lon)
        """
        if sample is None:
            return None
        # debug_print("StandardizeTransform: sample {}".format(len(sample)), DEBUG)
        err_header = "StandardizeTransform:"
        if len(sample) > 2:
            filenames = sample[-1]
            # the last element is the filenames
            # check if the filenames are in the correct format
            if type(filenames[0]) == str:
                err_header = f"{err_header} {filenames}: "

        data_x, data_y = sample[:2]
        x = data_x.copy()
        y = data_y.copy()
        # check multistep shape here
        target_shape = 0
        for col in self.data_cols_x:
            start_idx, end_idx = get_index_from_colnames(self.col_names, col)
            target_shape += end_idx - start_idx
        # debug_print("StandardizeTransform: x shape: {}".format(x.shape), DEBUG)
        assert x.shape[0] == target_shape, f"Input data {x.shape} shape does not match \
            the expected shape {target_shape}"
        # print("before norm x shape: ", x.shape)
        if self.normalize_input:
            x = normalize_data_var_names2(x, self.data_cols_x, self.col_names,
                                 self.data_mean, self.data_std, err_header=err_header, threshold=self.threshold)

        if self.normalize_output:
            y = normalize_data_var_names2(y, self.data_cols_y, self.col_names,
                                    self.data_mean, self.data_std, err_header=err_header, threshold=self.threshold)
        if x is None or y is None:
            return None
        if self.include_raw:
            res = [x, y, data_x, data_y]
        else:
            res = [x, y]
        res.extend(sample[2:])
        return res

class StandardizeTransformNext:
    def __init__(
            self,
            data_mean,
            data_std,
            data_cols_x,
            data_cols_y,
            col_names,
            multistep=0,
            normalize_input=True,
            normalize_output=True,
            include_raw=False,
            threshold=4
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
        self.include_raw = include_raw
        if self.include_raw:
            raise NotImplementedError
        self.threshold = threshold

    def __call__(self, sample):
        """
        Takes in data_x and data_y and standardizes it.

        data_x: input data, shape (n_samples, n_features, lat, lon)
        data_y: output data, shape (n_samples, n_features, lat, lon)
        """
        if sample is None:
            return None
        # debug_print("StandardizeTransform: sample {}".format(len(sample)), DEBUG)
        err_header = "StandardizeTransform:"
        if len(sample) > 2:
            filenames = sample[-1]
            # the last element is the filenames
            # check if the filenames are in the correct format
            if type(filenames[0]) == str:
                err_header = f"{err_header} {filenames}: "
        
        data_sets = [[*sample[:2]], [*sample[2:4]]] # current inputs and next inputs
        res = []
        for data_x, data_y in data_sets:
            # data_x, data_y = sample[:2]
            x = data_x.copy()
            y = data_y.copy()
            # check multistep shape here
            target_shape = 0
            for col in self.data_cols_x:
                start_idx, end_idx = get_index_from_colnames(self.col_names, col)
                target_shape += end_idx - start_idx
            # debug_print("StandardizeTransform: x shape: {}".format(x.shape), DEBUG)
            assert x.shape[0] == target_shape, f"Input data {x.shape} shape does not match \
                the expected shape {target_shape}"
            # print("before norm x shape: ", x.shape)
            if self.normalize_input:
                x = normalize_data_var_names2(x, self.data_cols_x, self.col_names,
                                    self.data_mean, self.data_std, err_header=err_header, threshold=self.threshold)

            if self.normalize_output:
                y = normalize_data_var_names2(y, self.data_cols_y, self.col_names,
                                        self.data_mean, self.data_std, err_header=err_header, threshold=self.threshold)
            if x is None or y is None:
                return None
            res.extend([x, y])
            # if self.include_raw:
            #     res.extend([data_x, data_y])
        # if self.include_raw:
        #     res = [x, y, data_x, data_y]
        # else:
            # res = [x, y]
        res.extend(sample[4:])
        return res

class FlattenSpatialTransformNext:
    def __call__(self, sample):
        """
        Takes in data_x and data_y and flattens the spatial dimensions.

        data_x: input data, shape (n_features, lat, lon)
        data_y: output data, shape (n_features, lat, lon)

        returns:
            data_x: input data, shape (lat*lon, n_features)
        """
        if sample is None:
            return None
        data_sets = [[*sample[:2]], [*sample[2:4]]] # current inputs and next inputs
        res = []
        for data_x, data_y in data_sets:
            # data_x, data_y = sample[:2]
            data_x = data_x.reshape(data_x.shape[0], -1).T
            data_y = data_y.reshape(data_y.shape[0], -1).T
            res.extend([data_x, data_y])
        res.extend(sample[4:])
        return res
