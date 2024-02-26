import numpy as np
import torch

def to_inference_shape2(data):
    """
    the raw data is in format of (channels, lat, lon)
    the output data is in shape of (lat*lon, channels)
    """
    data = np.transpose(data, (1, 2, 0))
    data = np.reshape(data, (-1, data.shape[-1]))
    return data

def to_inference_shape(data):
    """
    the raw data is in format of (batch, channels, lat, lon)
    the output data is in shape of (batch, lat*lon, channels)
    """
    data = np.transpose(data, (0, 2, 3, 1))
    data = np.reshape(data, (data.shape[0], -1, data.shape[-1]))
    return data

def to_inference_shape_torch2(data):
    """
    the raw data is in format of (batch, channels, lat, lon)
    the output data is in shape of (batch, lat*lon, channels)
    """
    # complete this function
    data = data.permute(0,2,3,1)

    data = torch.reshape(data, (data.shape[0], -1, data.shape[-1]))
    return data

# write a torch version of to_inference_shape
def to_inference_shape_torch(data):
    """
    the raw data is in format of (batch, channels, lat, lon)
    the output data is in shape of (batch, lat*lon, channels)
    """
    # complete this function
    data = data.permute(0,2,3,1)

    data = torch.reshape(data, (data.shape[0], -1, data.shape[-1]))
    return data

def inverse_to_inference_shape(data):
    """
    the raw data is in format of (batch*lat*lon, channels)
    the output data is in shape of (batch, channels, lat, lon)
    """
    data = np.reshape(data, (-1, 96, 144, data.shape[-1]))
    data = np.transpose(data, (0, 3, 1, 2))
    return data


if __name__ == "__main__":
    # generate a test to validate the correctness of the functions
    test_data_raw = np.random.rand(1, 1, 96, 144)
    test_data = to_inference_shape(test_data_raw)
    print(test_data.shape)
    test_data_torch = to_inference_shape_torch(torch.from_numpy(test_data_raw))
    assert(np.allclose(test_data, test_data_torch.numpy()))
    test_data = inverse_to_inference_shape(test_data)
    assert test_data.shape == (1, 1, 96, 144)