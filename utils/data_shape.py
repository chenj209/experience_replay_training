import numpy as np

def to_inference_shape(data):
    """
    the raw data is in format of (batch, channels, lat, lon)
    the output data is in shape of (batch*lat*lon, channels)
    """
    data = np.transpose(data, (0, 2, 3, 1))
    data = np.reshape(data, (-1, data.shape[-1]))
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
    test_data = np.random.rand(100, 30, 96, 144)
    test_data = to_inference_shape(test_data)
    test_data = inverse_to_inference_shape(test_data)
    assert test_data.shape == (100, 30, 96, 144)