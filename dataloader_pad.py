from torch.utils import data
import os
import numpy as np
import time
import glob

# norm_vec = np.load('norm_vec.npz')
# channel_max_x = norm_vec['channel_max_x']
# channel_min_x = norm_vec['channel_min_x']
# channel_max_y = norm_vec['channel_max_y']
# channel_min_y = norm_vec['channel_min_y']




def normalization(data_x, data_y):
    x = data_x.copy()
    y = data_y.copy()
    

    
    x[:, 0:30,:,:]  = (x[:, 0:30,:,:] - 0)   /    (0.0238) * 2 - 1
    x[:,30:60,:,:]  = (x[:,30:60,:,:] - 159) / (323 - 159) * 2 - 1
    x[:,60:90,:,: ] = (x[:,60:90,:,:] + 2.13e-6) / (2.13e-6*2) * 2 - 1
    x[:,90:120,:,:] = (x[:,90:120,:,:] + 3.89e-3) / (3.89e-3*2) * 2 - 1
    x[:,120,:,:]    = (x[:,120,:,:] - 0)     / (1412 - 0) 
    x[:,121,:,:]    = (x[:,121,:,:] - 59928) / (105782 - 59928) 

    # output data (target)
    y[:, 0:30,:,:] = (y[:, 0:30,:,:] + 3.11e-6) / (3.11e-6*2) * 2 - 1
    y[:,30:60,:,:] = (y[:,30:60,:,:] + 3.63)    / (3.63*2) * 2    - 1
    y[:,60:61,:,:]    =  y[:,60:61,:,:] / (2.12e-6) * 2 - 1
    
    y[:,61:62,:,:]    = (y[:,61:62,:,:] - 0) / (1412 - 0)
    y[:,62:63,:,:]    = (y[:,62:63,:,:] - 0) / (1412 - 0)
    y[:,63:64,:,:]    = (y[:,63:64,:,:] - 0) / (1412 - 0)
    y[:,64:65,:,:]    = (y[:,64:65,:,:] - 0) / (1412 - 0)
    y[:,65:66,:,:]    = (y[:,65:66,:,:] - 0) / (1412 - 0)    
    
    
    data_x_norm, data_y_norm = x,y
    
    return data_x_norm, data_y_norm


# norm_vec = np.load('norm_vec_dede_mean_std.npz')
# channel_mean_x = norm_vec['channel_mean_x']
# channel_std_x = norm_vec['channel_std_x']
# channel_mean_y = norm_vec['channel_mean_y']
# channel_std_y = norm_vec['channel_std_y']

# def normalization(data_x, data_y):
#     data_x_norm = (data_x - channel_mean_x) / (channel_std_x+0.0000001)
#     data_y_norm = (data_y - channel_mean_y) / (channel_std_y+0.0000001)
#     return data_x_norm, data_y_norm

class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, datadir, is_train, train62, noise_std = 0, is_test=False, sample=None):
        """
        Param:
            datadir: the directory of the dataset
            is_train: if True, use the training set, else use the validation set
            noise_std: if >0, add gaussian noise to the data
            is_test: if True, is_train is False and return all the data as test
                    in addition, the output data (target) is not normalized
        """
        
        #################### 屏蔽掉一些可能存在异常的数据集 ##############################        
        all_files = glob.glob(datadir + "/*.npz") 
        if sample is not None:
            all_files = all_files[::sample]

        print('org file num:', len(all_files))
        for i in range(17507,17530):
            for file_name in all_files:
                if str(i) in file_name:
                    all_files.remove(file_name)

        print('after file num:', len(all_files))

        for i in ['00001', '08690', '17522', '26210']:
            for file_name in all_files:
                if i in file_name:
                    all_files.remove(file_name)

        print('hahahahahah after file num:', len(all_files))

        if is_test:
            self.files = all_files
        else: 
            test_files = all_files[-int(0.1*len(all_files)):]
            train_files = [file_name for file_name in all_files 
                           if file_name not in test_files]
            # test_files = train_files
            print('train files: {} test files: {}'
                  .format(len(train_files), len(test_files)))
            
            if is_train:
                self.files = train_files
            else:
                self.files = test_files

        self.size = len(self.files)
        self.noise_std = noise_std
        self.is_train = is_train
        self.train62 = train62
        self.is_test = is_test

        if self.train62:
            print('the input size is 62')
        else:
            print('the input size is 122')

    def __len__(self):
        'Denotes the total number of samples'
        return self.size

    def __getitem__(self, index):
        'Generates one sample of data'
        _file = np.load(self.files[index])
        tx = _file["data_x"]
        ty = _file["data_y"]
        x, y = normalization(tx, ty)

        # print(x.shape, y.shape)
        ### padding
        pad_size = 20
        left = x[:, :, :, :pad_size] #正数20个经度
        right = x[:, :, :, -pad_size:] #倒数20个经度
        x = np.concatenate((right, x, left), axis=3)

        x = x[0]
        y = y[0]

        if self.train62:
                x = x[list(range(60))+[120,121]]

        if self.is_train and self.noise_std>0:
            noise_x = np.random.randn(x.shape[0], x.shape[1], x.shape[2]) * self.noise_std
            noise_y = np.random.randn(y.shape[0], y.shape[1], y.shape[2]) * self.noise_std
            x = x + noise_x
            y = y + noise_y

        if self.is_test:
            return x, ty, tx
        return x, y


if __name__ == '__main__':
    training_set = Dataset(datadir='/cust_users/alg/data/sampled-image_set/sampling_at_0.1', is_train=True, train62=False, noise_std=0)
    trainloader = data.DataLoader(training_set, shuffle=True, batch_size=1, num_workers=4)
    for idx, batch in enumerate(trainloader):
        x, y = batch
        if idx == 0:
            np.save('checkcode_y', y.numpy())
        print(idx, x.size(), y.size())
