import numpy as np
import torch
import copy
import gc
import copy

class ReplayBuffer():
    def __init__(self, data_loader, inp_shape=[96*144, 65], max_size=300, weighted=False) -> None:
        self.ptr = 0
        self.size = 0
        self.input_shape = inp_shape
        self.max_size = max_size
        self.weighted = weighted
        self.data_loader = data_loader
        self.sample_stride = 1
        self.dataset_max_idx = data_loader.size
        self.buffer = {}
        self.buffer['inp'] = np.zeros((max_size, *inp_shape), dtype=np.float32)
        # inp_buf = np.zeros((max_size, *inp_shape), dtype=np.float32)
        # self.shm = shared_memory.SharedMemory(create=True, size=inp_buf.nbytes)
        # self.shm.unlink()
        # self.buffer['inp'] = np.ndarray(inp_buf.shape, dtype=inp_buf.dtype, buffer=self.shm.buf)
        self.buffer['target_idx'] = np.zeros((max_size, 1), dtype=np.uint32)
        if weighted:
            self.buffer['counter'] = np.zeros((max_size, 1), dtype=np.uint32)
            

    def store(self, inp_data, tar_idx, counter=None):

        Bs = tar_idx.shape[0]
        # print(tar_idx.shape)
        for i in range(Bs):
            if tar_idx[i][0] + self.sample_stride > self.dataset_max_idx or (self.weighted and counter[i][0] >= 44):
                continue
            else:
                self.buffer['inp'][self.ptr] = inp_data[i] # (96*144,65)
                #self.buffer['target_idx'][self.ptr][:] = (tar_idx[i] + self.sample_stride)[:]
                self.buffer['target_idx'][self.ptr][:] = (tar_idx[i]+self.sample_stride)[:] # (1,) get the next data
                if self.weighted:
                    self.buffer['counter'][self.ptr][:] = (counter[i] + 1)[:]

                self.ptr = self.ptr + 1
                if self.ptr >= self.max_size:
                    self.ptr = 0
                self.size = self.size + 1
                if self.size > self.max_size:
                    self.size = self.max_size

    
    def sample(self, batch_size):
        idx = np.random.permutation(self.size)
        inp = copy.deepcopy(self.buffer['inp'][idx[0:batch_size]])
        tar_idx = copy.deepcopy(self.buffer['target_idx'][idx[0:batch_size]])

        if self.weighted:
            tar_counter = copy.deepcopy(self.buffer['counter'][idx[0:batch_size]])

        target_data_list = []
        input_data_list = []

        for i in range(batch_size):
            # target_data_list.append(self.data_loader.dataset.get_target(tar_idx[i][0]))
            sample = self.data_loader[tar_idx[i][0]] # shape (13824, 309), (13824, 65)
            input_data, target_data = sample[:2]
            input_data[:,122:122+65] = inp[i]
            input_data_list.append(input_data)
            target_data_list.append(target_data)

        input_data = np.stack(input_data_list, axis=0)
        target_data = np.stack(target_data_list, axis=0)
        del target_data_list
        del input_data_list
        gc.collect()

        if self.weighted:
            return input_data, target_data, tar_idx, tar_counter
        else:
            return input_data, target_data, tar_idx
