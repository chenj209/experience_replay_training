import numpy as np
import torch
import copy
import gc
from multiprocessing import shared_memory
import utils.misc as utils
import copy

class replay_buff():
    def __init__(self, data_loader, inp_shape=[69, 721, 1440], max_size=300, weighted=False) -> None:
        self.ptr = 0
        self.size = 0
        self.input_shape = inp_shape
        self.max_size = max_size
        self.weighted = weighted
        self.data_loader = data_loader
        self.sample_stride = data_loader.dataset.sample_stride
        self.dataset_max_idx = data_loader.dataset.get_maxidx()
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
                self.buffer['inp'][self.ptr] = inp_data[i]
                self.buffer['target_idx'][self.ptr][:] = (tar_idx[i] + self.sample_stride)[:]
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

        for i in range(batch_size):
            target_data_list.append(self.data_loader.dataset.get_target(tar_idx[i][0]))
        target_data = np.stack(target_data_list, axis=0)
        del target_data_list
        gc.collect()

        if self.weighted:
            return inp, target_data, tar_idx, tar_counter
        else:
            return inp, target_data, tar_idx
