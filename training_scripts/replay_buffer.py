import numpy as np
import torch
import copy
import gc
import copy
import time

from multiprocessing import Pool, Array
import ctypes

def init_worker(data_loader_, buffer_inp_, buffer_tar_idx_, weighted_, buffer_counter_):
    global data_loader, buffer_inp, buffer_tar_idx, weighted, buffer_counter
    data_loader = data_loader_
    buffer_inp = buffer_inp_
    buffer_tar_idx = buffer_tar_idx_
    weighted = weighted_
    buffer_counter = buffer_counter_

def process_sample(args):
    tar_idx, inp_idx, batch_size, input_shape = args
    input_data_list = []
    target_data_list = []

    buffer_inp_np = np.frombuffer(buffer_inp, dtype=np.float32).reshape((batch_size, input_shape[1]))
    
    for i in range(len(tar_idx)):
        sample = data_loader.get_index(tar_idx[i][0])  # shape (13824, 309), (13824, 65)
        input_data, target_data = sample[:2]
        inp = buffer_inp_np[inp_idx[i]]
        input_data[:, 122:122+65] = inp
        input_data_list.append(input_data)
        target_data_list.append(target_data)

    input_data = np.stack(input_data_list, axis=0)
    target_data = np.stack(target_data_list, axis=0)
    
    if weighted:
        tar_counter = np.frombuffer(buffer_counter, dtype=np.float32).reshape((batch_size, -1))[tar_idx]
        return input_data, target_data, tar_idx, tar_counter
    else:
        return input_data, target_data, tar_idx


class ReplayBuffer():
    def __init__(self, data_loader, inp_shape=[96*144, 309], tar_shape=[96*144,65], max_size=300, weighted=False, workers=4) -> None:
        self.ptr = 0
        self.size = 0
        self.input_shape = inp_shape
        self.max_size = max_size
        self.weighted = weighted
        self.data_loader = data_loader
        self.sample_stride = 1
        self.workers = workers
        self.dataset_max_idx = data_loader.size-1
        self.buffer = {}
        self.buffer['inp'] = np.zeros((max_size, *inp_shape), dtype=np.float32)
        self.buffer['target'] = np.zeros((max_size, *tar_shape), dtype=np.float32)
        # inp_buf = np.zeros((max_size, *inp_shape), dtype=np.float32)
        # self.shm = shared_memory.SharedMemory(create=True, size=inp_buf.nbytes)
        # self.shm.unlink()
        # self.buffer['inp'] = np.ndarray(inp_buf.shape, dtype=inp_buf.dtype, buffer=self.shm.buf)
        self.buffer['target_idx'] = np.zeros((max_size, 1), dtype=np.uint32)
        if weighted:
            self.buffer['counter'] = np.zeros((max_size, 1), dtype=np.uint32)
            

    def store(self, inp_data, tar_data, tar_idx, counter=None):
        start = time.time()
        Bs = tar_data.shape[0]

        end_ptr = self.ptr + Bs
        if end_ptr > self.max_size:
            overflow = end_ptr - self.max_size
            indices1 = slice(self.ptr, self.max_size)
            indices2 = slice(0, overflow)
            self.buffer['inp'][indices1] = inp_data[:self.max_size-self.ptr]
            self.buffer['inp'][indices2] = inp_data[self.max_size-self.ptr:]
            self.buffer['target'][indices1] = tar_data[:self.max_size-self.ptr]
            self.buffer['target'][indices2] = tar_data[self.max_size-self.ptr:]
            self.buffer['target_idx'][indices1] = tar_idx[:self.max_size-self.ptr]
            self.buffer['target_idx'][indices2] = tar_idx[self.max_size-self.ptr:]
        else:
            indices = slice(self.ptr, end_ptr)
            self.buffer['inp'][indices] = inp_data
            self.buffer['target'][indices] = tar_data
            self.buffer['target_idx'][indices] = tar_idx

        self.ptr = end_ptr % self.max_size
        self.size = min(self.size + Bs, self.max_size)

        print(f"Store time: {time.time() - start}")
        #start = time.time()
        #Bs = tar_idx.shape[0]
        #self.buffer['inp'][self.ptr:self.ptr+Bs] = inp_data # (96*144,309)
        #self.buffer['target'][self.ptr:self.ptr+Bs] = tar_data # (96*144,309)
        ##self.buffer['target_idx'][self.ptr][:] = (tar_idx[i] + self.sample_stride)[:]
        #self.buffer['target_idx'][self.ptr:self.ptr+Bs] = (tar_idx+self.sample_stride) # (1,) get the next data
#
#        self.ptr = self.ptr + Bs
#        if self.ptr >= self.max_size:
#            self.ptr = 0
#        self.size = self.size + Bs
#        if self.size > self.max_size:
#            self.size = self.max_size
#        print(f"store time: {time.time() - start}")
        #print(tar_idx.shape)
        #for i in range(Bs):
        #    if tar_idx[i][0] + self.sample_stride > self.dataset_max_idx or (self.weighted and counter[i][0] >= 44):
        #        print("here skip:", tar_idx[i][0], self.dataset_max_idx)
        #        continue
        #    else:
        #        self.buffer['inp'][self.ptr] = inp_data[i] # (96*144,309)
        #        self.buffer['target'][self.ptr] = tar_data[i] # (96*144,309)
        #        #self.buffer['target_idx'][self.ptr][:] = (tar_idx[i] + self.sample_stride)[:]
        #        self.buffer['target_idx'][self.ptr][:] = (tar_idx[i]+self.sample_stride)[:] # (1,) get the next data
                #if self.weighted:
        #            self.buffer['counter'][self.ptr][:] = (counter[i] + 1)[:]
#
#                self.ptr = self.ptr + 1
#                if self.ptr >= self.max_size:
                #    self.ptr = 0
                #self.size = self.size + 1
                #if self.size > self.max_size:
                #    self.size = self.max_size
    
    def sample(self, batch_size):
        start = time.time()
        idx = np.random.permutation(self.size)
        #inp = copy.deepcopy(self.buffer['inp'][idx[0:batch_size]])
        #tar = copy.deepcopy(self.buffer['target'][idx[0:batch_size]])
        #tar_idx = copy.deepcopy(self.buffer['target_idx'][idx[0:batch_size]])
        inp = self.buffer['inp'][idx[0:batch_size]]
        tar = self.buffer['target'][idx[0:batch_size]]
        tar_idx = self.buffer['target_idx'][idx[0:batch_size]]

        if self.weighted:
            #tar_counter = copy.deepcopy(self.buffer['counter'][idx[0:batch_size]])
            tar_counter = self.buffer['counter'][idx[0:batch_size]]

        #target_data_list = tar
        #input_data_list = inp
        #print(inp.shape)

        # for i in range(batch_size):
            # target_data_list.append(self.data_loader.dataset.get_target(tar_idx[i][0]))
            # sample = self.data_loader.get_index(tar_idx[i][0]) # shape (13824, 309), (13824, 65)
            # input_data, target_data = sample[:2]
            # input_data[:,122:122+65] = inp[i]
            # input_data_list.append(input_data)
            # target_data_list.append(target_data)

        #input_data = np.stack(input_data_list, axis=0)
        input_data = inp
        #target_data = np.stack(target_data_list, axis=0)
        target_data = tar
        #del target_data_list
        #del input_data_list
        #gc.collect()
        print(f"Sampeld size {batch_size}, time: {time.time() - start}")

        if self.weighted:
            return input_data, target_data, tar_idx, tar_counter
        else:
            return input_data, target_data, tar_idx

    # def sample(self, batch_size):
    #     start = time.time()
    #     idx = np.random.permutation(self.size)
    #     inp = self.buffer['inp'][idx[0:batch_size]]
    #     tar_idx = self.buffer['target_idx'][idx[0:batch_size]]
    #     weighted = self.weighted
    #     num_workers = self.workers

    #     if num_workers == 1:
    #         # Sequential version
    #         input_data_list = []
    #         target_data_list = []
    #         for i in range(batch_size):
    #             sample = self.data_loader.get_index(tar_idx[i][0])  # shape (13824, 309), (13824, 65)
    #             input_data, target_data = sample[:2]
    #             input_data[:, 122:122+65] = inp[i]
    #             input_data_list.append(input_data)
    #             target_data_list.append(target_data)

    #         input_data = np.stack(input_data_list, axis=0)
    #         target_data = np.stack(target_data_list, axis=0)
            
    #         if weighted:
    #             tar_counter = self.buffer['counter'][idx[0:batch_size]]
    #             gc.collect()
    #             print(f"Sampled size {batch_size}, time: {time.time() - start}")
    #             return input_data, target_data, tar_idx, tar_counter
    #         else:
    #             gc.collect()
    #             print(f"Sampled size {batch_size}, time: {time.time() - start}")
    #             return input_data, target_data, tar_idx
    #     else:
    #         # Parallel version
    #         chunk_size = batch_size // num_workers

    #         buffer_inp = Array(ctypes.c_float, inp.flatten(), lock=False)
    #         buffer_tar_idx = Array(ctypes.c_int, tar_idx.flatten(), lock=False)
    #         buffer_counter = None
    #         if weighted:
    #             buffer_counter = Array(ctypes.c_float, self.buffer['counter'][idx[0:batch_size]].flatten(), lock=False)

    #         chunks = [(tar_idx[i:i+chunk_size], list(range(i, i+chunk_size)), batch_size, inp.shape) 
    #                   for i in range(0, batch_size, chunk_size)]

    #         with Pool(processes=num_workers, initializer=init_worker, initargs=(self.data_loader, buffer_inp, buffer_tar_idx, weighted, buffer_counter)) as pool:
    #             results = pool.map(process_sample, chunks)

    #         input_data = np.vstack([result[0] for result in results])
    #         target_data = np.vstack([result[1] for result in results])
    #         tar_idx_combined = np.vstack([result[2] for result in results])

    #         if weighted:
    #             tar_counter_combined = np.vstack([result[3] for result in results])
    #             gc.collect()
    #             print(f"Sampled size {batch_size}, time: {time.time() - start}")
    #             return input_data, target_data, tar_idx_combined, tar_counter_combined
    #         else:
    #             gc.collect()
    #             print(f"Sampled size {batch_size}, time: {time.time() - start}")
    #             return input_data, target_data, tar_idx_combined
