import os
import random
import numpy as np
from dataloader_time_embedded import filename_to_idx

if __name__ == "__main__":
   regular_file_x0 = "checkcode_x0.npy" 
   regular_file_x1 = "checkcode_x1.npy" 
   regular_file_x2 = "checkcode_x2.npy" 
   regular_file_y0 = "checkcode_y0.npy" 
   regular_file_y1 = "checkcode_y1.npy" 
   regular_file_y2 = "checkcode_y2.npy" 
   time_file_x0 = "checkcode_x_time0.npy" 
   time_file_y0 = "checkcode_y_time0.npy" 
   time_file_x1 = "checkcode_x_time1.npy" 
   time_file_y1 = "checkcode_y_time1.npy" 

   # first test, output y1 should match
   regular_y1 = np.load(regular_file_y1)
   time_y0 = np.load(time_file_y0)
   print("regualr_y1:", regular_y1.shape)
   print("time_y0:", time_y0.shape)
   print("Equals:", np.all(regular_y1 == time_y0))
   print("MSE:", np.mean(np.square(regular_y1 - time_y0)))

   # second test, input x0 should match
   regular_x0 = np.load(regular_file_x0)
   time_x0 = np.load(time_file_x0)

   print("regualr_x0:", regular_x0.shape)
   print("time_x0:", time_x0.shape)
   print("Slice Equals:", np.all(regular_x0 == time_x0[:,:122]))
   print("MSE:", np.mean(np.square(regular_x0 - time_x0[:,:122])))

   # third test, input x1 should match
   regular_x1 = np.load(regular_file_x1)
   print("regualr_x1:", regular_x1.shape)
   print("time_x0:", time_x0.shape)
   print("Slice Equals:", np.all(regular_x1 == time_x0[:,122:244]))
   print("MSE:", np.mean(np.square(regular_x1 - time_x0[:,122:244])))

   # fourth test, input y0 should match
   regular_y0 = np.load(regular_file_y0)
   print("regualr_y0:", regular_y0.shape)
   print("time_x0:", time_x0.shape)
   print("Slice Equals:", np.all(regular_y0 == time_x0[:,244:]))
   print("MSE:", np.mean(np.square(regular_y0 - time_x0[:,244:])))

    # fifth test, input y2 should match 
   time_y1 = np.load(time_file_y1)
   regular_y2 = np.load(regular_file_y2)
   print("regualr_y2:", regular_y2.shape)
   print("time_y1:", time_y1.shape)
   print("Equals:", np.all(regular_y2 == time_y1))
   print("MSE:", np.mean(np.square(regular_y2 - time_y1)))

   # sixth test, input x1 should match
   regular_x1 = np.load(regular_file_x1)
   time_x1 = np.load(time_file_x1)

   print("regualr_x1:", regular_x1.shape)
   print("time_x1:", time_x1.shape)
   print("Slice Equals:", np.all(regular_x1 == time_x1[:,:122]))
   print("MSE:", np.mean(np.square(regular_x1 - time_x1[:,:122])))

   # seventh test, input x2 should match
   regular_x2 = np.load(regular_file_x2)
   print("regualr_x2:", regular_x2.shape)
   print("time_x1:", time_x1.shape)
   print("Slice Equals:", np.all(regular_x2 == time_x1[:,122:244]))
   print("MSE:", np.mean(np.square(regular_x2 - time_x1[:,122:244])))

   # eighth test, input y1 should match
   regular_y1 = np.load(regular_file_y1)
   print("regualr_y1:", regular_y1.shape)
   print("time_x1:", time_x1.shape)
   print("Slice Equals:", np.all(regular_y1 == time_x1[:,244:]))
   print("MSE:", np.mean(np.square(regular_y1 - time_x1[:,244:])))
