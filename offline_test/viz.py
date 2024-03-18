import sys
import numpy as np
import matplotlib.pyplot as plt

path = sys.argv[1]
index = sys.argv[2]
rec = sys.argv[3]

rec_by_var = np.load(f"{path}/offline_test_ae_avg_recmse_by_var.npy")
print(rec_by_var.shape)
plt.plot(rec_by_var)
plt.show()
y = np.load(f"{path}/offline_test_ae_{index}_y.npy")
x = np.load(f"{path}/offline_test_ae_{index}_x.npy")
x_rec = np.load(f"{path}/offline_test_ae_{index}_x_rec.npy")
print("x_rec:", x_rec.shape)
fig, ax = plt.subplots(1,2)
#max_val = max(np.max(x[0,int(rec)]),np.max(x_rec[0,int(rec)]))
#min_val = min(np.min(x[0,int(rec)]),np.min(x_rec[0,int(rec)]))
#im1 = ax[0].imshow(x[0,int(rec)],vmin=min_val,vmax=max_val)
im1 = ax[0].imshow(x[0,int(rec)])
plt.colorbar(im1)
#ax[0].set_title(f"qtend level{rec}")
#im2 = ax[1].imshow(x_rec[0,int(rec)],vmin=min_val,vmax=max_val)
im2 = ax[1].imshow(x_rec[0,int(rec)])
#ax[1].set_title(f"qtend level{rec} rec")
plt.colorbar(im2)
plt.show()
#exit(0)
print(y.shape)
mask = np.load(f"{path}/offline_test_ae_subregion_mask.npy").astype(bool)
data = np.zeros((30,29,55))
mu = np.load(f"{path}/offline_test_ae_{index}_mu.npy").reshape((4,29,55))
var = np.exp(np.load(f"{path}/offline_test_ae_{index}_log_var.npy")).reshape((4,29,55))
#for i in range(30):
#    data[i,mask] = y[:,i]
#    data[data==0] = None
#for i in range(30):
#    im = plt.imshow(data[i])
#    plt.clim(y[:,i].min(), y[:,i].max())
#    plt.colorbar(im, extend="both")
#    plt.title(f"Level {i}")
#    plt.show()
#exit(0)


fig, ax = plt.subplots(2,2)
im1 = ax[0,0].imshow(mu[0])
im2 = ax[0,1].imshow(mu[1])
im3 = ax[1,0].imshow(mu[2])
im4 = ax[1,1].imshow(mu[3])
plt.colorbar(im1, ax=ax[0,0])
plt.colorbar(im2, ax=ax[0,1])
plt.colorbar(im3, ax=ax[1,0])
plt.colorbar(im4, ax=ax[1,1])

plt.show()




fig, ax = plt.subplots(2,2)
im1 = ax[0,0].imshow(var[0])
im2 = ax[0,1].imshow(var[1])
im3 = ax[1,0].imshow(var[2])
im4 = ax[1,1].imshow(var[3])
plt.colorbar(im1, ax=ax[0,0])
plt.colorbar(im2, ax=ax[0,1])
plt.colorbar(im3, ax=ax[1,0])
plt.colorbar(im4, ax=ax[1,1])

plt.show()
