import matplotlib.pyplot as plt
import numpy as np
import sys

data = np.load(sys.argv[1])
fig, ax = plt.subplots(1,2)
im1 = ax[0].imshow(data["x"][0,0])
plt.colorbar(im1)
im2 = ax[1].imshow(data["x_rec"][0,0])
plt.colorbar(im2)
plt.show()
