import numpy as np

Q_idx = (106,106+30)
dqls_idx = (593,593+30)
qtend_idx = (683,683+30)

DATA_PATH = "/share3/chenj209/spcam_new_data_32/"

target_file = "00100.npy"
target_file_prev = "00099.npy"

if __name__ == "__main__":
    target_data = np.load(DATA_PATH+target_file)
    target_data_prev = np.load(DATA_PATH+target_file_prev)
    Q = target_data[Q_idx[0]:Q_idx[1]]
    Q_prev = target_data_prev[Q_idx[0]:Q_idx[1]]
    dqls = target_data[dqls_idx[0]:dqls_idx[1]]
    dqls_prev = target_data_prev[dqls_idx[0]:dqls_idx[1]]
    qtend = target_data[qtend_idx[0]:qtend_idx[1]]
    qtend_prev = target_data_prev[qtend_idx[0]:qtend_idx[1]]
    print("Q mean: ", Q.mean())
    print("dqls_prev mean: ", dqls_prev.mean())
    print("qtend_prev mean: ", qtend_prev.mean())
    print("Q_prev mean: ", Q_prev.mean())
    print("Q_prev+dqls_prev+qtend_prev=Q:", np.allclose(Q, Q_prev+(dqls+qtend_prev)*1800))
    print("Q_prev+dqls_prev+qtend_prev=Q:", Q.mean(), (Q_prev+(dqls+qtend_prev)*1800).mean())

