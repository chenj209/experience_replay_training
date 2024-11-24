import numpy as np

Q_idx = (106,106+30)
T_idx = (383,383+30)
dTls_idx = (473,473+30)
dqls_idx = (593,593+30)
qtend_idx = (683,683+30)
stend_idx = (773,773+30)

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
    T = target_data[T_idx[0]:T_idx[1]]
    T_prev = target_data_prev[T_idx[0]:T_idx[1]]
    dTls = target_data[dTls_idx[0]:dTls_idx[1]]
    dTls_prev = target_data_prev[dTls_idx[0]:dTls_idx[1]]
    qtend = target_data[qtend_idx[0]:qtend_idx[1]]
    qtend_prev = target_data_prev[qtend_idx[0]:qtend_idx[1]]
    stend = target_data[stend_idx[0]:stend_idx[1]]
    stend_prev = target_data_prev[stend_idx[0]:stend_idx[1]]
    print("Q mean: ", Q.mean())
    print("dqls_prev mean: ", dqls_prev.mean())
    print("qtend_prev mean: ", qtend_prev.mean())
    print("Q_prev mean: ", Q_prev.mean())
    print("Q_prev+(dqls+qtend_prev)*1800=Q:", np.allclose(Q, Q_prev+(dqls+qtend_prev)*1800))
    print("T mean: ", T.mean())
    print("dTls_prev mean: ", dTls_prev.mean())
    print("stend_prev mean: ", stend_prev.mean())
    print("T_prev mean: ", T_prev.mean())
    print("T_prev+(dTls+stend_prev)*1800=T:", np.allclose(T, T_prev+(dTls+stend_prev)*1800))
    #print("Q_prev+dqls+qtend_prev=Q:", Q.mean(), (Q_prev+(dqls+qtend_prev)*1800).mean())

