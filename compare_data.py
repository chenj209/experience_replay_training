import numpy as np

if __name__ == "__main__":
    test_set = "/home/users/data/nncam_data/image_testset/"
    train_set = "/home/users/data/nncam_data/image_set/"

    train100 = np.load(train_set + "00300.npz")
    test100 = np.load(test_set + "00300.npz")

    print("Shape:")
    print("train data_x: ", train100["data_x"].shape)
    print("test data_x: ", test100["data_x"].shape)
    print("train data_y: ", train100["data_y"].shape)
    print("test data_y: ", test100["data_y"].shape)

    print("Mean:")
    print("train Q:", train100["data_x"][:,:30,:,:].mean())
    print("test Q:", test100["data_x"][:,:30,:,:].mean())
    print("train dQ:", train100["data_y"][:,:30,:,:].mean())
    print("test dQ:", test100["data_y"][:,:30,:,:].mean())

    print("train T:", train100["data_x"][:,30:60,:,:].mean())
    print("test T:", test100["data_x"][:,30:60,:,:].mean())
    print("train ds:", train100["data_y"][:,30:60,:,:].mean())
    print("test ds:", test100["data_y"][:,30:60,:,:].mean())



    print("train dqls:", train100["data_x"][:,60:90,:,:].mean())
    print("test dqls:", test100["data_x"][:,60:90,:,:].mean())
    print("train dtls:", train100["data_x"][:,90:120,:,:].mean())
    print("test dtls:", test100["data_x"][:,90:120,:,:].mean())
    print("train ps:", train100["data_x"][:,120:121,:].mean())
    print("test ps:", test100["data_x"][:,120:121,:,:].mean())
    print("train solin:", train100["data_x"][:,121:122,:,:].mean())
    print("test solin:", test100["data_x"][:,121:122,:,:].mean())

    print("train rad:", train100["data_y"][:,61:66,:,:].mean())
    print("test rad:", test100["data_y"][:,61:66,:,:].mean())
