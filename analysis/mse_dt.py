import sys
import os
import numpy as np
import xgboost as xgb
from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils import data
import matplotlib.pyplot as plt
from joblib import dump

sys.path.append(os.path.join(sys.path[0], "..", "dataloader"))
from dataloader_newformat import DatasetDisk
sys.path.append(os.path.join(sys.path[0], "..", "utils"))
from data_shape import to_inference_shape

if __name__ == "__main__":
    import os
    import glob
    data_dir = "/home/users/data/nncam_data/image_set/"
    if not os.path.isdir(data_dir):
        data_dir = "/data/nncam_data/image_set/"
    if not os.path.isdir(data_dir):
        # data_dir = "./data/"
        data_dir = "/pscratch/sd/c/chenjd21/spcam_new_data/"
        #data_dir = "./test_data/"
    if not os.path.isdir(data_dir):
        # data_dir = "./data/"
        #data_dir = "/pscratch/sd/c/chenjd21/spcam_new_data/"
        data_dir = "./test_data/"
    pred_dir = "/pscratch/sd/c/chenjd21/resmlp_pred/"
    if not os.path.isdir(pred_dir):
        pred_dir = "./pred_data/"
    file_names = glob.glob(data_dir + "*.npy")
    file_names.sort()
    file_names = file_names[35041:35041+17530:12]
    # file_names = [data_dir + fn for fn in file_names]
    print(file_names[:10])
    col_names = np.loadtxt(data_dir + "/col_names.txt", dtype=str)
    col_names_x = col_names
    #col_names_x = ["QL", "T_nn_in", "dqvls_nn_in", "dTls_nn_in", "SOLIN", "SPPS"]
    #col_names_y = ["qtend_check", "stend_check", "SOLL", "SOLLD", "SOLS", "SOLSD", "FSDS"]
    col_names_y = ["qtend_check"]
    training_set = DatasetDisk(file_names, col_names, col_names_x, col_names_y, is_train=True, noise_std=0, filename=True, output_normalized=False)
    trainloader = data.DataLoader(training_set, shuffle=False, batch_size=1, num_workers=1)
    X = []
    Y = []
    P = []
    files_to_check = []
    for idx, batch in enumerate(trainloader):
        x, y, x_raw, file_names = batch
        print(f"{idx}/{len(trainloader)},", "x:", x.size(), "y:", y.size(), "x_raw:", x_raw.size(), file_names, end="\r")
        preds = to_inference_shape(np.load(pred_dir + file_names[0].split("/")[-1])[None,])
        # preds = np.concatenate(preds, axis=0)
        # print("pred:", preds.shape)
        # X.append(x_raw[0])
        # Y.append(y[0])
        # P.append(preds)
        r2 = r2_score(y[0], preds, multioutput="variance_weighted")
        if r2 < 0:
            files_to_check.append(file_names[0])
            continue
        X.append(x_raw[0])
        Y.append(y[0])
        P.append(preds)
    X = np.concatenate(X, axis=0)
    Y = np.concatenate(Y, axis=0)
    P = np.concatenate(P, axis=0)
    print("X:", X.shape)
    print("Y:", Y.shape)
    print("P:", P.shape)
    # check R2
    print("r2 of pred: ", r2_score(Y, P, multioutput="variance_weighted"))
    print("files_to_check:", files_to_check)
    # print("r2 of pred: ", r2_score(Y[:,29], P[:,29], multioutput="variance_weighted"))
    # for i in range(X.shape[0]):
    #     print("idx:", i)
    #     print("r2 of pred: ", r2_score(Y[i], P[i], multioutput="variance_weighted"))
    #     print("r2 of pred: ", r2_score(Y[i][:,29], P[i][:,29], multioutput="variance_weighted"))
    #     print("r2 of pred: ", r2_score(Y[i].reshape(-1), P[i].reshape(-1)))
    # Load the California housing dataset
    # data = fetch_california_housing()
    # X = data.data
    # print(X.shape)
    # y = data.target
    # print(y.shape)
    MSE = np.sqrt(np.square(Y-P))
    print(MSE.shape)

    target_levels = [12,18,23,28]
    for tl in target_levels:
        # Split the dataset into training and testing sets
        scaler = MinMaxScaler()
        target = scaler.fit_transform(MSE[:,tl:tl+1])
        # X_train, X_temp, y_train, y_temp = train_test_split(X, target, test_size=0.3, random_state=42)
        # X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.2, random_state=42)

        # xgb_reg = xgb.XGBRegressor(objective ='reg:squarederror', n_estimators=500, eta=0.1, max_depth=7, subsample=0.7, colsample_bytree=0.8, random_state=42, eval_metric="rmse")
        # print(xgb_reg.get_params())

        # # Train the model
        # eval_set = [(X_train, y_train), (X_val, y_val)]
        # xgb_reg.fit(X_train, y_train, eval_set=eval_set, verbose=True)
        # # xgb_reg.fit(X_train, y_train)
        # # Retrieve performance metrics
        # results = xgb_reg.evals_result()

        # # Plotting
        # epochs = len(results['validation_0']['rmse'])
        # x_axis = range(0, epochs)

        # plt.figure(figsize=(10, 6))
        # plt.plot(x_axis, results['validation_0']['rmse'], label='Train')
        # plt.plot(x_axis, results['validation_1']['rmse'], label='Validation')
        # plt.legend()
        # plt.ylabel('RMSE')
        # plt.xlabel('Number of Trees')
        # plt.title('XGBoost RMSE over Boosting Rounds')
        # plt.show()

        # # Make predictions
        # y_pred = xgb_reg.predict(X_test)

        # # Evaluate the model
        # mse = mean_squared_error(y_test, y_pred)
        # rmse = np.sqrt(mse)
        # r2 = r2_score(y_test, y_pred)

        # print("Root Mean Squared Error:", rmse)
        # print("R-squared:", r2)

        # Initialize the DecisionTreeRegressor
        tree_reg = DecisionTreeRegressor(random_state=42)

        # Train the model
        tree_reg.fit(X_train, y_train)

        # Make predictions
        y_pred = tree_reg.predict(X_test)

        # Evaluate the model
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        print(f"Decision tree for level {tl}")
        print("Root Mean Squared Error:", rmse)
        print("R-squared:", r2)
        dump(tree_reg, 'tree_reg_' + str(tl) + '.joblib')
