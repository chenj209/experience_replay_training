if __name__ == "__main__":
    import os
    import glob
    import argparse
    import numpy as np
    from sklearn.tree import DecisionTreeRegressor, export_graphviz
    import graphviz
    from joblib import load
    parser = argparse.ArgumentParser()
    parser.add_argument("tree_dump", type=str)
    args = parser.parse_args()
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
    col_names = np.loadtxt(data_dir + "/col_names.txt", dtype=str)
    col_names_prev = [col_names[i]+"_prev" for i in range(len(col_names))]
    full_col_names = list(col_names_prev) + list(col_names)
    tree_clf = load(args.tree_dump)

    # Export as dot file
    dot_data = export_graphviz(tree_clf, out_file=None, 
                            feature_names=full_col_names,  
                            class_names=["qtend_RMSE"],
                            filled=True, rounded=True,  
                            special_characters=True)

    # Draw graph
    graph = graphviz.Source(dot_data)  
    graph.render(args.tree_dump.strip(".joblib"))  # Saves to file