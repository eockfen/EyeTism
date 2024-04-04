# it's all about the features
import os
import pandas as pd


# --- own train-test-split function -------------------------------------------
def split(X: pd.DataFrame, y: pd.DataFrame, test_set: list = []):
    if test_set == []:
        test_set = [
            112,
            113,
            120,
            135,
            138,
            165,
            166,
            176,
            191,
            193,
            20,
            203,
            207,
            216,
            233,
            253,
            258,
            271,
            272,
            283,
            287,
            4,
            45,
            47,
            73,
            74,
            8,
            81,
            95,
            96,
        ]

    # create comparison string
    cmp = "_" + "_|_".join([str(i) for i in test_set]) + "_"

    # find indices of test set
    idx = X.index.str.contains(cmp)

    # define train/test sets
    X_train = X[~idx]
    X_test = X[idx]
    y_train = y[~idx]
    y_test = y[idx]

    return X_train, X_test, y_train, y_test


# --- if script is run by it's own --------------------------------------------
if __name__ == "__main__":
    curdir = os.path.dirname(__file__)
    path_df = os.path.join(curdir, "..", "data", "df.csv")

    df = pd.read_csv(path_df, index_col=0)
    df = df.set_index("id", drop=True)

    X = df.drop({"img", "sp_idx", "dummy_feature_name"}, axis=1)
    y = X.pop("asd")

    X_train, X_test, y_train, y_test = split(X, y)
    print(X_train.shape)
    print(X_test.shape)

    # get_sp_features(who="TD")
    # calculate_sp_features(sp_file=sp_file)
    # calculate_saliency_features(sp_file=sp_file)