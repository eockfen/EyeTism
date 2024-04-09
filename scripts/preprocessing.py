# it's all about the features
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def display_corr_matrix(data, thresh: float = None):
    """Display a heatmap from a given dataset

    Args:
        data (dataset): dataframe containing columns to be correlated with each other
        thresh (float): threshold correlation value defines which r's should be annotated

    Returns:
        g (graph)
    """
    # Create a correlation matrix & a mask for the lower triangle
    cm = data.corr()
    mask = np.zeros_like(cm)
    mask[np.triu_indices_from(mask)] = None

    txt_s = cm.values.astype(str)
    txt_s[np.abs(cm.values) >= thresh] = "X"
    txt_s[np.abs(cm.values) < thresh] = ""

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(240, 10, sep=20, n=9, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    annot = False if thresh is None else txt_s
    plt.figure(figsize=(9, 9))
    g = sns.heatmap(
        cm,
        cmap=cmap,
        mask=mask,
        square=True,
        annot=annot,
        fmt="",
        center=0,
        cbar_kws={"shrink": 0.5, "label": "Pearson r"},
    )
    g.grid(False)
    plt.show()


def check_correlations(df, thresh):
    corr_matrix = df.corr()

    # Find pairs with correlation >= 0.8
    high_corr_pairs = np.column_stack(
        np.where((np.abs(corr_matrix) >= thresh) & (corr_matrix != 1))
    )
    high_corr_cols = []

    # Extracting and printing the pairs
    seen_pairs = set()
    for i, j in high_corr_pairs:
        col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
        if (col2, col1) not in seen_pairs:
            print(
                f"Correlation between {col1} and {col2} is {round(corr_matrix.iloc[i, j], 3)}"
            )
            seen_pairs.add((col1, col2))
            seen_pairs.add((col2, col1))
            high_corr_cols.append(col1)
            high_corr_cols.append(col2)

    # display corr. matrix
    display_corr_matrix(df, thresh=thresh)


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
    cmp = "_" + "_|_".join([f"{i:03.0f}" for i in test_set]) + "_"

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
