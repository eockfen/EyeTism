"""
    functions to evaluate models and results
"""

# import libraries ------------------------------------------------------------
import os
import math
import pickle
import pprint
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import imageio.v3 as iio
from tqdm import tqdm
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    RocCurveDisplay,
    fbeta_score,
    make_scorer,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
)
from sklearn.inspection import permutation_importance
from sklearn.model_selection import LearningCurveDisplay, ShuffleSplit
from scripts import utils as ut


# saving a model --------------------------------------------------------------
def save_model(
    m,
    file: str = "some_model.pickle",
    folder: str = None,
    overwrite: bool = False,
):
    """Saves models into "models" folder. If filename already exists, a number
    will be added, if overwrite flag is not set to True.

    Args:
        m (sklearn_model): model to be saved
        f (str): filename
        overwrite (bool): flag to indicate if model can be overwritten.
                            Defaults to False.
    """
    # defaults
    if folder is None:
        folder = os.path.join("..", "models")
    else:
        folder = os.path.join("..", "models", folder)
        if not os.path.exists(folder):
            os.makedirs(folder)

    # path to file & split into filename + extensiopn
    path_file = os.path.join(folder, f"{file}")
    filename, extension = os.path.splitext(path_file)

    # add suffix if neccessary
    counter = 1
    while os.path.exists(path_file) and not overwrite:
        path_file = filename + "_" + str(counter) + extension
        counter += 1

    # save as pickle
    if not os.path.exists(path_file) or overwrite:
        pickle.dump(m, open(path_file, "wb"))
        print(f" -> model saved in: '{path_file}'")
    else:
        print(f" ERROR -> do not overwrite previously saved model in: '{path_file}'")


# fit model and save it, unless it is already fitted --------------------------
def fit_or_load(
    mdl, X_train, y_train, file, folder: str = None, overwrite: bool = False
):
    filename = os.path.join("..", "models", folder, f"{file}")

    # load if exists and should not
    if os.path.exists(filename) and not overwrite:
        print(f" -> model loaded from: '{filename}'")
        return pickle.load(open(filename, "rb"))

    # return if no model found to load and no given to fit
    if mdl is None:
        return print(" -> no model given...")

    # else run & save
    mdl.fit(X_train, y_train)
    save_model(mdl, file, folder=folder, overwrite=overwrite)

    return mdl


# print some information about the model given --------------------------------
def model_info(estimator):
    """Displays some information about the (fitted!) model given.

    Args:
        model (model): Fitted (!) sklearn model.
    """

    # model name
    estimator_name = estimator.__class__.__name__

    # --- GridSearch or RandomizedSearch ---------------
    if estimator_name in ["GridSearchCV", "RandomizedSearchCV"]:
        print(" --------------- " + estimator_name + " --------------- ")

        #    print(" ----- parameter: -----")
        #    pprint.PrettyPrinter(width=20).pprint(model.get_params())
        print("\n ----- best estimator: -----")
        pprint.PrettyPrinter(width=20).pprint(estimator.best_estimator_)
        print("\n ----- best parameter: -----")
        pprint.PrettyPrinter(width=20).pprint(estimator.best_params_)

        estimator_name = estimator.estimator.steps[-1][-1].__class__.__name__
        estimator = estimator.best_estimator_.steps[-1][-1]

    # --- Random Forest ---------------
    if estimator_name == "RandomForestClassifier":
        n_nodes = []
        max_depths = []

        for ind_tree in estimator.estimators_:
            n_nodes.append(ind_tree.tree_.node_count)
            max_depths.append(ind_tree.tree_.max_depth)

        # how deep
        print(f"\n ----- {estimator_name} -----")
        print(f"   has on average {int(np.mean(n_nodes))} nodes")
        print(f"   has on average a maximum depth of {int(np.mean(max_depths))}\n")

    # --- Decision Tree ---------------
    if estimator_name == "DecisionTreeClassifier":
        max_depth = estimator.tree_.max_depth
        n_nodes = estimator.tree_.node_count

        # how deep
        print(f"\n ----- {estimator_name} -----")
        print(f"   has on average {int(np.mean(n_nodes))} nodes")
        print(f"   has on average a maximum depth of {int(np.mean(max_depth))}\n")


# create report for predictions -----------------------------------------------
def report(
    y_train=None,
    y_train_pred=None,
    y_train_proba=None,
    y_test=None,
    y_test_pred=None,
    y_test_proba=None,
):
    """outputs classification results: 'classification report",
    'confusion matrix', and the ROC- AUC curves. Depending on what is given.
    Results separately available for train and test set.

    Args:
        y_train (np.array, optional): Train set. Defaults to None.
        y_train_pred (np.array, optional): Train set predictions. Defaults to None.
        y_train_proba (np.array, optional): Train set probabilities. Defaults to None.
        y_test (np.array, optional): Test set. Defaults to None.
        y_test_pred (np.array, optional): Test set predictions. Defaults to None.
        y_test_proba (np.array, optional): Test set probabilities. Defaults to None.
    """
    line = "-" * 20
    cm_cmap = sns.light_palette("seagreen", as_cmap=True)

    nfig = 0
    nfig = nfig + 1 if y_train_pred is not None else nfig
    nfig = nfig + 1 if y_test_pred is not None else nfig
    nfig = nfig + 1 if (y_train_proba is not None or y_test_proba is not None) else nfig

    if nfig > 0:
        _, ax = plt.subplots(nrows=1, ncols=nfig, figsize=(4 * nfig, 4))
        cf = 1

    # classification report
    if y_train_pred is not None:
        print(line + " classification report for 'Train' " + line)
        print(classification_report(y_train, y_train_pred, digits=3))
        print(f"f(0.5)-score: {fbeta_score(y_train, y_train_pred, beta=0.5):.3f}")
        print(f"f(2.0)-score: {fbeta_score(y_train, y_train_pred, beta=2):.3f}\n")
    if y_test_pred is not None:
        print(line + " classification report for 'Test' " + line)
        print(classification_report(y_test, y_test_pred, digits=3))
        print(f"f(0.5)-score: {fbeta_score(y_test, y_test_pred, beta=0.5):.3f}")
        print(f"f(2.0)-score: {fbeta_score(y_test, y_test_pred, beta=2):.3f}\n")

    # confusion matrix
    if y_train_pred is not None or y_test_pred is not None:
        if y_train_pred is not None:  # train set
            plt.subplot(1, nfig, cf)
            sns.heatmap(
                confusion_matrix(y_train, y_train_pred),
                annot=True,
                cmap=cm_cmap,
                fmt="g",
                cbar=False,
            )
            plt.title("Confusion Matrix for Train")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            ax[cf - 1].set_box_aspect(1)
            cf += 1
        if y_test_pred is not None:  # test set
            plt.subplot(1, nfig, cf)
            sns.heatmap(
                confusion_matrix(y_test, y_test_pred),
                annot=True,
                cmap=cm_cmap,
                fmt="g",
                cbar=False,
            )
            plt.title("Confusion Matrix for Test")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            ax[cf - 1].set_box_aspect(1)
            cf += 1

    # ROC curve
    if y_train_proba is not None or y_test_proba is not None:
        ax[cf - 1].set_aspect("equal", "box")

        if y_train_proba is not None:
            fpr, tpr, _ = roc_curve(y_train, y_train_pred)
            auc = round(roc_auc_score(y_train, y_train_pred), 3)
            RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc, estimator_name="Train").plot(
                ax[cf - 1]
            )
        if y_test_proba is not None:
            fpr, tpr, _ = roc_curve(y_test, y_test_pred)
            auc = round(roc_auc_score(y_test, y_test_pred), 3)
            RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc, estimator_name="Test").plot(
                ax[cf - 1]
            )

    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------
def learning(
    mdl,
    X,
    y,
    score=make_scorer(fbeta_score, beta=2),
    score_name: str = "f2",
    cv: int = 10,
    verbose: int = 0,
):
    """Plots learning curves for the model(s) given.

    Args:
        mdl (model_object | list[model_object]): model or list of models
        X (pd.DataFrame): training features
        y (pd.Series): training target
        score (str | callable, optional): Which score will be evaluated. Defaults to f2.
        score_name (str, optional): Name of scorer for y-axis. Defaults to "f2".
        cv (int, optional): Number of cross-validation folds. Defaults to 10.
        verbose (int, optional): Verbose defines output printed. Defaults to 0.
    """
    # handle input
    if not isinstance(mdl, list):
        mdl = [mdl]

    # define subplot grid
    n = len(mdl)
    n_cols = min([n, 2])
    n_rows = math.ceil(n / n_cols)

    # set parameters
    common_params = {
        "X": X,
        "y": y,
        "train_sizes": np.linspace(0.1, 1.0, 9),
        "cv": ShuffleSplit(n_splits=cv, test_size=0.2, random_state=0),
        "score_type": "both",
        "line_kw": {"marker": "o"},
        "std_display_style": "fill_between",
        "scoring": score,
        "score_name": score_name,
        "verbose": verbose,
    }

    # plotting learning curves
    _, ax = plt.subplots(
        nrows=n_rows, ncols=n_cols, figsize=(n_cols * 6, n_rows * 4), sharey=True
    )

    if len(mdl) > 1:
        ax = ax.flatten()

    for ax_idx, estimator in enumerate(mdl):
        iax = ax if len(mdl) == 1 else ax[ax_idx]
        LearningCurveDisplay.from_estimator(estimator, **common_params, ax=iax)

        handles, label = iax.get_legend_handles_labels()
        iax.legend(handles, ["Training Score", "Test Score"])
        iax.set_title(f"Learning Curve for {estimator.__class__.__name__}")

    # hide unused axes
    if ax_idx + 1 < n_rows * n_cols:
        ax[ax_idx + 1].set_axis_off()

    # tidy up
    plt.tight_layout()


# -----------------------------------------------------------------------------
def error_compare_models(inp, X_test, y_test, proba: bool = True):
    # ----- prepare df containing prediction results -----
    scores = ["accuracy", "recall", "precision", "f1", "f2"]
    y = y_test.to_frame()
    y["img"] = [int(i.split("_")[1]) for i in y.index]

    for name, model in inp.items():
        y[name + "_pred"] = model.predict(X_test)
        if proba:
            proba_test = model.predict_proba(X_test)
            y[name + "_proba"] = proba_test[:, 1]

    # ----- calculate model statistics -----
    mdl_stats = pd.DataFrame(index=inp.keys())
    for name, model in inp.items():
        # scores
        mdl_stats.loc[name, "accuracy"] = accuracy_score(y["asd"], y[name + "_pred"])
        mdl_stats.loc[name, "recall"] = recall_score(y["asd"], y[name + "_pred"])
        mdl_stats.loc[name, "precision"] = precision_score(y["asd"], y[name + "_pred"])
        mdl_stats.loc[name, "f1"] = f1_score(y["asd"], y[name + "_pred"])
        mdl_stats.loc[name, "f2"] = fbeta_score(y["asd"], y[name + "_pred"], beta=2)

    # ----- calculate image statistics -----
    img_stats = pd.DataFrame(index=y["img"].unique())
    img_stats.sort_index(inplace=True)

    for name, model in inp.items():
        for img in y["img"].unique():
            # get data for this img
            _ = y[y["img"] == img]

            # scores
            img_stats.loc[img, name + "_accuracy"] = accuracy_score(
                _["asd"], _[name + "_pred"]
            )
            img_stats.loc[img, name + "_recall"] = recall_score(
                _["asd"], _[name + "_pred"]
            )
            img_stats.loc[img, name + "_precision"] = precision_score(
                _["asd"], _[name + "_pred"]
            )
            img_stats.loc[img, name + "_f1"] = f1_score(_["asd"], _[name + "_pred"])
            img_stats.loc[img, name + "_f2"] = fbeta_score(
                _["asd"], _[name + "_pred"], beta=2
            )

    # --------------- GENERAL MODEL OVERVIEW ------------------------------
    _, axarr = plt.subplots(1, len(scores), figsize=(10, 2))
    for i, score in enumerate(scores):
        ix = np.unravel_index(i, axarr.shape)

        clrs = [
            "#ff3f3f" if i == np.argmax(mdl_stats[score]) else "#7A76C2"
            for i in range(mdl_stats.shape[0])
        ]

        # barplot
        sns.barplot(
            data=mdl_stats,
            x=mdl_stats[score],
            y=mdl_stats.index,
            hue=mdl_stats.index,
            palette=clrs,
            ax=axarr[ix],
        )
        axarr[ix].set(xlim=(0, 1), ylabel="")

    plt.tight_layout()

    # --------------- DETAILLED MODEL OVERVIEW ------------------------------
    for name, model in inp.items():
        fig, axarr = plt.subplots(2, 3, figsize=(10, 10))
        fig.suptitle(name)

        # a) distribution of proba -----
        axi = 0
        ix = np.unravel_index(axi, axarr.shape)
        if proba:
            sns.histplot(
                data=y,
                x=name + "_proba",
                hue="asd",
                bins=50,
                kde=True,
                line_kws=dict(linewidth=3),
                ax=axarr[ix],
            )
            axarr[ix].vlines(0.5, 0, axarr[ix].get_ylim()[1], colors="k")
            axarr[ix].legend(
                labels=["ASD", "TD"], frameon=True, fancybox=True, facecolor="w"
            )
        else:
            axarr[ix].set_axis_off()

        # b) loop metrics -----
        for score in scores:
            mdl_score = name + "_" + score

            axi += 1
            ix = np.unravel_index(axi, axarr.shape)

            # define colors for bars - depending on their rank
            top_10 = img_stats.nlargest(10, mdl_score)
            top_3 = img_stats.nlargest(3, mdl_score)
            clrs = []
            for i in img_stats.index:
                if i in top_3.index:
                    clrs.append("#ff3f3f")
                elif i in top_10.index:
                    clrs.append("#fab95d")
                else:
                    clrs.append("#86859b")

            # barplot
            sns.barplot(
                data=img_stats,
                x=img_stats[mdl_score],
                y=img_stats.index.astype(str),
                hue=img_stats.index,
                legend=False,
                palette=clrs,
                ax=axarr[ix],
            )
            axarr[ix].set_ylabel("image")

        plt.tight_layout()

    # --------------- SINGLE IMAGE - MODEL COMPARISONS ------------------------------
    all_images = sorted(y["img"].unique())
    n_cols = 1 + len(scores)
    _, axarr = plt.subplots(
        len(all_images), n_cols, figsize=(3 * n_cols, 2.5 * len(all_images))
    )

    # loop images
    for ii, img in tqdm(enumerate(all_images)):
        # 0) ----- image -----
        ix = np.unravel_index(n_cols * ii, axarr.shape)
        loaded_img = iio.imread(
            os.path.join(
                "..",
                "data",
                "Saliency4ASD",
                "TrainingData",
                "Images",
                f"{int(img)}.png",
            )
        )

        axarr[ix].imshow(loaded_img)
        axarr[ix].grid(False)
        axarr[ix].set_title(f"{int(img)}.png")
        axarr[ix].tick_params(labelleft=False, labelbottom=False)

        # 1-5) scores
        for i, score in enumerate(scores):
            ix = np.unravel_index(n_cols * ii + i + 1, axarr.shape)

            cols = [c for c in img_stats.columns if score in c]
            names = [
                "_".join(c.split("_")[0:-1]) for c in img_stats.columns if score in c
            ]

            clrs = [
                "#ff3f3f" if i == np.argmax(img_stats.loc[img, cols]) else "#7A76C2"
                for i, _ in enumerate(img_stats.loc[img, cols])
            ]

            # barplot
            sns.barplot(
                x=img_stats[cols].loc[img],
                y=img_stats[cols].columns,
                hue=img_stats[cols].columns,
                palette=clrs,
                ax=axarr[ix],
            )
            axarr[ix].set_xlim(0, 1)
            axarr[ix].set(ylabel="", xlabel=score, yticks=list(range(len(inp))), yticklabels=names)
            if i > 0:
                axarr[ix].set_yticklabels([])

    plt.tight_layout()


# -----------------------------------------------------------------------------
def error_images(y_test, pred_test, proba_test):
    # ----- prepare df containing prediction results -----
    y = y_test.to_frame()
    y["img"] = [int(i.split("_")[1]) for i in y.index]
    y["pred"] = pred_test
    y["proba"] = proba_test[:, 1]
    y["error"] = ut.code_ytype(y_test, pred_test)

    # ----- calculate image statistics -----
    img_stats = pd.DataFrame(index=y["img"].unique())
    img_stats.sort_index(inplace=True)
    scores = ["acc", "recall", "precision", "f1", "f2"]

    for img in y["img"].unique():
        # get data for this img
        _ = y[y["img"] == img]

        # scores
        img_stats.loc[img, "acc"] = accuracy_score(_["asd"], _["pred"])
        img_stats.loc[img, "recall"] = recall_score(_["asd"], _["pred"])
        img_stats.loc[img, "precision"] = precision_score(_["asd"], _["pred"])
        img_stats.loc[img, "f1"] = f1_score(_["asd"], _["pred"])
        img_stats.loc[img, "f2"] = fbeta_score(_["asd"], _["pred"], beta=2)

    # --------------- GENERAL OVERVIEW ------------------------------
    _, axarr = plt.subplots(2, 3, figsize=(10, 10))

    # a) distribution of proba -----
    axi = 0
    ix = np.unravel_index(axi, axarr.shape)
    if proba_test is not None:
        sns.histplot(
            data=y,
            x="proba",
            hue="asd",
            bins=50,
            kde=True,
            line_kws=dict(linewidth=3),
            ax=axarr[ix],
        )
        axarr[ix].vlines(0.5, 0, axarr[ix].get_ylim()[1], colors="k")
        axarr[ix].legend(
            labels=["ASD", "TD"], frameon=True, fancybox=True, facecolor="w"
        )
    else:
        axarr[ix].set_axis_off()

    # b) loop metrics -----
    for score in scores:
        axi += 1
        ix = np.unravel_index(axi, axarr.shape)

        # define colors for bars - depending on their rank
        top_10 = img_stats.nlargest(10, score)
        top_3 = img_stats.nlargest(3, score)
        clrs = []
        for i in img_stats.index:
            if i in top_3.index:
                clrs.append("#ff3f3f")
            elif i in top_10.index:
                clrs.append("#fab95d")
            else:
                clrs.append("#86859b")

        # barplot
        sns.barplot(
            data=img_stats,
            x=img_stats[score],
            y=img_stats.index.astype(str),
            hue=img_stats.index,
            legend=False,
            palette=clrs,
            ax=axarr[ix],
        )
        axarr[ix].set_ylabel("image")

    plt.tight_layout()

    # --------------- SINGLE IMAGE RESULTS ------------------------------
    all_images = sorted(y["img"].unique())
    n_cols = 4 if proba_test is not None else 3
    _, axarr = plt.subplots(
        len(all_images), n_cols, figsize=(3 * n_cols, 3 * len(all_images))
    )

    # loop images
    for ii, img in tqdm(enumerate(all_images)):
        img_df = y[y["img"] == img]

        # 0) ----- image -----
        ix = np.unravel_index(n_cols * ii, axarr.shape)
        loaded_img = iio.imread(
            os.path.join(
                "..",
                "data",
                "Saliency4ASD",
                "TrainingData",
                "Images",
                f"{int(img)}.png",
            )
        )

        axarr[ix].imshow(loaded_img)
        axarr[ix].grid(False)
        axarr[ix].set_title(f"{int(img)}.png")
        axarr[ix].tick_params(labelleft=False, labelbottom=False)

        # 1) ----- confusion matrix -----
        ix = np.unravel_index(n_cols * ii + 1, axarr.shape)
        cm_cmap = sns.light_palette("seagreen", as_cmap=True)

        sns.heatmap(
            confusion_matrix(img_df["asd"], img_df["pred"]),
            annot=True,
            cmap=cm_cmap,
            fmt="g",
            cbar=False,
            annot_kws={"fontsize": 20},
            xticklabels=["TD", "ASD"],
            yticklabels=["TD", "ASD"],
            ax=axarr[ix],
        )
        axarr[ix].set_box_aspect(1)
        axarr[ix].set_title("Conf. Matrix for Test Image")
        axarr[ix].set_xlabel("Predicted")
        axarr[ix].set_ylabel("Actual")

        # 2) ----- scores -----
        clrs = ["#ff3f3f" if i > 0.8 else "#7A76C2" for i in img_stats.loc[img]]

        ix = np.unravel_index(n_cols * ii + 2, axarr.shape)
        sns.barplot(
            x=img_stats.columns,
            y=img_stats.loc[img],
            hue=img_stats.columns,
            palette=clrs,
            ax=axarr[ix],
        )
        axarr[ix].set_ylabel("score")
        axarr[ix].set_ylim(0, 1)

        # 3) ----- probabilities -----
        if proba_test is not None:
            ix = np.unravel_index(n_cols * ii + 3, axarr.shape)
            axarr[ix].vlines(0.5, -0.4, 1.4, colors="k")
            sns.scatterplot(
                data=img_df,
                x="proba",
                y="asd",
                hue="asd",
                s=100,
                legend=False,
                ax=axarr[ix],
            )
            sns.kdeplot(
                data=img_df,
                x="proba",
                hue="asd",
                legend=False,
                ax=axarr[ix],
            )
            axarr[ix].set(
                xlim=(0, 1),
                yticks=[0, 1],
                yticklabels=["TD", "ASD"],
                ylabel="",
                xlabel="probability",
            )

    plt.tight_layout()


# -----------------------------------------------------------------------------
def feat_importance(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    n_reps: int = 20,
    RSEED: int = 42,
    n_jobs: int = -1,
):
    # calculate permutation importance for training data
    result_train = permutation_importance(
        model, X_train, y_train, n_repeats=n_reps, random_state=RSEED, n_jobs=n_jobs
    )
    sorted_importances_idx_train = result_train.importances_mean.argsort()
    importances_train = pd.DataFrame(
        result_train.importances[sorted_importances_idx_train].T,
        columns=X_train.columns[sorted_importances_idx_train],
    )

    # calculate permutation importance for test data
    result_test = permutation_importance(
        model, X_test, y_test, n_repeats=n_reps, random_state=RSEED, n_jobs=n_jobs
    )
    sorted_importances_idx_test = result_test.importances_mean.argsort()
    importances_test = pd.DataFrame(
        result_test.importances[sorted_importances_idx_test].T,
        columns=X_test.columns[sorted_importances_idx_test],
    )

    # figure
    _, axs = plt.subplots(1, 2, figsize=(15, 6))
    importances_train.plot.box(vert=False, whis=10, ax=axs[0])
    axs[0].set_title("Permutation Importances - Train Set")
    axs[0].axvline(x=0, color="k", linestyle="--")
    axs[0].set_xlabel("Decrease in accuracy score")
    axs[0].figure.tight_layout()

    importances_test.plot.box(vert=False, whis=10, ax=axs[1])
    axs[1].set_title("Permutation Importances - Test Set")
    axs[1].axvline(x=0, color="k", linestyle="--")
    axs[1].set_xlabel("Decrease in accuracy score")
    axs[1].figure.tight_layout()


# --- if script is run by it's own --------------------------------------------
if __name__ == "__main__":
    # set file & folder name
    folder_name = "RF_grid"
    model_name = "RF_grid_v1_full.pickle"

    # # fit or load
    # grid_search_rf = fit_or_load(
    #     None, X_train, y_train, model_name, folder=folder_name
    # )
