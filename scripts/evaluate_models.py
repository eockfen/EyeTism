"""
    functions to evaluate models and results
"""

# import libraries --------------------
import os
import math
import pickle
import pprint
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    RocCurveDisplay,
    fbeta_score,
    make_scorer,
)
from sklearn.model_selection import LearningCurveDisplay, ShuffleSplit


# saving a model --------------------
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


# evaluate a fitted gridsearchCV --------------------
def eval_grid_search(grid, X_test: np.asarray) -> np.array:
    """Given a fitted GridSearchCV model and a test set, the y_predictions of
    the best estimators are returned.

    Args:
        grid (model): fitted model
        X_test (np.asarray): test set

    Returns:
        np.array: predictions for test set
    """
    best_params = grid.best_params_
    best_model = grid.best_estimator_

    print("Best parameters:")
    pprint.PrettyPrinter(width=30).pprint(best_params)

    return best_model.predict(X_test)


# print some information about the model given --------------------
def model_info(model):
    """Displays some information about the (fitted!) model given.

    Args:
        model (model): Fitted (!) sklearn model.
    """

    # model infos
    type_ = str(type(model))

    if "DecisionTreeClassifier" in type_:
        tree_name = "Decision Tree"
        tree_depth = model.tree_.max_depth
        tree_nodes = model.tree_.node_count

        # how deep
        print(f"`{tree_name}` has:")
        print(f" -> {tree_nodes} nodes")
        print(f" -> maximum depth {tree_depth}")

    elif "RandomForestClassifier" in type_:
        print("current parameter:")
        pprint.PrettyPrinter(width=20).pprint(model.get_params())

        n_nodes = []
        max_depths = []

        for ind_tree in model.estimators_:
            n_nodes.append(ind_tree.tree_.node_count)
            max_depths.append(ind_tree.tree_.max_depth)

        tree_name = "Random Forest"
        tree_depth = int(np.mean(max_depths))
        tree_nodes = int(np.mean(n_nodes))
        txt_avg = "on average"

        # how deep
        print(
            f'"{tree_name}" has {tree_nodes} nodes {txt_avg} with \
                maximum depth {tree_depth} {txt_avg}.'
        )

    elif "RandomizedSearchCV" in type_:
        best_model = model.best_estimator_
        bm_type = str(type(best_model))

        # here i need to match/case different kinds of estimators

        print("best parameter:")
        pprint.PrettyPrinter(width=20).pprint(model.best_params_)

        # how deep?
        n_nodes = []
        max_depths = []

        for ind_tree in best_model.estimators_:
            n_nodes.append(ind_tree.tree_.node_count)
            max_depths.append(ind_tree.tree_.max_depth)

        tree_name = bm_type.split(".")[-1].split("'")[0]
        tree_depth = int(np.mean(max_depths))
        tree_nodes = int(np.mean(n_nodes))
        txt_avg = "on average"

        # how deep
        print(
            f'"{tree_name}" has {tree_nodes} nodes {txt_avg} with \
                maximum depth {tree_depth} {txt_avg}.'
        )


# create report for predictions --------------------
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

    # classification report
    if y_train is not None:
        print(line + " classification report for 'Train' " + line)
        print(classification_report(y_train, y_train_pred, digits=3))
        print(f"f(0.5)-score: {fbeta_score(y_train, y_train_pred, beta=0.5):.3f}")
        print(f"f(2.0)-score: {fbeta_score(y_train, y_train_pred, beta=2):.3f}\n")
    if y_test is not None:
        print(line + " classification report for 'Test' " + line)
        print(classification_report(y_test, y_test_pred, digits=3))
        print(f"f(0.5)-score: {fbeta_score(y_test, y_test_pred, beta=0.5):.3f}")
        print(f"f(2.0)-score: {fbeta_score(y_test, y_test_pred, beta=2):.3f}\n")

    # confusion matrix
    if y_train is not None or y_test is not None:
        plt.figure(figsize=(10, 4))

        if y_train is not None:  # train set
            plt.subplot(1, 2, 1)
            sns.heatmap(
                confusion_matrix(y_train, y_train_pred),
                annot=True,
                cmap=cm_cmap,
                fmt="g",
            )
            plt.title("Confusion Matrix for Train")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
        if y_test is not None:  # test set
            plt.subplot(1, 2, 2)
            sns.heatmap(
                confusion_matrix(y_test, y_test_pred), annot=True, cmap=cm_cmap, fmt="g"
            )
            plt.title("Confusion Matrix for Test")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")

        plt.tight_layout()
        plt.show()

    # ROC curve
    if y_train_proba is not None or y_test_proba is not None:
        plt.figure(figsize=(5, 5))
        ax = plt.gca()
        ax.set_aspect("equal", "box")

        if y_train_proba is not None:
            print(line * 3)
            print(
                f"'Train': ROC AUC score = {round(roc_auc_score(y_train, y_train_pred),3)}"
            )
            fpr, tpr, _ = roc_curve(y_train, y_train_pred)
            auc = round(roc_auc_score(y_train, y_train_pred), 3)
            RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc, estimator_name="Train").plot(
                ax
            )
        if y_test_proba is not None:
            print(
                f"'Test': ROC AUC score = {round(roc_auc_score(y_test, y_test_pred),3)}"
            )
            fpr, tpr, _ = roc_curve(y_test, y_test_pred)
            auc = round(roc_auc_score(y_test, y_test_pred), 3)
            RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc, estimator_name="Test").plot(
                ax
            )

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
        "train_sizes": np.linspace(0.1, 1.0, 5),
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
