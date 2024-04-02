# -----------------------------------------------------------------------------
import os
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
    RocCurveDisplay
)
from sklearn.metrics import fbeta_score


# --------------------------------------------------------------------------------------------------
def save_model(m, f: str, overwrite=False):
    """Saves models into "models" folder. If filename already exists, a number
    will be added, if overwrite flag is not set to True.

    Args:
        m (sklearn_model): model to be saved
        f (str): filename
        overwrite (bool): flag to indicate if model can be overwritten.
                            Defaults to False.
    """
    # path & filename
    path = os.path.join("models", f"{f}")
    filename, extension = os.path.splitext(path)

    # add suffix if neccessary
    counter = 1
    while os.path.exists(path) and not overwrite:
        path = filename + "_" + str(counter) + extension
        counter += 1

    # save as pickle
    print(f'saving model in: "{path}"')
    pickle.dump(m, open(path, "wb"))


# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
def report(
    y_train=None,
    y_train_pred=None,
    y_train_proba=None,
    y_test=None,
    y_test_pred=None,
    y_test_proba=None,
    model=None,
):
    """outputs classification results, containing 'classification report", a
    "confusion matrix", and infos about model if able to  identify if. Results
    separately available for train and test set.

    Args:
        y_train (np.array, optional): Train set. Defaults to None.
        y_train_pred (np.array, optional): Train set predictions. Defaults to None.
        y_train_proba (np.array, optional): Train set probabilities. Defaults to None.
        y_test (np.array, optional): Test set. Defaults to None.
        y_test_pred (np.array, optional): Test set predictions. Defaults to None.
        y_test_proba (np.array, optional): Test set probabilities. Defaults to None.
        model (model, optional): Fitted sklearn model. Defaults to None.
    """
    line = "-" * 15
    cm_cmap = sns.light_palette("seagreen", as_cmap=True)

    # model infos
    if model is not None:
        type_ = str(type(model))

        if "DecisionTreeClassifier" in type_:
            tree_name = "Decision Tree"
            tree_depth = model.tree_.max_depth
            tree_nodes = model.tree_.node_count
            txt_avg = ""

            # how deep
            print(
                f'"{tree_name}" has {tree_nodes} nodes {txt_avg} with \
                    maximum depth {tree_depth} {txt_avg}.'
            )

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

    # confusion matrix
    plt.figure(figsize=(10, 4))

    if y_train is not None:  # train set
        plt.subplot(1, 2, 1)
        sns.heatmap(
            confusion_matrix(y_train, y_train_pred), annot=True, cmap=cm_cmap, fmt="g"
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

    # classification report
    if y_train is not None:
        print(line + " classification report for Train " + line)
        print(classification_report(y_train, y_train_pred, digits=3))
        print(f"f(0.5)-score: {fbeta_score(y_train, y_train_pred, beta=0.5):.3f}")
        print(f"f(2.0)-score: {fbeta_score(y_train, y_train_pred, beta=2):.3f}\n")
    if y_test is not None:
        print(line + " classification report for Test " + line)
        print(classification_report(y_test, y_test_pred, digits=3))
        print(f"f(0.5)-score: {fbeta_score(y_test, y_test_pred, beta=0.5):.3f}")
        print(f"f(2.0)-score: {fbeta_score(y_test, y_test_pred, beta=2):.3f}\n")

    # ROC curve
    if y_train_proba is not None or y_test_proba is not None:
        plt.figure(figsize=(5, 5))
        ax = plt.gca()
        ax.set_aspect("equal", "box")

        if y_train_proba is not None:
            print(f"'train': ROC AUC Score = {roc_auc_score(y_train, y_train_pred)}")
            fpr, tpr, _ = roc_curve(y_train, y_train_pred)
            auc = round(roc_auc_score(y_train, y_train_pred), 3)
            RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc, estimator_name='Train').plot(ax)
        if y_test_proba is not None:
            print(f"'test': ROC AUC Score = {roc_auc_score(y_test, y_test_pred)}")
            fpr, tpr, _ = roc_curve(y_test, y_test_pred)
            auc = round(roc_auc_score(y_test, y_test_pred), 3)
            RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc, estimator_name='Test').plot(ax)

        plt.show()
