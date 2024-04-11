# This is pure Code for 1 Run of the Best SVC Deepgaze Model

import os
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import set_config
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import classification_report
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier 

from xgboost import XGBClassifier

# import own modules
sys.path.append("..")  # Adds higher directory to python modules path.
from scripts import features as ft
from scripts import preprocessing as pp
from scripts import evaluate_models as em

# plt.style.use('https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-dark.mplstyle')
plt.style.use('https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-light.mplstyle')

import pickle

# metric
ftwo_scorer = make_scorer(fbeta_score, beta=2)

# defaults
RSEED = 42
cv = 10
n_jobs = -1
verbose = 1


# path to csv file
path_df = os.path.join("..", "data", "df_sam_resnet.csv")

# get features - or recalculate
recalculate_df = False
if os.path.isfile(path_df) and not recalculate_df:
    df = pd.read_csv(path_df)
else:
    df = ft.get_features()
    df.to_csv(path_df, index=False)

# set id as index
df = df.set_index("id", drop=True)

# drop first batch of useless variables
df = df.drop(columns=['img', 'sp_idx'])
df = df.drop(columns=[col for col in df.columns if "_obj" in col])  # drop 'object' columns

# processing
df = df[df["sp_fix_duration_ms_total"] <= 5000]
df = df.drop(columns=['sal_first_above_0.75*max_rank', 'sal_first_above_0.9*max_rank'])

# 11 Features List (optional)

feature_list = ["sp_fix_duration_ms_total","sp_fix_duration_ms_mean","sp_fix_duration_ms_var", "sal_first_fixation","sal_sum","sal_KLD", "obj_t_abs_on_background","obj_t_abs_on_animate", "obj_n_fix_background","obj_n_fix_inanimate","obj_n_fix_animate","asd"]

# 10 Features List same as above without "sp_fix_duration_ms_var"

# feature_list = ["sp_fix_duration_ms_total","sp_fix_duration_ms_mean", "sal_first_fixation","sal_sum","sal_KLD", "obj_t_abs_on_background","obj_t_abs_on_animate", "obj_n_fix_background","obj_n_fix_inanimate","obj_n_fix_animate","asd"]

# prepare features and target
X = df[feature_list]
y = X.pop("asd")

# train-test-split
X_train, X_test, y_train, y_test = pp.split(X, y)

# find numerical and categorical columns
num_cols = X_train.columns[X_train.dtypes != "object"]
cat_cols = X_train.columns[X_train.dtypes == "object"]

# The Column Transformer and Operations to do on Columns
# add other transformations at the end if needed
transformer = [("scaler", StandardScaler(), num_cols),
               ("ohe", OneHotEncoder(drop="first"), cat_cols  )]

# Add our transformer to a ColumnTransformer Object               
preprocessing = ColumnTransformer(transformer,
                                  remainder="passthrough")


# The Pipeline for the SVC Model
# Support Vector Classifier: apply scaling / encoding
svc_pipeline = Pipeline([
    ("preprocessor", preprocessing),
    ("classifier",SVC(probability=True))
])


# Support Vector Classifier Parameter Grid
param_grid_svc = {
    'classifier__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],  # Kernel types to try
    'classifier__C': [0.1, 1, 10, 100],  # Regularization parameter values
    'classifier__gamma': ['scale', 'auto'],  # Gamma parameter for RBF kernel
    'classifier__degree': [2, 3, 4]  # Degree of the polynomial kernel (only for poly kernel)
}

# Create GridSearchCV object
grid_search_svc = GridSearchCV(
    svc_pipeline,
    param_grid=param_grid_svc,
    cv=cv,
    scoring=ftwo_scorer,
    n_jobs=n_jobs,
    verbose=verbose,
)

# Fit the Model
grid_search_svc.fit(X_train, y_train)

# Get the best parameters and best estimator
best_params_svc = grid_search_svc.best_params_
best_est_svc = grid_search_svc.best_estimator_

# predict & proba
pred_test = grid_search_svc.predict(X_test)
proba_test = grid_search_svc.predict_proba(X_test)

pred_train = grid_search_svc.predict(X_train)
proba_train = grid_search_svc.predict_proba(X_train)

# Save the best estimator to a pickle file
with open('best_svc_resnet.pkl', 'wb') as file:
    pickle.dump(best_est_svc, file)