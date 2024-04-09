import pickle
import os
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

def load_models(model_folder="testing", model_names=["xgb_model.pkl", "rf_model.pkl", "svc_model.pkl"]):
    models = dict()
    
    for model_name in model_names:
        model_path = os.path.join(model_folder, model_name)
        with open(model_path, 'rb') as f:
            model_key = model_name.split('.')[0]  # Extracting model name without extension
            models[model_key] = pickle.load(f)
    
    return models

def get_stacking(models, X_train, y_train):
    # Define the base models
    level0 = [
        ('xgb', models['xgb']),
        ('rf', models['rf']),
        ('svc', models['svc'])
    ]
    # Define meta learner model
    level1 = LogisticRegression()
    
    # Define the stacking ensemble
    stacking_model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
    
    # Fit the stacking model
    stacking_model.fit(X_train, y_train)
    return stacking_model

def stacking_pred_proba(stacking_model, X_train, X_test):
    # predict & proba
    pred_test = stacking_model.predict(X_test)
    proba_test = stacking_model.predict_proba(X_test)

    pred_train = stacking_model.predict(X_train)
    proba_train = stacking_model.predict_proba(X_train)

    return pred_train, proba_train, pred_test, proba_test