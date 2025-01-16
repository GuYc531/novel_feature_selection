# from PIL import Image
# import statsmodels.api as sm
import os
import pickle
import sys
from typing import Tuple

import matplotlib.pyplot as plt
# import imputation
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils._bunch import Bunch
from sklearn.utils.validation import check_is_fitted
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, \
    roc_curve, roc_auc_score, log_loss, auc, RocCurveDisplay
import configs
from feature_selection import feature_selection_by_importance

# os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz/bin/'
data_dir = r'data/'


def read_covtype_dataset(data_dir: str) -> Bunch:
    """

    :param data_dir:
    :return:
    """
    if os.path.isfile(data_dir + 'covtype.pkl'):
        print("loading dataset from disk")
        with open(data_dir + 'covtype.pkl', 'rb') as f:
            dataset = pickle.load(f)
    else:
        print("loading dataset from HTTP request")
        dataset = datasets.fetch_covtype(as_frame=True, random_state=1, shuffle=True)
        with open(data_dir + 'covtype.pkl', "wb") as file:
            pickle.dump(dataset, file)

    return dataset


def describe_dataset(df: pd.DataFrame, y: pd.Series) -> None:
    """

    :param df:
    :param y:
    :return:
    """
    # describe data
    print(f"total features = {len(df.columns)}")
    print(f"total observations = {df.shape[0]}")
    print(f"features data types: \n {df.dtypes}")
    print(f"ratio of target column {y.value_counts()}")
    print(f"describe dataset: \n {df.describe()}")


def return_auc_score(test, labels, model):
    """

    :param test:
    :param labels:
    :param model:
    :return:
    """
    pred_test = model.predict_proba(test.values)[:, 1]
    fpr, tpr, thresholds = roc_curve(labels, pred_test)
    roc_auc_test = auc(fpr, tpr)
    preds = model.predict(test)

    return roc_auc_test


dataset = read_covtype_dataset(data_dir)
# df, utils = imputation.return_dataframe()
df = dataset.data
y = dataset.target

X = df.iloc[:1000]
y = y[:1000]

y = pd.Series([i - 1 for i in y])  # change labels from 1 to 7 to 0 to 6
describe_dataset(df, y)

# split to train test
# x_train, x_test, y_train, y_test = train_test_split(df,
#                                                     y,
#                                                     test_size=0.1,
#                                                     random_state=5,
#                                                     stratify=y)

# choose your best hyperparameters based on grid search
xgboost_model = XGBClassifier(**configs.xgboost_hyperparametes)
random_forest_model = RandomForestClassifier(**configs.random_forest_hyperparameters)
decision_tree_model = DecisionTreeClassifier()

models = [xgboost_model,
          random_forest_model,
          decision_tree_model]

model = random_forest_model
scores = ['roc-auc']
quantiles_number = 10

# for model in models:
# first we fit the model with all features to get general feature importance
model.fit(X, y)

stratified = True
shuffle = True
n_splits = 5

cv = StratifiedKFold(n_splits=n_splits, shuffle=shuffle) if stratified \
    else KFold(n_splits=n_splits, shuffle=shuffle)

# First fit if not fitted yet
# if not check_is_fitted(model):
#     print("estimator is not fitted yet")
#     sys.exit(0)

# features_importance_dict = dict()
# for i, (train, test) in enumerate(cv.split(df, y)):
#     xgboost_model.fit(df.iloc[train].values, y[train])
#     importance_df = compute_feature_importance_for_fitted_model(xgboost_model)
#     features_importance_dict[i] = importance_df['features_importance']

# empty_df = pd.DataFrame({})


# TODO: add check if quantile has features in it
metric = 'accuracy'

feature_class = feature_selection_by_importance(X, y, metric, quantiles_number, cv, model)
feature_class.compute_feature_importance_by_percentile()
feature_class.plot_feature_importance_across_quantiles(save_figure_in_path=True)