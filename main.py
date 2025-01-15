# from PIL import Image
# import statsmodels.api as sm
import os
import pickle

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

# os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz/bin/'
data_dir = r'data/'


def read_covtype_dataset(data_dir: str) -> Bunch:
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
    # describe data
    print(f"total features = {len(df.columns)}")
    print(f"total observations = {df.shape[0]}")
    print(f"features data types: \n {df.dtypes}")
    print(f"ratio of target column {y.value_counts()}")
    print(f"describe dataset: \n {df.describe()}")

def compute_feature_importance_for_fitted_model(model) -> pd.DataFrame:
    """
    Compute feature importance for fitted model.
    :param model: fitted model object based on tree model
    :return: pd.DataFrame of feature_names and feature_importance
    """
    importance_scores, feature_names = model.feature_importances_, model.feature_names_in_
    importance_df = pd.DataFrame({'feature_names': feature_names, 'feature_importance': importance_scores}).sort_values(
        by='feature_importance', ascending=True)

    return importance_df

def show_cv_ROC_try_percentiles(self, model, cv, X, Y, x_val, y_val, evaluation_part, model_name, ov):
    true_positive_rate_seires = []
    area_under_curve_series = []
    features_importances = {}
    mean_false_positive_rate = np.linspace(0, 1, 100)

    for i, (train, test) in enumerate(cv.split(X, Y)):
        model.fit(X.iloc[train].values, Y[train])

        features_importance_df = compute_feature_importance_for_fitted_model(model)
        features_importances[i] = features_importance_df['features_importance']

        pred = model.predict_proba(X.iloc[train].values)[:, 1]
        fpr, tpr, thresholds = roc_curve(Y[train], pred)
        roc_auc = auc(fpr, tpr)

        # if evaluation_part == "validation":
        #     pred = clf.predict_proba(x_val.values)[:, 1]
        #     fpr, tpr, thresholds = roc_curve(y_val, pred)
        #     roc_auc = auc(fpr, tpr)
        #
        # elif evaluation_part == "test":
        #     pred = clf.predict_proba(X.iloc[test].values)[:, 1]
        #     fpr, tpr, thresholds = roc_curve(Y[test], pred)
        #     roc_auc = auc(fpr, tpr)
        # elif evaluation_part == "special":
        #     pred = clf.predict_proba(X.values)[:, 1]
        #     fpr, tpr, thresholds = roc_curve(Y, pred)
        #     roc_auc = auc(fpr, tpr)
        #
        # else:
        #     pred = clf.predict_proba(X.iloc[train].values)[:, 1]
        #     fpr, tpr, thresholds = roc_curve(Y[train], pred)
        #     roc_auc = auc(fpr, tpr)


        inter_tpr = np.interp(mean_false_positive_rate, fpr, tpr)
        inter_tpr[0] = 0.0
        true_positive_rate_seires.append(inter_tpr)
        area_under_curve_series.append(roc_auc)

    mean_tpr = np.mean(true_positive_rate_seires, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_false_positive_rate, mean_tpr)
    std_auc = np.std(area_under_curve_series)
    return mean_auc, std_auc, features_importances, area_under_curve_series

def return_auc_score(test, labels, model):
    pred_test = model.predict_proba(test.values)[:, 1]
    fpr, tpr, thresholds = roc_curve(labels, pred_test)
    roc_auc_test = auc(fpr, tpr)
    preds = model.predict(test)

    return roc_auc_test
dataset = read_covtype_dataset(data_dir)
# df, utils = imputation.return_dataframe()
df = dataset.data
y = dataset.target

df = df.iloc[:1000]
y = y[:1000]

y = pd.Series([i - 1 for i in y])  # change labels from 1 to 7 to 0 to 6
describe_dataset(df, y)

# split to train test
x_train, x_test, y_train, y_test = train_test_split(df,
                                                    y,
                                                    test_size=0.1,
                                                    random_state=5,
                                                    stratify=y)

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
model.fit(x_train, y_train)

stratified = True
shuffle = True
n_splits = 5

cv = StratifiedKFold(n_splits=n_splits, shuffle=shuffle) if stratified \
    else KFold(n_splits=n_splits, shuffle=shuffle)

# First fit if not fitted yet
if not check_is_fitted(model):
    print("estimator is not fitted yet")

# features_importance_dict = dict()
# for i, (train, test) in enumerate(cv.split(df, y)):
#     xgboost_model.fit(df.iloc[train].values, y[train])
#     importance_df = compute_feature_importance_for_fitted_model(xgboost_model)
#     features_importance_dict[i] = importance_df['features_importance']

# empty_df = pd.DataFrame({})
importance_df = compute_feature_importance_for_fitted_model(model)

roc_aucs_train, roc_aucs_test = [], []
mean_trains, std_trains, mean_tests, std_tests, mean_vals, std_vals = [], [], [], [], [], []

# TODO: add check if quantile has features in it

for percent in range(quantiles_number):
    quantile = importance_df['feature_importance'].quantile(q=percent / quantiles_number)
    new_df = importance_df[importance_df['feature_importance'] >= quantile]
    new_x_train = x_train[new_df["feature_names"]]
    new_x_test = x_test[new_df["feature_names"]]
    # temp_model = model.copy()
    model.fit(new_x_train, y_train)
    print(str(type(model)).split('.')[-1])
    print(1 - percent / 10, "  --> total columns = ", len(new_x_train.columns), "over feature importance",
          quantile)

    # for i, (train, test) in enumerate(cv.split(df, y)):
    #     model.fit(df.iloc[train].values, y[train])
    y_train_predicted = model.predict(new_x_train)
    y_test_predicted = model.predict(new_x_test)

    # TODO: need to predict proba for roc_auc_score
    # roc_auc_train = roc_auc_score(y_train, y_train_predicted)
    # roc_auc_test = roc_auc_score(y_test, y_test_predicted)
    #
    # roc_aucs_train.append(roc_auc_train)
    # roc_aucs_test.append(roc_auc_test)

    # print(new_x_train.columns)
    # TODO: need to understand how this works
    mean_train, std_train, features_importance, auc_data_train = show_cv_ROC_try_percentiles(model, cv, new_x_train,
                                                                       y_train, new_x_test, y_test, "train",
                                                                       str(model), 'over_sampling')

    # mean_test, std_test, _, auc_data_test = utils.show_cv_ROC_try_percentiles(model, cv, new_x_train,
    #                                                                           y_train, new_x_test, y_test,
    #                                                                           "validation",
    #                                                                           str(model), over_sampling)
    # mean_train, std_train, features_importance, auc_data_train = utils.show_cv_ROC_try_percentiles(model, cv,
    #                                                                                                new_x_train,
    #                                                                                                y_train,
    #                                                                                                new_x_test,
    #                                                                                                y_test,
    #                                                                                                "special",
    #                                                                                                str(model),
    #                                                                                                over_sampling)
    # mean_val, std_val, _, auc_data_val = utils.show_cv_ROC_try_percentiles(model, cv, new_x_train, y_train,
    #                                                                        new_x_test, y_test,
    #                                                                        "validation", str(model),
    #                                                                        over_sampling)

    # roc_auc_train = utils.return_auc_score(new_x_train, y_train, model)
    # roc_auc_test = utils.return_auc_score(new_x_test, y_test, model)
    #
    # roc_aucs_train.append(roc_auc_train)
    # roc_aucs_test.append(roc_auc_test)

    mean_trains.append(mean_train)
    mean_tests.append(mean_test)
    mean_vals.append(mean_val)

    std_trains.append(std_train / np.sqrt(len(new_x_train.columns)))
    std_tests.append(std_test / np.sqrt(len(new_x_train.columns)))
    std_vals.append(std_val / np.sqrt(len(new_x_train.columns)))

    scores_for_anova["trains_{}".format(round(1 - percent / 10, 1))] = auc_data_train
    scores_for_anova["tests_{}".format(round(1 - percent / 10, 1))] = auc_data_test
    # scores_for_anova["vals_{}".format(round(1 - percent/10, 1))] = auc_data_val

# with open('scores_for_anova_{}_v3.csv'.format(str(model)[:10]), 'w') as file:
#     for key in scores_for_anova.keys():
#         file.write("%s.%s\n" % (key, scores_for_anova[key]))

X_axis = np.arange(10)  # 10 quantiles
fig, ax = plt.subplots(figsize=(6, 8), dpi=150)
plt.bar(X_axis - 0.3, mean_trains, 0.3, label='Train', color='white', edgecolor='black', yerr=std_trains)
plt.bar(X_axis, mean_tests, 0.3, label='Validation', color='grey', edgecolor='black', yerr=std_tests)
if max(roc_aucs_train) > 0.8:
    plt.ylim([0.5, 0.95])
else:
    plt.ylim([0.5, 0.75])
plt.xticks(X_axis, np.round(np.linspace(1, 0, 11), 2))
plt.xlabel("Percent of features by importance", fontsize=16)
plt.ylabel("ROC AUC", fontsize=16)
plt.title("Mean ROC AUC ")
plt.legend()
plt.show()
# png1 = BytesIO()
# fig.savefig(png1, format='png')
# png2 = Image.open(png1)
# png2 = png2.convert("L")
# png2.save('feat_import_vs_auc_{}.tiff'.format(str(model)[:10]))
# png1.close()
# plt.show()
