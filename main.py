from io import BytesIO

import xgboost
# from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.tree import plot_tree
from xgboost import XGBClassifier
# import imputation
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.utils._bunch import Bunch
import matplotlib.pyplot as plt
import pickle
# import statsmodels.api as sm
import os
import pandas as pd

# os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz/bin/'
data_dir = r'data/'

def read_covtype_dataset(data_dir: str) -> Bunch :
    if os.path.isfile(data_dir +'covtype.pickle'):
        print("loading dataset from disk")
        with open(data_dir +'covtype.pkl', 'rb') as f:
            dataset = pickle.load(f)
    else:
        print("loading dataset from HTTP request")
        dataset = datasets.fetch_covtype(as_frame=True, random_state=1, shuffle=True)
        with open(data_dir +'covtype.pkl', "wb") as file:
            pickle.dump(dataset, file)

    return dataset

dataset = read_covtype_dataset(data_dir)
# df, utils = imputation.return_dataframe()
df = dataset.data
y = dataset.target

# describe data
print(f"total features = {len(df.columns)}")
print(f"features data types = {df.dtypes}")
print(f"ratio of target column {y.value_counts()}")


y = df['pulse'].values
y = y.astype(int)
X = df.drop('pulse', axis=1)

print(X.info())
print("X shape:\n", np.shape(X))
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=5, stratify=y)

with open('scores_ADASYN_5.pickle', 'rb') as handle:
    scores_ADASYN_5 = pickle.load(handle)

with open('scores.pickle', 'rb') as handle:
    scores = pickle.load(handle)

with open('scores_BorderLineSMOTE_5.pickle', 'rb') as handle:
    scores_BorderLineSMOTE_5 = pickle.load(handle)

with open('scores_SMOTE_5.pickle', 'rb') as handle:
    scores_SMOTE_5 = pickle.load(handle)

with open('scores_SVMSMOTE_5.pickle', 'rb') as handle:
    scores_SVMSMOTE_5 = pickle.load(handle)

# best hyper parameters from grid search in modelling
XGmodel = XGBClassifier(objective='binary:logistic', gamma=7, learning_rate=0.1, max_depth=5, min_child_weight=7, reg_alpha=0.5, reg_lambda=0.5)
rf = RandomForestClassifier(class_weight='balanced', criterion='gini', max_depth=5, min_samples_split=5, n_estimators=30, max_leaf_nodes=10)
models = [XGmodel, rf]

log_reg_df_train = x_train[['in_mechanic', 'et', 'embryoscope', 'frozen', 'Age', 'endomet']]
log_reg_df_test = x_test[['in_mechanic', 'et', 'embryoscope', 'frozen', 'Age', 'endomet']]

plot_auc_vs_features = True
over_sampling = False

if plot_auc_vs_features:
    for model in models:
        cv = StratifiedKFold(n_splits=5)
        scores_for_anova = {}
        empty_df = pd.DataFrame({})
        features = []
        importancess = []

        for key, item in scores[str(model)[:10]][7].items():
            features.append(key)
            importancess.append(item[0])

        if str(model).startswith("R"):
            features.append('10')
            importancess.append(0.001)

            features.append('in_social_prevention')
            importancess.append(0.01)

            features.append('in_prevention')
            importancess.append(0.027)
        else:
            features.append('10')
            importancess.append(0)

            features.append('in_social_prevention')
            importancess.append(0)

            features.append('in_prevention')
            importancess.append(0)

        empty_df['features'] = features
        empty_df['importance'] = importancess
        empty_df = empty_df.sort_values(by='importance')

        roc_aucs_train = []
        roc_aucs_test = []

        mean_trains = []
        std_trains = []
        mean_tests = []
        std_tests = []
        mean_vals = []
        std_vals = []

        for percent in range(10):
            quantile = empty_df['importance'].quantile(q=percent/10)
            new_df = empty_df[empty_df['importance'] >= quantile]
            new_x_train = x_train[new_df["features"]]
            new_x_test = x_test[new_df["features"]]
            model.fit(new_x_train, y_train)
            print(str(model)[:12])
            print(1-percent/10, "  --> total columns = ", len(new_x_train.columns),"over feature importance", quantile)
            print(new_x_train.columns)
            # mean_train, std_train, features_importance, auc_data_train = utils.show_cv_ROC_try_percentiles(model, cv, new_x_train,
            #                                                                    y_train, new_x_test, y_test, "train",
            #                                                                    str(model), over_sampling)
            mean_test, std_test, _, auc_data_test = utils.show_cv_ROC_try_percentiles(model, cv, new_x_train,
                                                           y_train, new_x_test, y_test, "validation",
                                                           str(model), over_sampling)
            mean_train, std_train, features_importance, auc_data_train = utils.show_cv_ROC_try_percentiles(model, cv,
                                                                                                           new_x_train,
                                                                                                           y_train,
                                                                                                           new_x_test,
                                                                                                           y_test,
                                                                                                           "special",
                                                                                                           str(model),
                                                                                                           over_sampling)
            mean_val, std_val, _, auc_data_val = utils.show_cv_ROC_try_percentiles(model, cv, new_x_train, y_train,
                                                                                    new_x_test, y_test,
                                                         "validation", str(model), over_sampling)

            roc_auc_train = utils.return_auc_score(new_x_train, y_train, model)
            roc_auc_test = utils.return_auc_score(new_x_test, y_test, model)

            roc_aucs_train.append(roc_auc_train)
            roc_aucs_test.append(roc_auc_test)

            mean_trains.append(mean_train)
            mean_tests.append(mean_test)
            mean_vals.append(mean_val)

            std_trains.append(std_train / np.sqrt(len(new_x_train.columns)))
            std_tests.append(std_test / np.sqrt(len(new_x_train.columns)))
            std_vals.append(std_val / np.sqrt(len(new_x_train.columns)))

            scores_for_anova["trains_{}".format(round(1 - percent/10, 1))] = auc_data_train
            scores_for_anova["tests_{}".format(round(1 - percent/10, 1))] = auc_data_test
            # scores_for_anova["vals_{}".format(round(1 - percent/10, 1))] = auc_data_val


        with open('scores_for_anova_{}_v3.csv'.format(str(model)[:10]), 'w') as file:
            for key in scores_for_anova.keys():
                file.write("%s.%s\n" % (key, scores_for_anova[key]))

        X_axis = np.arange(10)  # 10 quantiles
        fig, ax = plt.subplots(figsize=(6, 8), dpi=150)
        plt.bar(X_axis - 0.3, mean_trains, 0.3, label='Train', color='white', edgecolor='black', yerr=std_trains)
        plt.bar(X_axis, mean_tests, 0.3, label='Validation', color='grey', edgecolor='black' , yerr= std_tests)
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
        png1 = BytesIO()
        fig.savefig(png1, format='png')
        png2 = Image.open(png1)
        png2 = png2.convert("L")
        png2.save('feat_import_vs_auc_{}.tiff'.format(str(model)[:10]))
        png1.close()
        plt.show()

