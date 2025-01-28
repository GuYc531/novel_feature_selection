import os
import pickle
import zipfile
from typing import Tuple
import kagglehub

import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils._bunch import Bunch
from xgboost import XGBClassifier
import configs
from feature_selection import compute_feature_importance_by_percentile, plot_feature_importance_across_quantiles, \
    _compute_feature_importance_df

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


def plot_basic_feature_importance(model) -> None:
    feature_importance_df = _compute_feature_importance_df(model=model)
    feature_importance_df = feature_importance_df.iloc[:30]  # only plot top 30 features by importance desc
    plt.bar(feature_importance_df['feature_names'], feature_importance_df['feature_importance'])
    plt.title("Feature Importances")
    plt.xlabel("Features")
    plt.ylabel("Importance Score")
    plt.xticks(rotation=45)
    plt.show()


def initialize_data_set(small_db=False):
    dataset = read_covtype_dataset(data_dir)
    df = dataset.data
    y = dataset.target

    X = df.iloc[:1000] if small_db else df
    y = y[:1000] if small_db else y

    y = pd.Series([i - 1 for i in y])  # change labels from 1 to 7 to 0 to 6
    return X, y


def load_secom_data_set() -> Tuple[pd.DataFrame, pd.Series]:
    # Define the path to the zip file and the directory to extract to
    zip_file_path = 'data/secom.zip'
    extract_to_directory = 'data/secom/'

    if not os.path.exists(extract_to_directory):
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to_directory)

        print(f"Extracted files to {extract_to_directory}")

    X = pd.read_csv(f'{data_dir}secom/secom.data', delimiter=' ', header=None)
    y = pd.read_csv(f'{data_dir}secom/secom_labels.data', delimiter=' ', header=None).iloc[:, 0]
    y = pd.Series([i if i == 1 else 0 for i in y])
    return X, y


def load_madelon_data_set() -> Tuple[pd.DataFrame, pd.Series]:
    # Define the path to the zip file and the directory to extract to
    zip_file_path = 'data/madelon.zip'
    extract_to_directory = 'data/madelon/'

    if not os.path.exists(extract_to_directory):
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to_directory)

        print(f"Extracted files to {extract_to_directory}")

    X = pd.read_csv(f'{data_dir}madelon/MADELON/madelon_train.data', delimiter=' ', header=None)
    y = pd.read_csv(f'{data_dir}madelon/MADELON/madelon_train.labels', delimiter=' ', header=None).iloc[:, 0]
    y = pd.Series([i if i == 1 else 0 for i in y])
    return X, y


def fix_df_to_pipeline(X: pd.DataFrame) -> pd.DataFrame:
    # fix X column names
    X.columns = [f"feature_{i}" for i in range(X.shape[1])]

    # Fill NaN values with column means
    X = X.fillna(X.mean())
    return X


def load_smoking_drinking_dataset():
    path = kagglehub.dataset_download("sooyoungher/smoking-drinking-dataset", path='')

    print("Path to dataset files:", path)
    X = pd.read_csv(f"{path}/smoking_driking_dataset_Ver01.csv")
    y = X.iloc[:, -1].map({'Y': 1, 'N': 0})
    X = X.iloc[:, :-1]
    X['sex'] = X['sex'].map({'Male': 1, 'Female': 0})
    return X, y


X, y = load_smoking_drinking_dataset()
describe_dataset(X, y)

# choose your best hyperparameters based on grid search
xgboost_model = XGBClassifier(**configs.xgboost_hyperparametes)
random_forest_model = RandomForestClassifier(**configs.random_forest_hyperparameters)
decision_tree_model = DecisionTreeClassifier()

models = [xgboost_model,
          random_forest_model,
          decision_tree_model]

model = xgboost_model
quantiles_number = 5

stratified, shuffle, n_splits = True, True, 5
cv = StratifiedKFold(n_splits=n_splits, shuffle=shuffle) if stratified \
    else KFold(n_splits=n_splits, shuffle=shuffle)

metric = 'f1'
split_to_validation = True

for model in models:
    model.fit(X, y)
    plot_basic_feature_importance(model)
    x_axis_labels, scores, features_importance = compute_feature_importance_by_percentile(
        model=model,
        cv=cv,
        df=X,
        y=y,
        split_to_validation=split_to_validation,
        quantiles_number=quantiles_number,
        metric=metric
    )

    plot_feature_importance_across_quantiles(
        scores=scores,
        x_axis_labels=x_axis_labels,
        quantiles_number=quantiles_number,
        metric='f1',
        model=model,
        save_figure_in_path=True,
        path='plots/')
