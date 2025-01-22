from typing import Tuple, List, Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.constants import minute
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split, BaseCrossValidator
from functools import wraps

from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from tqdm import tqdm

valid_models = [RandomForestClassifier,
                DecisionTreeClassifier,
                XGBClassifier]

__all__ = ['compute_feature_importance_by_percentile', 'plot_feature_importance_across_quantiles']


def _check_is_model_fitted(method):
    @wraps(method)
    def wrapper(*args, **kwargs):
        if not hasattr(kwargs['model'], 'feature_importances_'):
            raise ValueError(f"The model must be fitted before calling `{method.__name__}`.")
        return method(*args, **kwargs)

    return wrapper


def _validate_model(method):
    @wraps(method)
    def wrapper(model, *args, **kwargs):
        if not isinstance(model, tuple(valid_models)):
            raise TypeError(f"The model {model} must be one of {[type(i).__name__ for i in valid_models]}")
        return method(model, *args, **kwargs)

    return wrapper


def _set_cross_validation_policy(cv: BaseCrossValidator) -> BaseCrossValidator:
    if cv is not None and (isinstance(cv, StratifiedKFold) or isinstance(cv, KFold)):
        return cv

    stratified = True
    shuffle = True
    n_splits = 5
    cv = StratifiedKFold(n_splits=n_splits, shuffle=shuffle) if stratified \
        else KFold(n_splits=n_splits, shuffle=shuffle)
    print(f"not given cross validation option , selected one is {cv.__name__}")

    return cv


def _compute_quantile_and_relevant_df(df: pd.DataFrame, importance_df: pd.DataFrame, percent: float,
                                      quantiles_number: int) \
        -> Tuple[float, pd.DataFrame]:
    """

    :param df:
    :param importance_df:
    :param percent:
    :param quantiles_number:
    :return:
    """

    quantile = importance_df['feature_importance'].quantile(q=percent / quantiles_number)
    percentile_features_df = importance_df[importance_df['feature_importance'] >= quantile]
    percentile_x = df[percentile_features_df["feature_names"]]

    return quantile, percentile_x


def plot_feature_importance_across_quantiles(
        quantiles_number: int,
        mean_trains: list,
        std_trains: list,
        mean_tests: list,
        std_tests: list,
        x_axis_labels: list,
        metric: str,
        model,
        save_figure_in_path: bool = False,
        path: str = ''
) -> None:
    """

    :param save_figure_in_path:
    :param path:
    :return:
    """
    model_name = _extract_model_name(model=model)
    x_axis = np.arange(quantiles_number)
    fig = plt.figure()
    plt.bar(x_axis - 0.3, mean_trains, 0.3, label='Train', color='white', edgecolor='black',
            yerr=std_trains)
    plt.bar(x_axis, mean_tests, 0.3, label='Test', color='grey', edgecolor='black', yerr=std_tests)
    plt.xticks(x_axis, x_axis_labels, fontsize=8)
    plt.ylim(np.min([mean_trains, mean_tests]) - 0.1, np.max([mean_trains, mean_tests]) + 0.1)
    plt.xlabel("Feature number, Feature quantile", fontsize=12)
    plt.ylabel(metric, fontsize=16)
    plt.title(f"Mean {metric} score +/- std for model {model_name}")
    plt.legend()
    if save_figure_in_path:
        plt.savefig(f'{path}feature_selection_figure_{model_name}.png')
    plt.show()
    plt.close()


def _extract_model_name(model) -> str:
    return type(model).__name__


@_check_is_model_fitted
def compute_feature_importance_by_percentile(cv: BaseCrossValidator,
                                             model,
                                             df: pd.DataFrame,
                                             y: pd.Series,
                                             split_to_validation: bool = False,
                                             quantiles_number: int = 5,
                                             evaluation_parts: List[str] = ['train', 'test'],
                                             metric: str = 'f1'
                                             ) -> Tuple[
    list[np.floating], list[np.floating], list[np.floating], list[np.floating], list[np.floating], dict[int, Any]]:
    model = _check_model_type(model)
    model_name = _extract_model_name(model=model)
    importance_df = _compute_feature_importance_df(model=model)
    cv = _set_cross_validation_policy(cv=cv)

    x_axis_labels, features_importance = list(), dict()
    mean_trains, std_trains, mean_tests, std_tests = list(), list(), list(), list()

    print(f" model = {model_name}")

    for percent in tqdm(range(quantiles_number), desc="Processing Quantiles"):
        # get percentile and create new df after filtering

        quantile, percentile_x = _compute_quantile_and_relevant_df(df=df,
                                                                   importance_df=importance_df,
                                                                   percent=percent,
                                                                   quantiles_number=quantiles_number)

        model.fit(percentile_x, y)
        print(
            f"""{round(1 - percent / quantiles_number, 3)}, --> total columns =  {len(percentile_x.columns)}, over feature importance {round(quantile, 3)}""")

        x_axis_labels.append(f'{len(percentile_x.columns)}, {round(quantile, 2)}')

        mean, std, features_importance = _compute_cv_metric_score(percentile_x=percentile_x,
                                                                  y=y,
                                                                  evaluation_parts=evaluation_parts,
                                                                  cv=cv,
                                                                  model=model,
                                                                  metric=metric,
                                                                  split_to_validation=split_to_validation)

        mean_trains.append(mean['train'])
        std_trains.append(std['train'])

        mean_tests.append(mean['test'])
        std_tests.append(std['test'])

    return x_axis_labels, mean_trains, std_trains, mean_tests, std_tests, features_importance


def _compute_feature_importance_df(model) -> pd.DataFrame:
    """
    Compute feature importance for fitted model.
    :return: pd.DataFrame of feature_names and feature_importance
    """
    importance_scores, feature_names = model.feature_importances_, model.feature_names_in_
    importance_df = pd.DataFrame({'feature_names': feature_names,
                                  'feature_importance': importance_scores}).sort_values(
        by='feature_importance', ascending=True)

    return importance_df


def _compute_prediction_metric(y_true: np.array,
                               y_predictions: np.array,
                               multi_label: bool = False,
                               metric: str = 'f1') -> float:
    """

    :param y_true:
    :param y_predictions:
    :return:
    """

    average = 'weighted' if multi_label else 'binary'
    if metric == 'accuracy':
        return accuracy_score(y_true=y_true, y_pred=y_predictions)
    elif metric == 'precision':
        return precision_score(y_true=y_true, y_pred=y_predictions, average=average)
    elif metric == 'recall':
        return recall_score(y_true=y_true, y_pred=y_predictions, average=average)
    elif metric == 'f1':
        return f1_score(y_true=y_true, y_pred=y_predictions, average=average)
    else:
        raise ValueError(f"Selected metric {metric} should one of accuracy, precision, recall, f1")


def _compute_cv_metric_score(percentile_x: pd.DataFrame,
                             y: pd.Series,
                             evaluation_parts: list,
                             cv,
                             model,
                             split_to_validation: bool = False,
                             validation_size: float = 0.2,
                             metric: str = 'f1') -> Tuple[
    dict[str, np.floating], dict[str, np.floating], dict[int, any]]:
    """

    :param percentile_x:
    :return:
    """
    multi_label = True if len(np.unique(y)) > 2 else False

    metric_list_train, metric_list_test, features_importance = list(), list(), dict()
    if split_to_validation:
        percentile_x, x_validation, y, y_validation = train_test_split(x=percentile_x,
                                                                       y=y,
                                                                       test_size=validation_size,
                                                                       random_state=5,
                                                                       stratify=y)
        metric_list_validation = []

    for i, (train, test) in enumerate(cv.split(percentile_x, y)):
        # fit new df and compute feature  importance
        model.fit(percentile_x.iloc[train], y[train])
        features_importance_df = _compute_feature_importance_df(model=model)
        features_importance[i] = features_importance_df['feature_importance']

        # TODO: return the mean and std for all the feature importance

        # compute score
        for evaluation_part in evaluation_parts:
            evaluation_indexes = train if evaluation_part == 'train' else test

            predictions = model.predict(percentile_x.iloc[evaluation_indexes])
            metric_score = _compute_prediction_metric(y_true=y[evaluation_indexes],
                                                      y_predictions=predictions,
                                                      multi_label=multi_label,
                                                      metric=metric)

            if evaluation_part == 'train':
                metric_list_train.append(metric_score)
            else:
                metric_list_test.append(metric_score)

        if split_to_validation:
            try:
                predictions = model.predict(x_validation)
                metric_score = _compute_prediction_metric(y_validation, predictions)
                metric_list_validation.append(metric_score)
            except NameError:
                print(f" x_validation or y_validation or metric_list_validation are not defined")

    metric_mean_score = {'train': np.mean(metric_list_train),
                         'test': np.mean(metric_list_test)}
    metric_std_score = {'train': np.std(metric_list_train),
                        'test': np.std(metric_list_test)}

    if split_to_validation:
        try:
            metric_mean_score = {'validation': np.mean(metric_list_validation)}
            metric_std_score = {'validation': np.std(metric_list_validation)}
        except NameError:
            print(f"metric_list_validation is not defined")

    return metric_mean_score, metric_std_score, features_importance


@_validate_model
def _check_model_type(model):
    return model


# TODO: need to validate if quantile number is not big for the splits
def _check_for_valid_quantiles_number(quantiles_number):
    return quantiles_number
