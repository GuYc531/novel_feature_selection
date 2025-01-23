from typing import Tuple, List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import BaseCrossValidator
from functools import wraps

from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from tqdm import tqdm

from feature_selection_utils import _compute_cv_metric_score, _compute_feature_importance_df, \
    _extract_model_name, _set_cross_validation_policy, _compute_quantile_and_relevant_df

valid_models = [RandomForestClassifier,
                DecisionTreeClassifier,
                XGBClassifier]

supported_metrics = ['f1', 'accuracy', 'recall', 'precision']

__all__ = ['compute_feature_importance_by_percentile', 'plot_feature_importance_across_quantiles']


def _check_is_model_fitted(method):
    @wraps(method)
    def wrapper(*args, **kwargs):
        if not hasattr(kwargs['model'], 'feature_importances_'):
            raise ValueError(f"The model must be fitted before calling `{method.__name__}`.")
        if not isinstance(kwargs['df'], pd.DataFrame):
            raise ValueError(f"train data (df) must be type of pd.DataFrame not {type(kwargs['df'])}")
        if not isinstance(kwargs['y'], pd.Series):
            raise ValueError(f"label data must be type of pd.Series not {type(kwargs['y'])}")
        if kwargs['metric'] not in supported_metrics:
            raise ValueError(f"Selected metric {kwargs['metric']} must be one of {[i for i in supported_metrics]}")
        return method(*args, **kwargs)

    return wrapper


def _validate_model(method):
    @wraps(method)
    def wrapper(model, *args, **kwargs):
        if not isinstance(model, tuple(valid_models)):
            raise TypeError(f"The model {model} must be one of {[type(i).__name__ for i in valid_models]}")
        return method(model, *args, **kwargs)

    return wrapper


@_validate_model
def _check_model_type(model):
    return model


@_check_is_model_fitted  # @_validate_inputs
def compute_feature_importance_by_percentile(cv: BaseCrossValidator,
                                             model,
                                             df: pd.DataFrame,
                                             y: pd.Series,
                                             split_to_validation: bool = False,
                                             quantiles_number: int = 5,
                                             evaluation_parts: List[str] = ['train', 'test'],
                                             metric: str = 'f1'
                                             ) -> Tuple[
    list[str], dict[str, list[np.floating]], pd.DataFrame]:
    model = _check_model_type(model)
    model_name = _extract_model_name(model=model)
    importance_df = _compute_feature_importance_df(model=model)
    cv = _set_cross_validation_policy(cv=cv)

    x_axis_labels, features_importance = list(), dict()
    mean_trains, std_trains, mean_tests, std_tests = list(), list(), list(), list()
    final_features_importance_df = pd.DataFrame()
    mean_validation, std_validation = (list(), list()) if split_to_validation else (None, None)

    print(f" model = {model_name}")

    for index, percent in tqdm(enumerate(range(quantiles_number)), desc="Processing Quantiles"):
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

        if split_to_validation:
            mean_validation.append(mean['test'])
            std_validation.append(std['test'])

        final_features_importance_df = pd.DataFrame.from_dict(features_importance) if index == 0 else pd.concat(
            [final_features_importance_df, pd.DataFrame.from_dict(features_importance)],
            ignore_index=True, axis=0)

    scores = {
        'mean_trains': mean_trains,
        'std_trains': std_trains,
        'mean_tests': mean_tests,
        'std_tests': std_tests
    }
    if split_to_validation:
        scores['mean_validation'] = mean_validation
        scores['std_validation'] = std_validation

    return x_axis_labels, scores, final_features_importance_df


def plot_feature_importance_across_quantiles(
        quantiles_number: int,
        scores: dict,
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
    plt.bar(x_axis - 0.3, scores['mean_trains'], 0.3, label='Train', color='white', edgecolor='black',
            yerr=scores['std_trains'])
    plt.bar(x_axis, scores['mean_tests'], 0.3, label='Test', color='grey', edgecolor='black', yerr=scores['std_tests'])
    plt.xticks(x_axis, x_axis_labels, fontsize=8)
    if 'mean_validation' in scores.keys():
        plt.bar(x_axis + 0.3, scores['mean_validation'], 0.3, label='Validation', color='silver', edgecolor='black',
                yerr=scores['std_validation'])

    plt.ylim(np.min([scores['mean_trains'], scores['mean_tests']]) - 0.1,
             np.max([scores['mean_trains'], scores['mean_tests']]) + 0.1) if 'mean_validation' in scores.keys() else \
        plt.ylim(np.min([scores['mean_trains'], scores['mean_tests'], scores['mean_validation']]) - 0.1,
                 np.max([scores['mean_trains'], scores['mean_tests'], scores['mean_validation']]) + 0.1)
    plt.xlabel("Feature number, Feature quantile", fontsize=12)
    plt.ylabel(metric, fontsize=16)
    plt.title(f"Mean {metric} score +/- std for model {model_name}")
    plt.legend()
    if save_figure_in_path:
        plt.savefig(f'{path}feature_selection_figure_{model_name}.png')
    plt.show()
    plt.close()
