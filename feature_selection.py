import os
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
    _extract_model_name, _set_cross_validation_policy, _compute_quantile_and_relevant_df, \
    _check_for_valid_quantiles_number

valid_models = [RandomForestClassifier,
                DecisionTreeClassifier,
                XGBClassifier]

supported_metrics = ['f1', 'accuracy', 'recall', 'precision']

__all__ = ['compute_feature_importance_by_percentile', 'plot_feature_importance_across_quantiles']


def _validate_inputs(method):
    """
    Decorator to validate the inputs of a method used in compute_feature_importance_by_percentile pipeline.

    This decorator ensures that the provided arguments meet the required conditions for
    executing the wrapped method. It checks the following:

    - The `model` argument must have the attributes `feature_importances_` and `feature_names_in_`.
    - The `df` argument must be of type `pd.DataFrame`.
    - The `y` argument must be of type `pd.Series`.
    - The `metric` argument must be one of the supported metrics.

    Parameters:
        method (callable): The method to be wrapped and validated.

    Returns:
        callable: The wrapped method with input validation.

    Raises:
        ValueError: If any of the following conditions are not met:
            - The `model` does not have a `feature_importances_` attribute.
            - The `model` does not have a `feature_names_in_` attribute.
            - The `df` is not of type `pd.DataFrame`.
            - The `y` is not of type `pd.Series`.
            - The `metric` is not in the list of supported metrics.

    Example:
        @validate_inputs
        def compute_feature_importance_by_percentile(*args, **kwargs):
            pass

        kwargs = {
            'model': trained_model,
            'df': dataframe,
            'y': labels,
            'metric': 'f1'
        }
        my_function(**kwargs)

    Notes:
        - The `supported_metrics` variable defined in the scope if the class as variable 'supported_metrics'
          containing a list of allowed metric names.
    """

    @wraps(method)
    def wrapper(*args, **kwargs):
        if not hasattr(kwargs['model'], 'feature_importances_'):
            raise ValueError(f"The model must be fitted before calling `{method.__name__}`.")
        if not hasattr(kwargs['model'], 'feature_names_in_'):
            raise ValueError(f"The model must be fitted with feature names before calling `{method.__name__}`.")
        if not isinstance(kwargs['df'], pd.DataFrame):
            raise ValueError(f"train data (df) must be type of pd.DataFrame not {type(kwargs['df'])}")
        if not isinstance(kwargs['y'], pd.Series):
            raise ValueError(f"label data must be type of pd.Series not {type(kwargs['y'])}")
        if kwargs['metric'] not in supported_metrics:
            raise ValueError(f"Selected metric {kwargs['metric']} must be one of {[i for i in supported_metrics]}")
        return method(*args, **kwargs)

    return wrapper


def _validate_model(method):
    """
        Decorator to validate the type of the `model` argument passed to a method.

        This decorator ensures that the `model` argument is an instance of one of the
        types specified in the `valid_models` list. If the `model` is not a valid type,
        it raises a `TypeError` with a descriptive error message.

        Args:
            method (Callable): The method to be wrapped and validated.

        Returns:
            Callable: The wrapped method with validation applied.

        Raises:
            TypeError: If `model` is not an instance of any type in `valid_models`.
    """

    @wraps(method)
    def wrapper(model, *args, **kwargs):
        if not isinstance(model, tuple(valid_models)):
            raise TypeError(f"The model {model} must be one of {[type(i).__name__ for i in valid_models]}")
        return method(model, *args, **kwargs)

    return wrapper


@_validate_model
def _check_model_type(model):
    """
      Validates and returns the given model.

      This function uses the `_validate_model` decorator to ensure that the
      `model` argument is an instance of one of the types specified in the
      `valid_models` list. If the validation passes, it simply returns the model.

      Args:
          model: The model to be validated and returned.

      Returns:
          The validated `model`.

      Raises:
          TypeError: If `model` is not an instance of any type in `valid_models`.
    """
    return model


@_validate_inputs
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
    """
    Computes feature importance, and performance metric scores at different percentiles of feature importance values.

    This function iteratively evaluates the impact of progressively filtering
    features by their importance percentiles, as determined by the model, on
    a given evaluation metric across specified cross-validation splits.

    Args:
        cv (BaseCrossValidator): The cross-validation splitting strategy.
        model: The machine learning model to evaluate, validated by `_check_model_type`,
         must be one of `valid_models`.
        df (pd.DataFrame): The input feature dataset.
        y (pd.Series): The target labels corresponding to the dataset.
        split_to_validation (bool, optional): Whether to include a validation set
            in addition to train and test splits. Defaults to False.
        quantiles_number (int, optional): The number of percentiles to split
            feature importance into. Defaults to 5.
        evaluation_parts (List[str], optional): Parts of the dataset to evaluate
            the metric on (e.g., `['train', 'test']`). Defaults to `['train', 'test']`.
        metric (str, optional): The evaluation metric to compute (e.g., `'f1'`, `'accuracy'`).
            Defaults to `'f1'`.

    Returns:
        Tuple[list[str], dict[str, list[np.floating]], pd.DataFrame]:
            - A list of x-axis labels indicating the number of features and
              the corresponding percentile.
            - A dictionary containing lists of mean and standard deviation
              of the metric for train, test, and optionally validation splits.
            - A DataFrame containing the feature importance data across all quantiles.

    Raises:
        ValueError: If invalid arguments are provided (e.g., unsupported metric, invalid dataset).
        TypeError: If the `model` is not a valid type as per `_check_model_type`.

    Example:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from sklearn.model_selection import KFold
        >>> import pandas as pd

        >>> df = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
        >>> y = pd.Series([0, 1, 0])
        >>> model = RandomForestClassifier()
        >>> cv = KFold(n_splits=3)

        >>> x_axis_labels, scores, final_features_importance_df = compute_feature_importance_by_percentile(
        ...     cv=cv,
        ...     model=model,
        ...     df=df,
        ...     y=y,
        ...     quantiles_number=3,
        ...     metric='accuracy'
        ... )
    """

    model = _check_model_type(model)
    model_name = _extract_model_name(model=model)
    importance_df = _compute_feature_importance_df(model=model)
    cv = _set_cross_validation_policy(cv=cv)
    _check_for_valid_quantiles_number(quantiles_number, df)

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


def _validate_path(path:str) -> None:
    if not os.path.exists(path):
        raise ValueError(f"Given path {path} does not exists in directory ")


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
        Plots feature importance scores across quantiles and visualizes the impact of feature selection on a specified metric.

        This function generates a bar plot displaying the mean and standard deviation of a metric
        (e.g., accuracy, F1 score) for train, test, and optionally validation splits, across feature importance quantiles.

        Args:
            quantiles_number (int): The number of quantiles used for feature importance filtering.
            scores (dict): A dictionary containing mean and standard deviation metric scores for
                'train', 'test', and optionally 'validation' splits. Keys include:
                - 'mean_trains': Mean metric scores for the training split.
                - 'std_trains': Standard deviation of metric scores for the training split.
                - 'mean_tests': Mean metric scores for the testing split.
                - 'std_tests': Standard deviation of metric scores for the testing split.
                - Optional: 'mean_validation' and 'std_validation' for validation scores.
            x_axis_labels (list): Labels for the x-axis, representing feature count and corresponding quantile.
            metric (str): The name of the evaluation metric being plotted (e.g., 'f1', 'accuracy').
            model: The machine learning model, used to extract the model name for the plot title and file name.
            save_figure_in_path (bool, optional): Whether to save the plot to a file. Defaults to False.
            path (str, optional): The directory path where the plot will be saved if `save_figure_in_path` is True.
                Defaults to an empty string.

        Returns:
            None: The function generates and displays a plot. If `save_figure_in_path` is True, it also saves the plot to the specified path.

        Example:
            >>> from sklearn.ensemble import RandomForestClassifier
            >>> import numpy as np

            >>> x_axis_labels, scores, final_features_importance_df = compute_feature_importance_by_percentile(
            ...     cv=cv,
            ...     model=model,
            ...     df=df,
            ...     y=y,
            ...     quantiles_number=3,
            ...     metric='accuracy'
            ... )
            >>> plot_feature_importance_across_quantiles(
            ...     quantiles_number=3,
            ...     scores=scores,
            ...     x_axis_labels=x_axis_labels,
            ...     metric='f1',
            ...     model=model,
            ...     save_figure_in_path=False
            ... )

        Notes:
            - The plot includes error bars (standard deviation) for the metric scores.
            - The x-axis labels represent the number of features and their importance quantiles.
            - Ensure the `scores` dictionary has the expected keys and values before calling the function.

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
        _validate_path(path)
        plt.savefig(f'{path}feature_selection_figure_{model_name}.png')
    plt.show()
    plt.close()
