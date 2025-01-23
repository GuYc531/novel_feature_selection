from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, BaseCrossValidator


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


def _extract_model_name(model) -> str:
    return type(model).__name__


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


def _append_feature_importance_dict(features_importance_df: pd.DataFrame,
                                    features_importance_dict: dict = {},
                                    new: bool = False) -> dict[str, list]:
    if new:
        features_importance_dict = pd.Series([[float(i)] for i in features_importance_df.feature_importance.values],
                                             index=features_importance_df.feature_names).to_dict()
    else:
        for index, row in features_importance_df.iterrows():
            features_importance_dict[str(row['feature_names'])].append(row['feature_importance'])

    return features_importance_dict


def _compute_cv_metric_score(percentile_x: pd.DataFrame,
                             y: pd.Series,
                             evaluation_parts: list,
                             cv,
                             model,
                             split_to_validation: bool = False,
                             validation_size: float = 0.2,
                             metric: str = 'f1') -> Tuple[
    dict[str, np.floating], dict[str, np.floating], dict[str, list]]:
    """

    :param percentile_x:
    :return:
    """
    multi_label = True if len(np.unique(y)) > 2 else False

    metric_list_train, metric_list_test = list(), list()
    if split_to_validation:
        percentile_x, x_validation, y, y_validation = train_test_split(percentile_x,
                                                                       y,
                                                                       test_size=validation_size,
                                                                       random_state=5,
                                                                       stratify=y)
        metric_list_validation = []

    for i, (train, test) in enumerate(cv.split(percentile_x, y)):
        # fit new df and compute feature  importance
        model.fit(percentile_x.iloc[train], y.iloc[train])
        features_importance_df = _compute_feature_importance_df(model=model)
        features_importance = _append_feature_importance_dict(features_importance_df=features_importance_df,
                                                              features_importance_dict=features_importance) if i > 0 \
            else _append_feature_importance_dict(features_importance_df=features_importance_df, new=True)

        # compute score
        for evaluation_part in evaluation_parts:
            evaluation_indexes = train if evaluation_part == 'train' else test

            predictions = model.predict(percentile_x.iloc[evaluation_indexes])
            metric_score = _compute_prediction_metric(y_true=y.iloc[evaluation_indexes],
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
                metric_score = _compute_prediction_metric(y_true=y_validation,
                                                          y_predictions=predictions,
                                                          multi_label=multi_label,
                                                          metric=metric)
                metric_list_validation.append(metric_score)
            except NameError:
                print(f" x_validation or y_validation or metric_list_validation are not defined")

    metric_mean_score = {'train': np.mean(metric_list_train),
                         'test': np.mean(metric_list_test)}
    metric_std_score = {'train': np.std(metric_list_train),
                        'test': np.std(metric_list_test)}

    if split_to_validation:
        try:
            metric_mean_score['validation'] = np.mean(metric_list_validation)
            metric_std_score['validation'] = np.std(metric_list_validation)
        except NameError:
            print(f"metric_list_validation is not defined")

    return metric_mean_score, metric_std_score, features_importance


# TODO: need to validate if quantile number is not big for the splits
def _check_for_valid_quantiles_number(quantiles_number):
    return quantiles_number
