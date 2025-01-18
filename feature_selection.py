from typing import Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss
from sklearn.model_selection import StratifiedKFold, KFold
from functools import wraps

from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

valid_models = [RandomForestClassifier,
                DecisionTreeClassifier,
                XGBClassifier]


def check_is_model_fitted(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        if not getattr(self.model, 'feature_importances_', False).any():
            raise ValueError(f"The model must be fitted before calling `{method.__name__}`.")
        return method(self, *args, **kwargs)

    return wrapper


def validate_model(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        if not isinstance(args[0], tuple(valid_models)):
            raise TypeError(f"The model {args[0]} must be one of {[type(i).__name__ for i in valid_models]}")
        return method(self, *args, **kwargs)

    return wrapper


class FeatureSelectionByImportance:
    def __init__(self, df: pd.DataFrame,
                 y: pd.Series,
                 metric: str,
                 quantiles_number: int,
                 model,
                 cv=None):
        """

        :param df:
        :param y:
        :param metric:
        :param quantiles_number:
        :param cv:
        :param model:
        """
        self.df = df
        self.y = y
        self.multi_label = True if len(np.unique(y)) > 2 else False

        self.quantiles_number = self.check_for_valid_quantiles_number(quantiles_number)

        self.model = self.check_model_type(model)
        # self.model_name = str(type(self.model)).split('.')[-1].split("'")[0]
        self.model_name = type(model).__name__
        self.metric = metric
        self.cv = self.set_cross_validation_policy(cv)
        self.X_axis_labels = []
        self.mean_trains, self.std_trains, self.mean_tests, self.std_tests = [], [], [], []
        self.evaluation_parts = ['train', 'test']

    @staticmethod
    def set_cross_validation_policy(cv):

        if cv is not None and (isinstance(cv, StratifiedKFold) or isinstance(cv, KFold)):
            return cv

        stratified = True
        shuffle = True
        n_splits = 5

        cv = StratifiedKFold(n_splits=n_splits, shuffle=shuffle) if stratified \
            else KFold(n_splits=n_splits, shuffle=shuffle)
        return cv

    def compute_quantile_and_rellevant_df(self, importance_df: pd.DataFrame, percent: float) \
            -> Tuple[float, pd.DataFrame]:
        """

        :param importance_df:
        :param percent:
        :return:
        """
        quantile = importance_df['feature_importance'].quantile(q=percent / self.quantiles_number)
        percentile_features_df = importance_df[importance_df['feature_importance'] >= quantile]
        percentile_x = self.df[percentile_features_df["feature_names"]]

        return quantile, percentile_x

    def plot_feature_importance_across_quantiles(self, save_figure_in_path=False, path='') -> None:
        """

        :param save_figure_in_path:
        :param path:
        :return:
        """
        x_axis = np.arange(self.quantiles_number)
        fig = plt.figure()
        plt.bar(x_axis - 0.3, self.mean_trains, 0.3, label='Train', color='white', edgecolor='black',
                yerr=self.std_trains)
        plt.bar(x_axis, self.mean_tests, 0.3, label='Test', color='grey', edgecolor='black', yerr=self.std_tests)
        plt.xticks(x_axis, self.X_axis_labels, fontsize=8)
        plt.xlabel("Feature number, Feature quantile", fontsize=12)
        plt.ylabel(self.metric, fontsize=16)
        plt.title(f"Mean {self.metric} score +/- std for model {self.model_name}")
        plt.legend()
        if save_figure_in_path:
            plt.savefig(f'{path}feature_selection_figure.png')
        plt.show()
        plt.close()

    def compute_feature_importance_by_percentile(self) -> None:

        importance_df = self.compute_feature_importance_df()
        for percent in range(self.quantiles_number):
            # get percentile and create new df after filtering

            quantile, percentile_x = self.compute_quantile_and_rellevant_df(importance_df, percent)
            self.model.fit(percentile_x, self.y)
            print(f" model = {self.model_name}")
            print(
                f"""{round(1 - percent / self.quantiles_number, 3)}, --> total columns =  {len(percentile_x.columns)}, over feature importance {round(quantile, 3)}""")
            self.X_axis_labels.append(f'{len(percentile_x.columns)}, {round(quantile, 2)}')

            # for flag in self.flags:
            mean, std, features_importances = self.compute_cv_metric_score(percentile_x)

            self.mean_trains.append(mean['train'])
            self.std_trains.append(std['train'])

            self.mean_tests.append(mean['test'])
            self.std_tests.append(std['test'])


    @check_is_model_fitted
    def compute_feature_importance_df(self) -> pd.DataFrame:
        """
        Compute feature importance for fitted model.
        :return: pd.DataFrame of feature_names and feature_importance
        """
        importance_scores, feature_names = self.model.feature_importances_, self.model.feature_names_in_
        importance_df = pd.DataFrame({'feature_names': feature_names,
                                      'feature_importance': importance_scores}).sort_values(
            by='feature_importance', ascending=True)

        return importance_df


    def compute_prediction_metric(self, y_true: np.array, y_predictions: np.array) -> float:
        """

        :param y_true:
        :param y_predictions:
        :return:
        """

        average = 'weighted' if self.multi_label else 'binary'
        if self.metric == 'accuracy':
            return accuracy_score(y_true=y_true, y_pred=y_predictions)
        elif self.metric == 'precision':
            return precision_score(y_true=y_true, y_pred=y_predictions, average=average)
        elif self.metric == 'recall':
            return recall_score(y_true=y_true, y_pred=y_predictions, average=average)
        elif self.metric == 'f1':
            return f1_score(y_true=y_true, y_pred=y_predictions, average=average)
        # elif self.metric == 'log_loss':
        #     return log_loss(y_true=y_true, y_pred=y_predictions)
        else:
            raise ValueError(f"Selected metric {self.metric} should one of accuracy, precision, recall, f1")


    def compute_cv_metric_score(self, percentile_x: pd.DataFrame) -> Tuple[
        dict[str, np.floating], dict[str,np.floating], dict[int, any]]:
        """

        :param percentile_x:
        :return:
        """

        metric_list_train = []
        metric_list_test = []
        features_importances = {}

        for i, (train, test) in enumerate(self.cv.split(percentile_x, self.y)):
            # evaluation_part = train if flag == 'train' else test

            # fit new df and compute feature  importance
            self.model.fit(percentile_x.iloc[train], self.y[train])
            features_importance_df = self.compute_feature_importance_df()
            features_importances[i] = features_importance_df['feature_importance']

            # TODO: return the mean and std for all the feature importance
            # compute score
            for evaluation_part in self.evaluation_parts:
                evaluation_indexes = train if evaluation_part == 'train' else test
                predictions = self.model.predict(percentile_x.iloc[evaluation_indexes])
                metric_score = self.compute_prediction_metric(self.y[evaluation_indexes], predictions)
                if evaluation_part == 'train':
                    metric_list_train.append(metric_score)
                else:
                    metric_list_test.append(metric_score)

        metric_mean_score = {'train': np.mean(metric_list_train),
                             'test': np.mean(metric_list_test)}
        metric_std_score = {'train': np.std(metric_list_train),
                            'test': np.std(metric_list_test)}

        return metric_mean_score, metric_std_score, features_importances


    @validate_model
    def check_model_type(self, model):
        return model


    @staticmethod
    def check_for_valid_quantiles_number(quantiles_number):
        return quantiles_number
