from typing import Tuple, AnyStr

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, KFold


class feature_selection_by_importance():
    def __init__(self, df: pd.DataFrame, y: pd.Series, metric: str, quantiles_number: int, cv, model):
        self.df = df
        self.y = y
        self.quantiles_number = quantiles_number
        self.model = model
        self.metric = metric
        self.cv = self.set_cross_validation_policy(cv)

    def set_cross_validation_policy(self, cv):
        if cv is not None:
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
        :param quantiles_number:
        :param X:
        :return:
        """
        quantile = importance_df['feature_importance'].quantile(q=percent / self.quantiles_number)
        percentile_features_df = importance_df[importance_df['feature_importance'] >= quantile]
        percentile_X = self.df[percentile_features_df["feature_names"]]

        return quantile, percentile_X

    def plot_feature_importance_across_quantiles(self, save_figure_in_path=False):
        # TODO: fix saveing figures
        X_axis = np.arange(self.quantiles_number)
        fig = plt.figure()
        plt.bar(X_axis - 0.3, self.mean_trains, 0.3, label='Train', color='white', edgecolor='black', yerr=self.std_trains)
        plt.bar(X_axis, self.mean_tests, 0.3, label='Test', color='grey', edgecolor='black', yerr=self.std_tests)
        plt.xticks(X_axis, self.X_axis_labels, fontsize=6)
        plt.xlabel("Feature number, Feature quantile", fontsize=12)
        plt.ylabel(self.metric, fontsize=16)
        plt.title(f"Mean {self.metric} score +/- std")
        plt.legend()
        plt.show()
        plt.savefig('feature_selection_figure.png')

    def compute_feature_importance_by_percentile(self) -> None:
        self.X_axis_labels = []
        self.mean_trains, self.std_trains, self.mean_tests, self.std_tests = [], [], [], []
        importance_df = self.compute_feature_importance_df()

        for percent in range(self.quantiles_number):
            # get percentile and create new df after filtering

            quantile, percentile_X = self.compute_quantile_and_rellevant_df(importance_df, percent)
            self.model.fit(percentile_X, self.y)
            print(str(type(self.model)).split('.')[-1])
            print(
                f"{1 - percent / 10}, --> total columns = {len(percentile_X.columns)}, over feature importance {quantile}")
            self.X_axis_labels.append(f'{len(percentile_X.columns)}, {round(quantile, 2)}')
            # y_predicted = self.model.predict(percentile_X)
            flags = ['train', 'test']

            for flag in flags:
                mean, std, features_importances = self.compute_cv_metric_score(percentile_X, flag)

                if flag == 'train':
                    self.mean_trains.append(mean)
                    self.std_trains.append(std)
                else:
                    self.mean_tests.append(mean)
                    self.std_tests.append(std)

    def compute_feature_importance_df(self) -> pd.DataFrame:
        """
        Compute feature importance for fitted model.
        :param model: fitted model object based on tree model
        :return: pd.DataFrame of feature_names and feature_importance
        """
        # TODO: add check if model is fitted if not fit with whole data
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
        if self.metric == 'accuracy':
            return accuracy_score(y_true, y_predictions)
        else:
            return 0

    def compute_cv_metric_score(self, percentile_X: pd.DataFrame, flag: str) \
            -> Tuple[np.floating, np.floating, dict[int, any]]:
        """

        :param X:
        :param y:
        :param flag:
        :param metric:
        :param forest_model:
        :return:
        """

        metric_list = []
        features_importances = {}

        for i, (train, test) in enumerate(self.cv.split(percentile_X, self.y)):
            evaluation_part = train if flag == 'train' else test

            # fit new df and compute feature  importance
            self.model.fit(percentile_X.iloc[train], self.y[train])
            features_importance_df = self.compute_feature_importance_df()
            features_importances[i] = features_importance_df['feature_importance']

            # compute score
            predictions = self.model.predict(percentile_X.iloc[evaluation_part])
            metric_score = self.compute_prediction_metric(self.y[evaluation_part], predictions)
            metric_list.append(metric_score)

        metric_mean_score = np.mean(metric_list)
        metric_std_score = np.std(metric_list)

        return metric_mean_score, metric_std_score, features_importances
