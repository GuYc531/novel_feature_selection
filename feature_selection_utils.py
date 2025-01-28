from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, BaseCrossValidator


def _set_cross_validation_policy(cv: BaseCrossValidator) -> BaseCrossValidator:
    """
        Sets and validates the cross-validation policy.

        This function ensures that a valid cross-validation object is provided or creates a default
        cross-validation strategy if none is specified. If `cv` is already an instance of `StratifiedKFold`
        or `KFold`, it is returned as-is. Otherwise, a default `StratifiedKFold` or `KFold` is created with
        pre-configured parameters.

        Args:
            cv (BaseCrossValidator): An optional cross-validation object. If None, a default strategy
                is created based on `StratifiedKFold` or `KFold`.

        Returns:
            BaseCrossValidator: A valid cross-validation object.

        Notes:
            - If no `cv` is provided, the default is a `StratifiedKFold` with 5 splits and shuffling enabled.
            - Falls back to `KFold` if `stratified` is set to False (though this isn't configurable in the function).

        Raises:
            ValueError: The function does not currently raise exceptions but assumes `cv` is either None
            or a valid cross-validator.

        """
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
        Computes the quantile of feature importance and filters the dataset to include only relevant features.

        This function calculates a specified quantile of feature importance scores, filters the features
        that meet or exceed that quantile, and returns the filtered dataset along with the computed quantile value.

        Args:
            df (pd.DataFrame): The original dataset containing all features.
            importance_df (pd.DataFrame): A DataFrame containing feature importance scores with columns:
                - `'feature_names'`: The names of the features.
                - `'feature_importance'`: The importance scores of the features based on fit done by the user.
            percent (float): The current percentile being evaluated (e.g., 0 for the first quantile).
            quantiles_number (int): The total number of quantiles to divide the feature importance range into.

        Returns:
            Tuple[float, pd.DataFrame]:
                - A float representing the computed quantile value for feature importance.
                - A DataFrame (`percentile_x`) containing only the features whose importance scores
                  are greater than or equal to the computed quantile.

        Notes:
            - The quantile is computed by dividing the `percent` by the `quantiles_number` and applying
              it to the `'feature_importance'` column of the `importance_df`.
            - The filtered dataset (`percentile_x`) only contains features with importance scores greater
              than or equal to the computed quantile.

        """
    quantile = importance_df['feature_importance'].quantile(q=percent / quantiles_number)
    percentile_features_df = importance_df[importance_df['feature_importance'] >= quantile]
    percentile_x = df[percentile_features_df["feature_names"]]

    return quantile, percentile_x


def _extract_model_name(model) -> str:
    """
        Extracts the name of the model's class.

        Args:
            model: The machine learning model instance.

        Returns:
            str: The name of the model's class.

        Example:
            >>> from sklearn.ensemble import RandomForestClassifier
            >>> model = RandomForestClassifier()
            >>> _extract_model_name(model)
            'RandomForestClassifier'
        """
    return type(model).__name__


def _compute_feature_importance_df(model) -> pd.DataFrame:
    """
        Computes and returns a DataFrame of feature importance scores for the given model.

        This function extracts the feature importance scores and feature names from the given model
        and returns a DataFrame with these values. The DataFrame is sorted in ascending order of feature importance.

        Args:
            model: The trained machine learning model. It must have the attributes `feature_importances_`
                   (importance scores for the features) and `feature_names_in_` (names of the features).

        Returns:
            pd.DataFrame: A DataFrame with two columns:
                - `'feature_names'`: The names of the features.
                - `'feature_importance'`: The corresponding importance scores for each feature, sorted in ascending order.

        Example:
            >>> from sklearn.ensemble import RandomForestClassifier
            >>> model = RandomForestClassifier()
            >>> model.fit(X_train, y_train)
            >>> importance_df = _compute_feature_importance_df(model)
            >>> print(importance_df)
               feature_names  feature_importance
            0         feature3            0.45
            1         feature1            0.35
            2         feature2            0.20

        Notes:
            - This function assumes the model has already been trained and has the necessary attributes.
            - The function sorts the DataFrame by feature importance in ascending order.
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
       Computes a specified prediction metric (accuracy, precision, recall, or f1) for model evaluation.

       This function calculates one of the common classification metrics based on the given ground truth
       (`y_true`) and predicted values (`y_predictions`). It supports both binary and multi-label classification
       tasks, with an option to select the metric for evaluation.

       Args:
           y_true (np.array): Ground truth (correct) labels for the data.
           y_predictions (np.array): Predicted labels by the model.
           multi_label (bool, optional): If True, the metric is computed for a multi-label classification task
                                         with a weighted average. Defaults to False (binary classification).
           metric (str, optional): The metric to compute. Options are:
               - 'accuracy': Computes accuracy score.
               - 'precision': Computes precision score.
               - 'recall': Computes recall score.
               - 'f1': Computes F1 score. Defaults to 'f1'.

       Returns:
           float: The computed metric value.

       Raises:
           ValueError: If an invalid metric is selected (not one of 'accuracy', 'precision', 'recall', or 'f1').

       Example:
           >>> y_true = [0, 1, 1, 0]
           >>> y_predictions = [0, 1, 0, 1]
           >>> _compute_prediction_metric(y_true, y_predictions, metric='accuracy')
           0.5

           >>> _compute_prediction_metric(y_true, y_predictions, multi_label=True, metric='f1')
           0.666...

       Notes:
           - The `average` parameter is set to `'binary'` for binary classification and `'weighted'` for multi-label.
           - For multi-label classification, the metric is averaged across all labels.

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
    """
        Appends feature importance scores to an existing dictionary or creates a new one.

        This function updates a dictionary of feature importance values by either initializing it with a new set
        of values from the provided DataFrame or appending new values to the existing dictionary. The DataFrame
        should contain two columns: 'feature_names' and 'feature_importance'.

        Args:
            features_importance_df (pd.DataFrame): A DataFrame containing the feature names and their corresponding
                                                    importance scores. Must have columns 'feature_names' and
                                                    'feature_importance'.
            features_importance_dict (dict, optional): A dictionary of feature importance scores where the keys are
                                                        feature names and the values are lists of importance scores.
                                                        Defaults to an empty dictionary.
            new (bool, optional): If True, the function will initialize the dictionary with the feature importance
                                  values from the DataFrame. If False, it appends the values to the existing dictionary.
                                  Defaults to False.

        Returns:
            dict[str, list]: The updated dictionary where the keys are feature names (as strings) and the values are lists
                             of feature importance scores.

        Notes:
            - The function assumes that the DataFrame columns are correctly named and that feature importance values are
              numeric (float).
            - When `new=True`, the function initializes the dictionary with the feature importance values,
              overwriting any existing data.
            - When `new=False`, the function appends the feature importance values to the existing lists in the dictionary.

        """
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
    Computes cross-validation metrics for the specified model and returns mean and standard deviation scores
    for specified evaluation parts (train, test, validation).

    This function performs cross-validation on the provided data and calculates a specified evaluation metric
    (e.g., accuracy, precision, recall, f1) for each fold. It also computes feature importances and can split
    the dataset into training, testing, and validation sets if specified.

    Args:
        percentile_x (pd.DataFrame): The input features for training the model.
        y (pd.Series): The target values corresponding to `percentile_x`.
        evaluation_parts (list): A list of evaluation parts to calculate the metric for. Possible values are ['train', 'test', 'validation'].
        cv: A cross-validator object (e.g., StratifiedKFold or KFold) used to split the data.
        model: A trained machine learning model with `.fit()` and `.predict()` methods.
        split_to_validation (bool, optional): If True, splits the data into a training set and a validation set
                                               based on `validation_size`. Defaults to False.
        validation_size (float, optional): The proportion of the dataset to be used as the validation set when
                                           `split_to_validation` is True. Defaults to 0.2 (20%).
        metric (str, optional): The metric to compute. Options are 'accuracy', 'precision', 'recall', 'f1'. Defaults to 'f1'.

    Returns:
        Tuple[dict[str, np.floating], dict[str, np.floating], dict[str, list]]:
            - A tuple containing:
                - `metric_mean_score`: A dictionary with mean scores for each evaluation part (train, test, validation).
                - `metric_std_score`: A dictionary with standard deviation scores for each evaluation part.
                - `features_importance`: A dictionary containing the feature importance values for each fold.

    Notes:
        - The function calculates the specified metric for each fold and returns the mean and standard deviation
          across all folds.
        - The `split_to_validation` flag controls whether a validation set is created and evaluated, and `validation_size`
          determines the size of this set.
        - The `multi_label` flag is automatically set based on the target variable `y` (if it has more than two unique values,
          multi-label classification is assumed).
        - The `metric` parameter allows for flexible evaluation based on the selected metric.
        - If `split_to_validation` is enabled, it tries to compute the metric for the validation set (using a separate
          validation dataset).

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


def _check_for_valid_quantiles_number(quantiles_number: int, df: pd.DataFrame) -> None:
    """
    Checks if the specified number of quantiles is valid based on the number of features in the dataset.

    This function ensures that the number of quantiles selected is lower than the number of features
    in the input dataset. If the number of quantiles is too high (i.e., more than the number of features), it raises
    a `ValueError`.

    Args:
        quantiles_number (int): The number of quantiles to be used in the analysis.
        df (pd.DataFrame): The dataset whose features are being analyzed.

    Raises:
        ValueError: If `quantiles_number` is more than the number of features (columns) in `df`.

    Example:
        >>> df = pd.DataFrame([[1, 2], [3, 4], [5, 6]])
        >>> _check_for_valid_quantiles_number(3, df)  # Valid as 3 quantiles >= 2 features
        >>> _check_for_valid_quantiles_number(1, df)  # Raises ValueError as 1 quantile < 2 features
    """
    if df.shape[1] < quantiles_number:
        raise ValueError(
            f"number of quantiles selected needs to be higher than the number of features in the dataset selected")
