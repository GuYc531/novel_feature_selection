
xgboost_hyperparametes = {
    'objective': 'binary:logistic',
    'gamma': 7,
    'learning_rate': 0.1,
    'max_depth': 5,
    'min_child_weight': 7,
    'reg_alpha': 0.5,
    'reg_lambda': 0.5
}

random_forest_hyperparameters = {
    'class_weight': 'balanced',
    'criterion': 'gini',
    'max_depth': 5,
    'min_samples_split': 5,
    'n_estimators': 30,
    'max_leaf_nodes': 10
}
