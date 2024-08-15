import warnings
from typing import Optional

import numpy as np
import optuna
import pandas as pd
from loguru import logger
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# Suppress unnecessary warnings for a cleaner output
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


def fit_xgboost_regressor(
    X: pd.DataFrame,
    y: pd.Series,
    hyper_param_search_trials: Optional[int] = 0,
) -> XGBRegressor:
    """
    Fits an XGBoost regressor model to the given dataset with optional hyperparameter tuning.

    Parameters:
    -----------
    X : pd.DataFrame
        The feature matrix where each row represents an observation and each column represents a feature.
    y : pd.Series
        The target variable corresponding to each observation in the dataset.
    hyper_param_search_trials : Optional[int], default=0
        The number of trials for hyperparameter optimization using Optuna. If set to 0,
        the model will be trained using XGBoost's default hyperparameters. If set to a positive integer,
        the function will perform hyperparameter optimization for the specified number of trials.

    Returns:
    --------
    model : XGBRegressor
        An XGBoost regressor model fitted to the provided data.
    """
    # Check if hyperparameter tuning is requested
    if hyper_param_search_trials == 0:
        # No hyperparameter tuning; use default parameters
        model = XGBRegressor()
        model.fit(X, y)

    else:
        # Define the objective function for Optuna
        def objective(trial):
            """
            Objective function for Optuna to optimize hyperparameters for the XGBoost model.

            This function is called by the Optuna framework to suggest and evaluate a set of hyperparameters.
            It performs cross-validation on the provided dataset and returns the mean Mean Absolute Error (MAE)
            across the validation folds. The goal is to minimize this MAE.

            Parameters:
            -----------
            trial : optuna.Trial
                An Optuna trial object that suggests hyperparameters for the XGBoost model.

            Returns:
            --------
            float
                The mean MAE across all cross-validation folds for the current set of hyperparameters.
            """
            # Suggest hyperparameters to tune
            param = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 50),
                'learning_rate': trial.suggest_float(
                    'learning_rate', 1e-4, 0.1, log=True
                ),
                'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
                'alpha': trial.suggest_float('alpha', 1e-5, 1e1, log=True),
                'lambda': trial.suggest_float('lambda', 1e-5, 1e1, log=True),
            }
            # Create an XGBRegressor model with the suggested hyperparameters
            model = XGBRegressor(**param)

            # Time-based cross-validation with 3 splits
            tscv = TimeSeriesSplit(n_splits=3)

            mae_scores = []
            for train_index, test_index in tscv.split(X):
                # Split the data into training and test sets
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                # Fit the model and evaluate it
                model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_test, y_test)],
                    early_stopping_rounds=10,
                    verbose=False,
                )
                y_pred = model.predict(X_test)
                mae = mean_absolute_error(y_test, y_pred)
                mae_scores.append(mae)

            # Return the mean MAE score across folds as the objective to minimize
            return np.mean(mae_scores)

        # Create an Optuna study to minimize the objective function
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=hyper_param_search_trials)

        # Log the best hyperparameters and value found
        logger.info(f'Best hyperparameters: {study.best_params}')
        logger.info(f'Best value: {study.best_value}')

        # Fit the model with the best hyperparameters found during the study
        model = XGBRegressor(**study.best_params)
        model.fit(X, y)

    return model


def fit_lasso_regressor(
    X: pd.DataFrame,
    y: pd.Series,
    hyper_param_search_trials: Optional[int] = 0,
) -> Lasso:
    """
    Fits a Lasso regression model to the given dataset with optional hyperparameter tuning.

    Parameters:
    -----------
    X : pd.DataFrame
        The feature matrix where each row represents an observation and each column represents a feature.
    y : pd.Series
        The target variable corresponding to each observation in the dataset.
    hyper_param_search_trials : Optional[int], default=0
        The number of trials for hyperparameter optimization using Optuna. If set to 0,
        the model will be trained using a default alpha value of 0.1. If set to a positive integer,
        the function will perform hyperparameter optimization for the specified number of trials.

    Returns:
    --------
    model : Lasso
        A Lasso regression model fitted to the provided data.
    """
    # Check if hyperparameter tuning is requested
    if hyper_param_search_trials == 0:
        # Log the default hyperparameter value
        logger.info(
            'Fitting Lasso regression model with default hyperparameter of alpha=0.1.'
        )

        # Create a pipeline with data scaling and Lasso regression with default alpha
        model = Pipeline([('scaler', StandardScaler()), ('lasso', Lasso(alpha=0.1))])
        model.fit(X, y)

    else:
        # Log the start of hyperparameter optimization
        logger.info(
            f'Performing {hyper_param_search_trials} trials of hyperparameter optimization for Lasso regression model.'
        )

        def objective(trial):
            """
            Objective function for Optuna to optimize the alpha hyperparameter for the Lasso model.

            This function is called by the Optuna framework to suggest and evaluate a set of hyperparameters.
            It performs cross-validation on the provided dataset and returns the mean Mean Absolute Error (MAE)
            across the validation folds. The goal is to minimize this MAE.

            Parameters:
            -----------
            trial : optuna.Trial
                An Optuna trial object that suggests hyperparameters for the Lasso model.

            Returns:
            --------
            float
                The mean MAE across all cross-validation folds for the current set of hyperparameters.
            """
            # Suggest a hyperparameter value for alpha
            alpha = trial.suggest_float('alpha', 1e-4, 1e1, log=True)

            # Create a Lasso regression model with the suggested alpha value
            model = Pipeline(
                [
                    ('scaler', StandardScaler()),  # Standardize the features
                    ('lasso', Lasso(alpha=alpha)),
                ]
            )

            # Time-based cross-validation with 4 splits
            tscv = TimeSeriesSplit(n_splits=4)

            mae_scores = []

            for train_index, test_index in tscv.split(X):
                # Split the data into training and test sets
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                # Fit the model and evaluate it
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                mae = mean_absolute_error(y_test, y_pred)
                mae_scores.append(mae)

            # Return the mean MAE score across folds as the objective to minimize
            return np.mean(mae_scores)

        # Create an Optuna study to minimize the objective function
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=hyper_param_search_trials)

        # Log the best hyperparameters and value found
        logger.info(f"Best alpha: {study.best_params['alpha']}")
        logger.info(f'Best value: {study.best_value}')

        # Fit the model with the best hyperparameters found during the study
        model = Pipeline(
            [
                ('scaler', StandardScaler()),  # Standardize the features
                ('lasso', Lasso(alpha=study.best_params['alpha'])),
            ]
        )
        model.fit(X, y)

    return model
