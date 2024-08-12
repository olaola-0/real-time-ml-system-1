from typing import Optional

import pandas as pd
from sklearn.linear_model import Lasso
from xgboost import XGBRegressor


def fit_xgboost_regressor(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    tuner_hyper_params: Optional[bool] = False,
) -> XGBRegressor:
    """
    Fits an XGBoost regressor to the training data.

    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        tuner_hyper_params (bool): Whether to tune hyperparameters

    Returns:
        XGBRegressor: Fitted XGBoost regressor
    """
    model = XGBRegressor()
    model.fit(X_train, y_train)
    return model


def fit_lasso_regressor(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    tuner_hyper_params: Optional[bool] = False,
) -> Lasso:
    """
    Fits a Lasso regressor to the training data.

    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        tuner_hyper_params (bool): Whether to tune hyperparameters

    Returns:
        Lasso: Fitted Lasso regressor
    """
    model = Lasso(alpha=0.1)
    model.fit(X_train, y_train)
    return model
