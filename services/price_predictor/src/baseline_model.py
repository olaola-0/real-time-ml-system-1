import pandas as pd


class BaselineModel:
    def __init__(self, n_candles_into_the_future: int):
        self.n_candles_into_the_future = n_candles_into_the_future

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Predicts the target metric for the given input features by computing the last oberved target metric
        (last observed price change) and use it as the prediction for the next n_candles_into_the_future candles
        """
        X_ = X.copy()
        X_['target'] = 0

        return X_['target']
