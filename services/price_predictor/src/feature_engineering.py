from typing import Optional

import pandas as pd
import talib
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineering(BaseEstimator, TransformerMixin):
    """
    A Scikit Learn transformer that creates technical indicators from cryptocurrency price data.
    This transformer adds features to the input DataFrame that can be used for training a machine learning model.
    It wraps the existing 'add_features' function into a scikit-learn transformer, allowing it to persist using the 'joblib' library.

    This approach offers several benefits:
    - It enables hyper-parameter optimization of the feature engineering pipeline.
    - The best pipeline can be saved to disk alongside the model pickle, eliminating the need for a separate JSON file.
    """

    def __init__(
        self,
        n_candles_into_future: int,
        # Momentum Indicators
        RSI_timeperiod: Optional[int] = 14,
        MOM_timeperiod: Optional[int] = 10,
        MACD_fastperiod: Optional[int] = 12,
        MACD_slowperiod: Optional[int] = 26,
        MACD_signalperiod: Optional[int] = 9,
        ADX_timeperiod: Optional[int] = 14,
        ROC_timeperiod: Optional[int] = 10,
        STOCH_fastk_period: Optional[int] = 5,
        STOCH_slowk_period: Optional[int] = 3,
        STOCH_slowk_matype: Optional[int] = 0,
        STOCH_slowd_period: Optional[int] = 3,
        STOCH_slowd_matype: Optional[int] = 0,
        ULTOSC_timeperiod1: Optional[int] = 7,
        ULTOSC_timeperiod2: Optional[int] = 14,
        ULTOSC_timeperiod3: Optional[int] = 28,
        # Statistics Indicators
        STDDEV_timeperiod: Optional[int] = 5,
        STDDEV_nbdev: Optional[int] = 1,
        # Volatility Indicators
        ATR_timeperiod: Optional[int] = 14,
        # Volume Indicators
        MFI_timeperiod: Optional[int] = 14,
        OBV_timeperiod: Optional[int] = 14,
        FI_timeperiod: Optional[int] = 14,
    ):
        """
        Saves the parameters as attributes of the transformer

        Args:
            - n_candles_into_future: The number of candles into the future to predict

            # Momentum Indicators
            - RSI_timeperiod: The time period for the RSI (Relative Strength Index) indicator
            - MOM_timeperiod: The time period for the Momentum indicator
            - MACD_fastperiod: The time period for the fast EMA in the MACD (Moving Average Convergence Divergence) indicator
            - MACD_slowperiod: The time period for the slow EMA in the MACD (Moving Average Convergence Divergence)indicator
            - MACD_signalperiod: The time period for the signal line in the MACD indicator
            - MFI_timeperiod: The time period for the MFI (Money Flow Index) indicator
            - ADX_timeperiod: The time period for the ADX (Average Directional Movement Index) indicator
            - ROC_timeperiod: The time period for the ROC (Rate of Change) indicator
            - STOCH_fastk_period: The time period for the fast K in the Stochastic Oscillator indicator
            - STOCH_slowk_period: The time period for the slow K in the Stochastic Oscillator indicator
            - STOCH_slowk_matype: The moving average type for the slow K in the Stochastic Oscillator indicator
            - STOCH_slowd_period: The time period for the slow D in the Stochastic Oscillator indicator
            - STOCH_slowd_matype: The moving average type for the slow D in the Stochastic Oscillator indicator
            - ULTOSC_timeperiod1: The time period for the first time period in the Ultimate Oscillator indicator
            - ULTOSC_timeperiod2: The time period for the second time period in the Ultimate Oscillator indicator
            - ULTOSC_timeperiod3: The time period for the third time period in the Ultimate Oscillator indicator

            # Statistics Indicators
            - STDDEV_timeperiod: The time period for the Standard Deviation indicator
            - STDDEV_nbdev: The number of standard deviations to use in the Standard Deviation indicator

            # Volatility Indicators
            - ATR_timeperiod: The time period for the Average True Range indicator

            # Volume Indicators
            - OBV_timeperiod: The time period for the OBV (On-Balance Volume) indicator
            - VPT_timeperiod: The time period for the VPT (Volume Price Trend) indicator
            - FI_timeperiod: The time period for the FI (Force Index) indicator
            - PVI_timeperiod: The time period for the PVI (Positive Volume Index) indicator

        Returns:
            None
        """
        self.n_candles_into_future = n_candles_into_future

        # Momentum Indicators
        self.RSI_timeperiod = RSI_timeperiod
        self.MOM_timeperiod = MOM_timeperiod
        self.MACD_fastperiod = MACD_fastperiod
        self.MACD_slowperiod = MACD_slowperiod
        self.MACD_signalperiod = MACD_signalperiod
        self.ADX_timeperiod = ADX_timeperiod
        self.ROC_timeperiod = ROC_timeperiod
        self.STOCH_fastk_period = STOCH_fastk_period
        self.STOCH_slowk_period = STOCH_slowk_period
        self.STOCH_slowk_matype = STOCH_slowk_matype
        self.STOCH_slowd_period = STOCH_slowd_period
        self.STOCH_slowd_matype = STOCH_slowd_matype
        self.ULTOSC_timeperiod1 = ULTOSC_timeperiod1
        self.ULTOSC_timeperiod2 = ULTOSC_timeperiod2
        self.ULTOSC_timeperiod3 = ULTOSC_timeperiod3

        # Statistics Indicators
        self.STDDEV_timeperiod = STDDEV_timeperiod
        self.STDDEV_nbdev = STDDEV_nbdev

        # Volatility Indicators
        self.ATR_timeperiod = ATR_timeperiod

        # Volume Indicators
        self.OBV_timeperiod = OBV_timeperiod
        self.MFI_timeperiod = MFI_timeperiod
        self.FI_timeperiod = FI_timeperiod

        self.final_features = [
            'RSI',
            'MOM',
            'MACD',
            'MACD_signal',
            'ADX',
            'ROC',
            'STOCH_slowk',
            'STOCH_slowd',
            'ULTOSC',
            'STDDEV',
            'ATR',
            'MFI',
            'OBV',
            'FI',
            'last_observed_target',
            'day_of_week',
            'hour_of_day',
            'minute_of_hour',
        ]

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Adds features to the input DataFrame that can be used for training a machine learning model.

        Args:
            X: A DataFrame containing the input data

        Returns:
            X: A DataFrame containing the input data with additional features
        """
        # Add the technical indicators to the input DataFrame
        return add_features(
            X,
            n_candles_into_future=self.n_candles_into_future,
            RSI_timeperiod=self.RSI_timeperiod,
            MOM_timeperiod=self.MOM_timeperiod,
            MACD_fastperiod=self.MACD_fastperiod,
            MACD_slowperiod=self.MACD_slowperiod,
            MACD_signalperiod=self.MACD_signalperiod,
            ADX_timeperiod=self.ADX_timeperiod,
            ROC_timeperiod=self.ROC_timeperiod,
            STOCH_fastk_period=self.STOCH_fastk_period,
            STOCH_slowk_period=self.STOCH_slowk_period,
            STOCH_slowk_matype=self.STOCH_slowk_matype,
            STOCH_slowd_period=self.STOCH_slowd_period,
            STOCH_slowd_matype=self.STOCH_slowd_matype,
            ULTOSC_timeperiod1=self.ULTOSC_timeperiod1,
            ULTOSC_timeperiod2=self.ULTOSC_timeperiod2,
            ULTOSC_timeperiod3=self.ULTOSC_timeperiod3,
            STDDEV_timeperiod=self.STDDEV_timeperiod,
            STDDEV_nbdev=self.STDDEV_nbdev,
            ATR_timeperiod=self.ATR_timeperiod,
            MFI_timeperiod=self.MFI_timeperiod,
            OBV_timeperiod=self.OBV_timeperiod,
            FI_timeperiod=self.FI_timeperiod,
        )[self.final_features]


def add_features(
    X: pd.DataFrame,
    n_candles_into_future: int,
    RSI_timeperiod: Optional[int] = 14,
    MOM_timeperiod: Optional[int] = 10,
    MACD_fastperiod: Optional[int] = 12,
    MACD_slowperiod: Optional[int] = 26,
    MACD_signalperiod: Optional[int] = 9,
    ADX_timeperiod: Optional[int] = 14,
    ROC_timeperiod: Optional[int] = 10,
    STOCH_fastk_period: Optional[int] = 5,
    STOCH_slowk_period: Optional[int] = 3,
    STOCH_slowk_matype: Optional[int] = 0,
    STOCH_slowd_period: Optional[int] = 3,
    STOCH_slowd_matype: Optional[int] = 0,
    ULTOSC_timeperiod1: Optional[int] = 7,
    ULTOSC_timeperiod2: Optional[int] = 14,
    ULTOSC_timeperiod3: Optional[int] = 28,
    STDDEV_timeperiod: Optional[int] = 5,
    STDDEV_nbdev: Optional[int] = 1,
    ATR_timeperiod: Optional[int] = 14,
    MFI_timeperiod: Optional[int] = 14,
    OBV_timeperiod: Optional[int] = 14,
    FI_timeperiod: Optional[int] = 14,
) -> pd.DataFrame:
    """ """
    X_ = X.copy()

    # Add momentum indicators
    X_ = add_RSI(X_, timeperiod=RSI_timeperiod)
    X_ = add_MOM(X_, timeperiod=MOM_timeperiod)
    X_ = add_MACD(
        X_,
        fastperiod=MACD_fastperiod,
        slowperiod=MACD_slowperiod,
        signalperiod=MACD_signalperiod,
    )
    X_ = add_ADX(X_, timeperiod=ADX_timeperiod)
    X_ = add_ROC(X_, timeperiod=ROC_timeperiod)
    X_ = add_STOCH(
        X_,
        fastk_period=STOCH_fastk_period,
        slowk_period=STOCH_slowk_period,
        slowk_matype=STOCH_slowk_matype,
        slowd_period=STOCH_slowd_period,
        slowd_matype=STOCH_slowd_matype,
    )
    X_ = add_ULTOSC(
        X_,
        timeperiod1=ULTOSC_timeperiod1,
        timeperiod2=ULTOSC_timeperiod2,
        timeperiod3=ULTOSC_timeperiod3,
    )

    # Add statistics indicators
    X_ = add_STDDEV(X_, timeperiod=STDDEV_timeperiod, nbdev=STDDEV_nbdev)

    # Add volatility indicators
    X_ = add_ATR(X_, timeperiod=ATR_timeperiod)

    # Add last observed target
    X_ = add_last_observed_target(X_, n_candles_into_future=n_candles_into_future)

    # Add volume indicators
    X_ = add_MFI(X_, timeperiod=MFI_timeperiod)
    X_ = add_OBV(X_)
    X_ = add_FI(X_)

    # Add temporal time features
    X_ = add_temporal_features(X_)

    return X_


def add_RSI(X: pd.DataFrame, timeperiod: Optional[int] = 14) -> pd.DataFrame:
    """
    Adds the Relative Strength Index (RSI) indicator to the input DataFrame
    which measures the speed and change of price movements and is calculated as follows:
    RSI = 100 - (100 / (1 + RS))
    where RS = Average gain of up periods during the specified time period / Average loss of down periods during the specified time period
    """
    X['RSI'] = talib.RSI(X['close'], timeperiod=timeperiod)
    return X


def add_MOM(X: pd.DataFrame, timeperiod: Optional[int] = 10) -> pd.DataFrame:
    """
    Adds the Momentum (MOM) indicator to the input DataFrame
    which measures the change in price over a specified time period
    Calculated as:
    MOM = Price(t) - Price(t - n)
    where:
    Price(t) is the closing price at time t, and Price(t - n) is the closing price n periods ago
    """
    X['MOM'] = talib.MOM(X['close'], timeperiod=timeperiod)
    return X


def add_MACD(
    X: pd.DataFrame,
    fastperiod: Optional[int] = 12,
    slowperiod: Optional[int] = 26,
    signalperiod: Optional[int] = 9,
) -> pd.DataFrame:
    """
    Adds the Moving Average Convergence Divergence (MACD) indicator to the input DataFrame
    which is calculated as the difference between the 12-day EMA and the 26-day EMA
    Calculated as:
    MACD = EMA(12) - EMA(26)
    where:
    EMA(12) is the 12-day Exponential Moving Average, and EMA(26) is the 26-day Exponential Moving Average
    """
    macd, macdsignal, _ = talib.MACD(
        X['close'],
        fastperiod=fastperiod,
        slowperiod=slowperiod,
        signalperiod=signalperiod,
    )
    X['MACD'] = macd
    X['MACD_signal'] = macdsignal
    return X


def add_ADX(X: pd.DataFrame, timeperiod: Optional[int] = 14) -> pd.DataFrame:
    """
    Adds the Average Directional Movement Index (ADX) indicator to the input DataFrame
    which measures the strength of a trend without regard to its direction
    Calculated as:
    ADX = 100 * EMA(ABS(+DI - -DI) / (+DI + -DI))
    where:
    +DI = Current High - Previous High
    -DI = Previous Low - Current Low
    """
    X['ADX'] = talib.ADX(X['high'], X['low'], X['close'], timeperiod=timeperiod)
    return X


def add_ROC(X: pd.DataFrame, timeperiod: Optional[int] = 10) -> pd.DataFrame:
    """
    Adds the Rate of Change (ROC) indicator to the input DataFrame
    which measures the percentage change in price between the current price and the price n periods ago
    Calculated as:
    ROC = (Price(t) - Price(t - n)) / Price(t - n)
    where:
    Price(t) is the closing price at time t, and Price(t - n) is the closing price n periods ago
    """
    X['ROC'] = talib.ROC(X['close'], timeperiod=timeperiod)
    return X


def add_STOCH(
    X: pd.DataFrame,
    fastk_period: Optional[int] = 5,
    slowk_period: Optional[int] = 3,
    slowk_matype: Optional[int] = 0,
    slowd_period: Optional[int] = 3,
    slowd_matype: Optional[int] = 0,
) -> pd.DataFrame:
    """
    Adds the Stochastic Oscillator indicator to the input DataFrame
    which measures the location of the close relative to the high-low range over a specified time period
    Calculated as:
    %K = 100 * (Close - Low(n)) / (High(n) - Low(n))
    %D = 3-day SMA of %K
    where:
    Close = Current Close
    Low(n) = Lowest Low over the specified time period
    High(n) = Highest High over the specified time period
    """
    slowk, slowd = talib.STOCH(
        X['high'],
        X['low'],
        X['close'],
        fastk_period=fastk_period,
        slowk_period=slowk_period,
        slowk_matype=slowk_matype,
        slowd_period=slowd_period,
        slowd_matype=slowd_matype,
    )
    X['STOCH_slowk'] = slowk
    X['STOCH_slowd'] = slowd
    return X


def add_ULTOSC(
    X: pd.DataFrame,
    timeperiod1: Optional[int] = 7,
    timeperiod2: Optional[int] = 14,
    timeperiod3: Optional[int] = 28,
) -> pd.DataFrame:
    """
    Adds the Ultimate Oscillator indicator to the input DataFrame
    which measures the buying or selling pressure by comparing the close in relation to the high-low range
    Calculated as:
    BP = Close - Low(n)
    TR = High(n) - Low(n)
    Average1 = (Sum of BP over timeperiod1) / (Sum of TR over timeperiod1)
    Average2 = (Sum of BP over timeperiod2) / (Sum of TR over timeperiod2)
    Average3 = (Sum of BP over timeperiod3) / (Sum of TR over timeperiod3)
    Ultimate Oscillator = 100 * ((4 * Average1) + (2 * Average2) + Average3) / (4 + 2 + 1)
    """
    X['ULTOSC'] = talib.ULTOSC(
        X['high'],
        X['low'],
        X['close'],
        timeperiod1=timeperiod1,
        timeperiod2=timeperiod2,
        timeperiod3=timeperiod3,
    )
    return X


def add_STDDEV(
    X: pd.DataFrame, timeperiod: Optional[int] = 5, nbdev: Optional[int] = 1
) -> pd.DataFrame:
    """
    Adds the Standard Deviation indicator to the input DataFrame
    which measures the dispersion of a set of values from their average
    Calculated as:
    STDDEV = SQRT(SUM((Price - Average Price)^2) / n)
    where:
    Price = Close price
    Average Price = Average of the close prices over the specified time period
    n = Number of observations
    """
    X['STDDEV'] = talib.STDDEV(X['close'], timeperiod=timeperiod, nbdev=nbdev)
    return X


def add_ATR(X: pd.DataFrame, timeperiod: Optional[int] = 14) -> pd.DataFrame:
    """
    Adds the Average True Range (ATR) indicator to the input DataFrame
    which measures market volatility by calculating the moving average of the true range
    Calculated as:
    ATR = EMA(True Range, n)
    where:
    True Range = Max(High - Low, ABS(High - Previous Close), ABS(Low - Previous Close))
    """
    X['ATR'] = talib.ATR(X['high'], X['low'], X['close'], timeperiod=timeperiod)
    return X


def add_MFI(X: pd.DataFrame, timeperiod: Optional[int] = 14) -> pd.DataFrame:
    """
    Adds the Money Flow Index (MFI) indicator to the input DataFrame
    which measures the strength of money flowing in and out of a security
    Calculated as:
    MFI = 100 - (100 / (1 + Money Ratio))
    where:
    Money Ratio = Positive Money Flow / Negative Money Flow
    Positive Money Flow = Sum of Positive Money over the specified time period
    Negative Money Flow = Sum of Negative Money over the specified time period
    """
    X['MFI'] = talib.MFI(
        X['high'], X['low'], X['close'], X['volume'], timeperiod=timeperiod
    )
    return X


def add_OBV(X: pd.DataFrame) -> pd.DataFrame:
    """
    Adds the On-Balance Volume (OBV) indicator to the input DataFrame
    which measures buying and selling pressure by adding the volume on up days and subtracting the volume on down days
    Calculated as:
    OBV = Previous OBV + Volume if Close > Previous Close
    OBV = Previous OBV - Volume if Close < Previous Close
    """
    X['OBV'] = talib.OBV(X['close'], X['volume'])
    return X


def add_FI(X: pd.DataFrame) -> pd.DataFrame:
    """
    Adds the Force Index (FI) indicator to the input DataFrame
    which measures the strength of a price trend by multiplying the volume by the price change
    Calculated as:
    FI = Volume * (Close - Previous Close)
    """
    X['FI'] = X['volume'] * (X['close'] - X['close'].shift(1))
    return X


def add_temporal_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Adds columns with temporal features to the input DataFrame using the timestamp column
    - day_of_week: The day of the week (0 = Monday, 6 = Sunday)
    - hour_of_day: The hour of the day (0-23)
    - minute_of_hour: The minute of the hour (0-59)

    Args:
        X: A DataFrame containing the input data

    Returns:
        X: A DataFrame containing the input data with additional temporal features
    """
    X_ = X.copy()

    # Ensure the 'timestamp' column is a datetime object
    #X_['timestamp'] = pd.to_datetime(X_['timestamp'])

    # Extract temporal features
    X_['day_of_week'] = pd.to_datetime(X_['timestamp']).dt.dayofweek
    X_['hour_of_day'] = pd.to_datetime(X_['timestamp']).dt.hour
    X_['minute_of_hour'] =pd.to_datetime(X_['timestamp']).dt.minute

    return X_


def add_last_observed_target(
    X: pd.DataFrame, n_candles_into_future: int
) -> pd.DataFrame:
    """
    Adds the last observed target value to the input DataFrame
    which is the closing price n candles into the future

    Args:
        X: A DataFrame containing the input data
        n_candles_into_future: The number of candles into the future to predict

    Returns:
        X: A DataFrame containing the input data with the last observed target value
    """
    X_ = X.copy()

    X_['last_observed_target'] = X_['close'].pct_change(n_candles_into_future)

    return X_