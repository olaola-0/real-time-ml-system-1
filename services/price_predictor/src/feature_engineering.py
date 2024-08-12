from typing import Optional

import pandas as pd
import talib


def add_features(
    X: pd.DataFrame,
    n_candles_into_future: int,
    rsi_period: Optional[int] = 14,
    momentum_period: Optional[int] = 14,
    volatility_period: Optional[int] = 5,
    fillna: Optional[bool] = True,
) -> pd.DataFrame:
    """
    Adds multiple features to the dataframe using the input dataframe and parameters.
        - OBV indicator -> On-Balance Volume
        - RSI indicator -> Relative Strength Index `rsi` column
        - Momentum indicator -> Momentum `momentum` column
        - Volatility indicator -> Standard deviation `std` column
        - Last observed target -> `last_observed_target` column
        - Temporal features -> `day_of_week`, `hour_of_day`, `minute_of_hour` columns

    Parameters:
        - X (pd.DataFrame): input dataframe
        - n_candles_into_future (int): number of candles into the future to shift the target column
        - rsi_period (int): lookback period for RSI
        - momentum_period (int): lookback period for momentum
        - volatility_period (int): lookback period for volatility
        - fillna (bool): whether to fill NaN values with 0

    Returns:
        pd.DataFrame: dataframe with additional features
    """
    X_ = add_obv_indicator(X=X, fillna=fillna)
    X_ = add_momentum_indicators(
        X=X_, rsi_period=rsi_period, momentum_period=momentum_period, fillna=fillna
    )
    X_ = add_volatility_indicator(X=X_, timeperiod=volatility_period, fillna=fillna)
    X_ = add_macd_indicator(X=X_, fillna=fillna)
    X_ = add_last_observed_target(X=X_, n_candles_into_future=n_candles_into_future)
    X_ = add_temporal_features(X=X_)

    return X_


def add_temporal_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Adds temporal features to the dataframe using df['datetime']
        - day_of_week
        - hour_of_day
        - minute_of_hour

    Parameters:
        df (pd.DataFrame): input dataframe

    Returns:
        pd.DataFrame: dataframe with additional features
    """
    X_ = X.copy()

    X_['day_of_week'] = X_['datetime'].dt.dayofweek  # Monday=0, Sunday=6
    X_['hour_of_day'] = X_['datetime'].dt.hour
    X_['minute_of_hour'] = X_['datetime'].dt.minute

    return X_


def add_moving_average_indicators(
    X: pd.DataFrame,
    short_window: Optional[int] = 20,
    long_window: Optional[int] = 50,
    fillna: Optional[bool] = True,
) -> pd.DataFrame:
    """
    Adds moving average indicators to the dataframe using df['close'] column.
    - Short Simple Moving Average (short_sma) -> SMA of close price over short window
    - Long Simple Moving Average (long_sma) -> SMA of close price over long window
    - Short Exponential Moving Average (short_ema) -> EMA of close price over short window
    - Long Exponential Moving Average (long_ema) -> EMA of close price over long window

    Parameters:
        - X (pd.DataFrame): input dataframe
        - short_window (int): lookback period for short window
        - long_window (int): lookback period for long window
        - fillna (bool): whether to fill NaN values with 0

    Returns:
        pd.DataFrame: dataframe with additional features
    """
    X_ = X.copy()

    # Short and long simple moving averages
    X_['short_sma'] = talib.SMA(X_['close'], timeperiod=short_window)
    X_['long_sma'] = talib.SMA(X_['close'], timeperiod=long_window)

    # Short and long Exponential moving averages
    X_['short_ema'] = talib.EMA(X_['close'], timeperiod=short_window)
    X_['long_ema'] = talib.EMA(X_['close'], timeperiod=long_window)

    if fillna:
        X_['short_sma'] = X_['short_sma'].fillna(0)
        X_['long_sma'] = X_['long_sma'].fillna(0)
        X_['short_ema'] = X_['short_ema'].fillna(0)
        X_['long_ema'] = X_['long_ema'].fillna(0)

    return X_


def add_obv_indicator(X: pd.DataFrame, fillna: Optional[bool] = True) -> pd.DataFrame:
    """
    Adds On-Balance Volume (OBV) to the dataframe using df['close'] and df['volume'] columns.
    - On-Balance Volume (obv) -> OBV is a momentum indicator that uses volume flow to predict changes in stock price.
    - It is calculated by adding the current period's volume to a cumulative total if the price moves up
    - Subtracting the current period's volume from the cumulative total if the price moves down.

    Parameters:
        - X (pd.DataFrame): input dataframe
        - fillna (bool): whether to fill NaN values with 0
    """
    X_ = X.copy()

    X_['obv'] = talib.OBV(X_['close'], X_['volume'])

    if fillna:
        X_['obv'] = X_['obv'].fillna(0)

    return X_


def add_momentum_indicators(
    X: pd.DataFrame,
    rsi_period: Optional[int] = 14,  # Relative Strength Index
    momentum_period: Optional[int] = 14,
    fillna: Optional[bool] = True,
) -> pd.DataFrame:
    """
    Adds momentum indicators to the dataframe using dX['close']
        - RSI indicator -> Relative Strength Index `rsi` column
        - Momentum indicator -> Momentum `momentum` column

    Parameters:
        - X (pd.DataFrame): input dataframe

    Returns:
        pd.DataFrame: dataframe with additional features
    """
    X_ = X.copy()

    # RSI indicator
    X_['rsi'] = talib.RSI(X_['close'], timeperiod=rsi_period)

    # Momentum indicator
    X_['momentum'] = talib.MOM(X_['close'], timeperiod=momentum_period)

    if fillna:
        X_['rsi'] = X_['rsi'].fillna(0)
        X_['momentum'] = X_['momentum'].fillna(0)

    return X_


def add_volatility_indicator(
    X: pd.DataFrame,
    timeperiod: Optional[int] = 1,
    nbdev: Optional[int] = 1,  # Number of deviations
    fillna: Optional[bool] = True,
) -> pd.DataFrame:
    """
    Adds a new column 'std' to the dataframe representing the standard deviation of the close price.
    to capture the volatility in the market.

    Parameters:
        - X (pd.DataFrame): input dataframe
        - timeperiod (int): lookback period for the standard deviation
        - nbdev (int): number of deviations for the standard deviation
        - fillna (bool): whether to fill NaN values with 0
    """
    X_ = X.copy()

    X_['std'] = talib.STDDEV(X_['close'], timeperiod=timeperiod, nbdev=nbdev)

    if fillna:
        X_['std'] = X_['std'].fillna(0)

    return X_


def add_bollinger_bands(
    X: pd.DataFrame,
    timeperiod: Optional[int] = 20,
    nbdev: Optional[int] = 2,
    fillna: Optional[bool] = True,
) -> pd.DataFrame:
    """
    Adds Bollinger Bands to the dataframe using df['close'] column.
    - Bollinger Bands (upper, middle, lower) -> `bb_upper`, `bb_middle`, `bb_lower` columns
    It is a volatility indicator derived from the standard deviation of the close price.
    And used to identify overbought and oversold conditions. Makes use of moving averages and standard deviations.

    Parameters:
        - X (pd.DataFrame): input dataframe
        - timeperiod (int): lookback period for moving average
        - nbdev (int): number of deviations for upper and lower bands
        - fillna (bool): whether to fill NaN values with 0
    """
    X_ = X.copy()

    # Bollinger Bands (upper, middle, lower) -> `bb_upper`, `bb_middle`, `bb_lower` columns
    upper, middle, lower = talib.BBANDS(
        X_['close'], timeperiod=timeperiod, nbdevup=nbdev, nbdevdn=nbdev
    )

    X_['bb_upper'] = upper
    X_['bb_middle'] = middle
    X_['bb_lower'] = lower

    if fillna:
        X_['bb_upper'] = X_['bb_upper'].fillna(0)
        X_['bb_middle'] = X_['bb_middle'].fillna(0)
        X_['bb_lower'] = X_['bb_lower'].fillna(0)

    return X_


def add_macd_indicator(
    X: pd.DataFrame,
    fastperiod: Optional[int] = 12,
    slowperiod: Optional[int] = 26,
    signalperiod: Optional[int] = 9,
    fillna: Optional[bool] = True,
) -> pd.DataFrame:
    """
    Adds MACD (Moving Average Convergence Divergence) to the dataframe using df['close'] column.
    - MACD (macd) -> MACD Line
    - Signal Line (signal) -> 9-day EMA of MACD Line
    - MACD Histogram (hist) -> MACD Line - Signal Line

    Parameters:
        - X (pd.DataFrame): input dataframe
        - fastperiod (int): lookback period for short-term EMA
        - slowperiod (int): lookback period for long-term EMA
        - signalperiod (int): lookback period for signal EMA
        - fillna (bool): whether to fill NaN values with 0
    """
    X_ = X.copy()

    macd, macd_signal, _ = talib.MACD(
        X_['close'],
        fastperiod=fastperiod,
        slowperiod=slowperiod,
        signalperiod=signalperiod,
    )
    X_['macd'] = macd
    X_['macd_signal'] = macd_signal

    if fillna:
        X_['macd'] = X_['macd'].fillna(0)
        X_['macd_signal'] = X_['macd_signal'].fillna(0)

    return X_


def add_last_observed_target(
    X: pd.DataFrame, n_candles_into_future: int
) -> pd.DataFrame:
    """
    Adds the last observed target to the dataframe by shifting the target column n_candles_into_future.

    Parameters:
        - X (pd.DataFrame): input dataframe
        - n_candles_into_future (int): number of candles into the future to shift the target column

    Returns:
        pd.DataFrame: dataframe with additional features
    """
    X_ = X.copy()

    X_['last_observed_target'] = X_['close'].pct_change(n_candles_into_future)

    # The first `n_candles_into_future` rows will have NaN values because there is no historical data
    # to compute the pct_change. We can fill these NaN values with 0.
    X_['last_observed_target'].fillna(0, inplace=True)

    return X_
