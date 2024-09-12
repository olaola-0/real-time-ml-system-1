import pandas as pd


def interpolate_missing_candles(
    ohlc_data: pd.DataFrame, ohlc_window_sec: int
) -> pd.DataFrame:
    """
    Interpolates missing candlesticks in the OHLC data

    Args:
        ohlc_data (pd.DataFrame): OHLC data
        ohlc_window_sec (int): OHLC window size in seconds

    Returns:
        pd.DataFrame: OHLC data with missing candlesticks interpolated
    """
    ohlc_data.set_index('timestamp', inplace=True)

    # Complete list of timestamps for which we need to have rows in the dataset
    from_ms = ohlc_data.index.min()
    to_ms = ohlc_data.index.max()
    labels = range(from_ms, to_ms, ohlc_window_sec * 1000)

    # Reindex the dataframe to add missing rows
    ohlc_data = ohlc_data.reindex(labels)

    # Interpolate missing values using forward fill for close prices
    ohlc_data['close'].ffill(inplace=True)

    # If ohlc_data['open', 'high', 'low'] are missing, fill them with the close price
    ohlc_data['open'].fillna(ohlc_data['close'], inplace=True)
    ohlc_data['high'].fillna(ohlc_data['close'], inplace=True)
    ohlc_data['low'].fillna(ohlc_data['close'], inplace=True)

    # If ohlc_data['volume'] is missing, fill it with 0 and
    # if ohlc_data['vwap'] is missing, fill it with the close price
    ohlc_data['volume'].fillna(0, inplace=True)
    ohlc_data['vwap'].fillna(ohlc_data['close'], inplace=True)

    # Forward fill the product_id
    ohlc_data['product_id'].ffill(inplace=True)

    # Reset the index
    ohlc_data.reset_index(inplace=True)

    # Ensure that there are no missing datetime values
    ohlc_data['datetime'] = pd.to_datetime(ohlc_data['timestamp'], unit='ms')

    return ohlc_data


def create_target_metric(
    ohlc_data: pd.DataFrame, ohlc_window_sec: int, prediction_window_sec: int
) -> pd.DataFrame:
    """
    Creates the target metric as a new column in the dataset by calculating the percentage change in the close price n_candles_into_future.

    Args:
        ohlc_data (pd.DataFrame): OHLC data
        ohlc_window_sec (int): Window in seconds to fetch OHLC data
        prediction_window_sec (int): Size of the prediction window in seconds

    Returns:
        pd.DataFrame: OHLC data with the target metric as a new column
    """
    # Check that prediction_window_sec is a multiple of ohlc_window_sec
    assert (
        prediction_window_sec % ohlc_window_sec == 0
    ), 'prediction_window_sec must be a multiple of ohlc_window_sec'

    # Calculate the number of candles into the future
    n_candles_into_future = prediction_window_sec // ohlc_window_sec

    # Calculate the percentage change in the close price n_candles_into_future
    ohlc_data['close_pct_change'] = ohlc_data['close'].pct_change(n_candles_into_future)

    # Shift the target column by n_candles_into_future to have the target value for the current candle
    ohlc_data['target'] = ohlc_data['close_pct_change'].shift(-n_candles_into_future)

    # Drop the close_pct_change column
    ohlc_data.drop(columns=['close_pct_change'], inplace=True)

    ohlc_data.dropna(subset=['target'], inplace=True)

    return ohlc_data