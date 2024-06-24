from datetime import timedelta
from typing import Any, List, Optional, Tuple

from loguru import logger as logging
from quixstreams import Application

from src.config import config


def custom_ts_extractor(
    value: Any,
    headers: Optional[List[Tuple[str, bytes]]],
    timestamp: float,
    timestamp_type,
) -> int:
    """
    Specfying custom timestamp extractor to use the timestamp from the trade message payload instead
    of the default Kafka timestamp.

    We want to use the 'timestamp_ms' field from the message value, and not the timestamp of the the message
    that Kafka generates when the message is saved into the topic.
    """
    return value['timestamp_ms']


def trade_to_ohlc(
    kafka_input_topic: str,
    kafka_output_topic: str,
    kafka_broker_address: str,
    kafka_consumer_group: str,
    ohlc_window_seconds: int,
) -> None:
    """
    Reads trades from a Kafka input topic,
    aggregates them into OHLC candles using a specified time window in 'ohlc_window_seconds',
    and saves the OHLC data into a Kafka output topic.

    Args:
        kafka_input_topic (str): The kafka topic to read trade data from.
        kafka_output_topic (str): The kafka topic to write OHLC data to.
        kafka_broker_address (str): The address of the Kafka broker.
        kafka_consumer_group (str): The Kafka consumer group to use for consuming trade messages.
        ohlc_window_seconds (int): The window size in seconds for the OHLC aggregation.
    Returns:
        None
    """

    # This handles all low level communication with Kafka
    app = Application(
        broker_address=kafka_broker_address,
        consumer_group=kafka_consumer_group,
        auto_offset_reset='latest',  # Process messages from the latest offset
    )

    # Specify the input and output topics for this application with the appropriate serializers
    input_topic = app.topic(
        name=kafka_input_topic,
        value_serializer='json',
        timestamp_extractor=custom_ts_extractor,
    )
    output_topic = app.topic(name=kafka_output_topic, value_serializer='json')

    # Create a StreamingDataFrame and connect it to the input topic
    sdf = app.dataframe(topic=input_topic)

    # Define the initial state for the OHLC candle
    def init_ohlc_candle(value: dict) -> dict:
        """
        Initialize the OHLC candle with the first trade data. This function is used as the initializer in the reducer.
        """
        return {
            'open': value['price'],
            'high': value['price'],
            'low': value['price'],
            'close': value['price'],
            'product_id': value['product_id'],
        }

    # Define the reducer function to update the OHLC candle with new trade data
    def update_ohlc_candle(ohlc_candle: dict, trade: dict) -> dict:
        """
        Update the OHLC candle with the new trade data and return the updated candle.

        Args:
            ohlc_candle (dict): The current OHLC candle.
            trade (dict): The new trade data.

        Returns:
            dict: The updated OHLC candle.
        """
        return {
            'open': ohlc_candle['open'],
            'high': max(ohlc_candle['high'], trade['price']),
            'low': min(ohlc_candle['low'], trade['price']),
            'close': trade['price'],
            'product_id': trade['product_id'],
        }

    # Apply transformations to the StreamingDataFrame to aggregate trades into OHLC candles -start
    sdf = sdf.tumbling_window(duration_ms=timedelta(seconds=ohlc_window_seconds))
    sdf = sdf.reduce(reducer=update_ohlc_candle, initializer=init_ohlc_candle).final()

    # Extract the OHLC prices from the value key
    sdf['open'] = sdf['value']['open']
    sdf['high'] = sdf['value']['high']
    sdf['low'] = sdf['value']['low']
    sdf['close'] = sdf['value']['close']
    sdf['product_id'] = sdf['value']['product_id']

    # Add a timestamp column to the OHLC data
    sdf['timestamp'] = sdf['end']

    # keep only the necessary columns
    sdf = sdf[['timestamp', 'open', 'high', 'low', 'close', 'product_id']]

    # Apply transformation to the StreamingDataFrame to aggregate trades into OHLC candles -end

    sdf = sdf.update(logging.info)

    # Write the OHLC data to the output topic
    sdf = sdf.to_topic(output_topic)

    # Start the streaming application
    app.run(sdf)


if __name__ == '__main__':
    trade_to_ohlc(
        kafka_input_topic=config.kafka_input_topic,
        kafka_output_topic=config.kafka_output_topic,
        kafka_broker_address=config.kafka_broker_address,
        kafka_consumer_group=config.kafka_consumer_group,
        ohlc_window_seconds=config.ohlc_window_seconds,
    )
