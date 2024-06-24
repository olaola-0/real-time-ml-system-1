from typing import Optional

from pydantic_settings import BaseSettings


class Config(BaseSettings):
    """
    Configuration class for the trade_to_ohlc service

    Attributes:
        kafka_broker_address (str): Kafka broker address
        kafka_input_topic (str): The Kafka topic to consume trade messages from
        kafka_output_topic (str): The Kafka topic to produce OHLC messages to
        kafka_consumer_group (str): The Kafka consumer group to use for consuming trade messages
        ohlc_window_seconds (int): The window size in seconds for the OHLC aggregation window  (e.g. 60 for 1 minute OHLC)
    """

    kafka_broker_address: Optional[str] = None
    kafka_input_topic: str
    kafka_output_topic: str
    kafka_consumer_group: str
    ohlc_window_seconds: int


config = Config()
