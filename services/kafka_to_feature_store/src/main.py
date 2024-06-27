import json
import uuid
from datetime import datetime, timezone
from typing import Optional

from loguru import logger as logging
from quixstreams import Application

from src.config import config
from src.hopsworks_api import push_data_to_feature_store


def get_current_utc_sec() -> int:
    """
    Returns the current time in seconds since the epoch in UTC.

    Returns:
        int: The current time in seconds since the epoch in UTC.
    """
    return int(datetime.now(timezone.utc).timestamp())


def kafka_to_feature_store(
    kafka_topic: str,
    kafka_broker_address: str,
    kafka_consumer_group: str,
    feature_group_name: str,
    feature_group_version: int,
    buffer_size: Optional[int],
    live_or_historical: Optional[str] = 'live',
    save_every_n_sec: Optional[int] = 600,
    create_new_consumer_group: Optional[bool] = False,
) -> None:
    """
    Reads OHLC data from the Kafka topic and writes it to HOPSWORKS Feature Store.
    Specifically, it writes the data to a feature group with the given name and version.

    Args:
        kafka_topic (str): The Kafka topic to read from.
        kafka_broker_address (str): The Kafka broker address.
        feature_group_name (str): The name of the feature group to write to.
        feature_group_version (int): The version of the feature group to write to.
        buffer_size (Optional[int]): The number of messages to buffer before writing to the feature store.
        live_or_historical (Optional[str]): Whether to run the service in live or historical mode;
            Live mode writes data to the online feature store, while historical mode writes to the offline feature store.
        save_every_n_sec (Optional[int]): The number of seconds to wait before writing the buffered data to the feature store.
        create_new_consumer_group (Optional[bool]): Whether to create a new consumer group or not.
    Returns:
        None
    """
    
    # Force the application to read from the beginning of the topic:
    # Create a unique consumer group name. Which means that the when the service is restarted, 
    # it will re-process all the messages in the 'kafka_topic' from the beginning.
    if create_new_consumer_group:
        # Generate a unique consumer group name using uuid
        kafka_consumer_group = f'{kafka_consumer_group}_{uuid.uuid4()}'
        logging.info(f'Created a new consumer group: {kafka_consumer_group}')

    # Initialize the application
    app = Application(
        broker_address=kafka_broker_address,
        consumer_group=kafka_consumer_group,
        auto_offset_reset='earliest' if live_or_historical == 'historical' else 'latest',
    )

    # Get the current time in seconds since the epoch in UTC
    last_saved_to_feature_store_ts = get_current_utc_sec()

    # Initialize a buffer to store messages before writing to the feature store
    buffer = []

    # Create a consumer and start a polling loop
    with app.get_consumer() as consumer:
        consumer.subscribe(topics=[kafka_topic])

        while True:
            # Poll for new messages
            message = consumer.poll(1)

            # Calculate the number of seconds since the last time we saved data to the feature store
            sec_since_last_saved = (
                get_current_utc_sec() - last_saved_to_feature_store_ts
            )

            if (message is not None) and message.error():
                # If there is a message, but it is an error, log the error and continue
                logging.error(f'Kafka error: {message.error()}')
                continue

            elif (message is None) and (sec_since_last_saved < save_every_n_sec):
                # If there is no new messages in the input topic and we have not hit the timer limit,
                # Skip and continue polling messages from Kafka.
                logging.debug('No new messages in Kafka topic.')
                logging.debug(
                    f'Last saved to feature store: {sec_since_last_saved} seconds ago. Limit: {save_every_n_sec}.'
                )
                continue

            else:
                # There is a message to process; Parse the message from kafka into a dictionary
                if message is not None:
                    ohlc_data = json.loads(message.value().decode('utf-8'))

                    # Append the data to the buffer
                    buffer.append(ohlc_data)
                    logging.debug(
                        f'Message{ohlc_data} was pushed to buffer. Buffer size: {len(buffer)}'
                    )

                # If the buffer is full or we have hit the timer limit, write the data to the feature store
                if (len(buffer) >= buffer_size) or (sec_since_last_saved >= save_every_n_sec):
                    # If the buffer is not empty, write the data to the feature store
                    if len(buffer) > 0:
                        try:
                            push_data_to_feature_store(
                                feature_group_name=feature_group_name,
                                feature_group_version=feature_group_version,
                                data=buffer,
                                online_or_offline='online'
                                if live_or_historical == 'live'
                                else 'offline',
                            )
                            logging.debug(f'Wrote {len(buffer)} records to feature store.')

                        except Exception as e:
                            logging.error(f'Failed to write data to feature store: {e}')
                            continue

                        # Reset the buffer
                        buffer = []

                        # Update the last saved to feature store timestamp
                        last_saved_to_feature_store_ts = get_current_utc_sec()


if __name__ == '__main__':
    logging.debug(config.model_dump())

    try:
        kafka_to_feature_store(
            kafka_topic=config.kafka_topic,
            kafka_broker_address=config.kafka_broker_address,
            kafka_consumer_group=config.kafka_consumer_group,
            feature_group_name=config.feature_group_name,
            feature_group_version=config.feature_group_version,
            buffer_size=config.buffer_size,
            live_or_historical=config.live_or_historical,
            save_every_n_sec=config.save_every_n_sec,
            create_new_consumer_group=config.create_new_consumer_group,
        )
    except KeyboardInterrupt:
        logging.info('Shutting down Kafka to Feature Store service.')
