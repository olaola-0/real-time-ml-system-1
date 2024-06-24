"""Description: Main entry point for the trade producer service."""

# Gets msgs from a websocket and sends them to redpanda topic
from typing import List

from loguru import logger as logging
from quixstreams import Application

from src.config import config
from src.kraken_api.rest import KrakenRestAPIMultipleProducts
from src.kraken_api.trade import Trade
from src.kraken_api.websocket import KrakenWebsocketTradeAPI


def produce_trades(
    kafka_broker_address: str,
    kafka_topic_name: str,
    product_ids: List[str],
    live_or_historical: str,
    last_n_days: int,
) -> None:
    """
    Reads trades from Kraken websocket API and saves them to a Kafka topic.

    Args:
        kafka_broker_address (str): The address of the Kafka broker.
        kafka_topic_name (str): The name of the Kafka topic.
        product_ids (List[str]): List of product IDs to subscribe to.
        live_or_historical (str): The mode of operation for the trade producer service (live or historical).
        last_n_days (int): The number of days of historical trade data to retrieve.
    Returns:
        None
    """
    # Create an application. The application is the main entry point for interacting with the QuixStreams API
    app = Application(broker_address=kafka_broker_address)

    # The topic to which the trade events will be sent
    topic = app.topic(name=kafka_topic_name, value_serializer='json')

    logging.info(f'Starting trade producer service for products: {product_ids}')

    # Create an instance of the KrakenWebsocketTradeAPI class if the mode of operation is live
    if live_or_historical == 'live':
        kraken_api = KrakenWebsocketTradeAPI(product_ids=product_ids)
    else:
        # Create an instance of the KrakenRestAPI class if the mode of operation is historical
        kraken_api = KrakenRestAPIMultipleProducts(
            product_ids=product_ids,
            last_n_days=last_n_days,
            n_threads=1,  # Use a single thread to avoid rate limiting
            cache_dir=config.cache_dir_historical_data,
        )

    # Create a Producer instance
    logging.info('Creating producer instance...')
    with app.get_producer() as producer:
        while True:
            # Check if all historical trades have been produced
            if kraken_api.is_done():
                logging.info('All historical trades have been produced.')
                break

            try:
                # Get the trades from the Kraken API
                trades_response: List[Trade] = kraken_api.get_trades()
                # logging.info(f'Received {len(trades_response)} trades from Kraken API.')

                # Iterate over the trades and send them to the Kafka topic
                for trade in trades_response:
                    # Serialize an event using the defined Topic
                    message = topic.serialize(
                        key=trade.product_id, value=trade.model_dump()
                    )

                    # Produce a message into the Kafka topic
                    producer.produce(
                        topic=topic.name, value=message.value, key=message.key
                    )
                    logging.info(f'Produced {trade.product_id}: {trade}')
                # sleep(1)
            except Exception as e:
                # Log the full error message that occurs during the trade production process
                logging.error(f'Error producing trades: {str(e)}')


if __name__ == '__main__':
    #
    logging.debug('Configuration:')
    logging.debug(config.model_dump)

    # Start the trade producer service
    try:
        produce_trades(
            kafka_broker_address=config.kafka_broker_address,
            kafka_topic_name=config.kafka_topic,
            product_ids=config.product_ids,
            live_or_historical=config.live_or_historical,
            last_n_days=config.last_n_days,
        )
    except KeyboardInterrupt:
        logging.info('Trade producer service stopped.')
