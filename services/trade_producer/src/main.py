"""Description: Main entry point for the trade producer service."""

# Gets msgs from a websocket and sends them to redpanda topic
from quixstreams import Application
from typing import Dict, List
from time import sleep
import logging
from src import config


# Configure logging for the application to display INFO level messages 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import the KrakenWebsocketTradeAPI class from the kraken_api module
from src.kraken_api import KrakenWebsocketTradeAPI


def produce_trades(kafka_broker_address: str, kafka_topic_name: str, product_ids: List[str]) -> None:
    """
    Reads trades from Kraken websocket API and saves them to a Kafka topic.
    
    Args:
        kafka_broker_address (str): The address of the Kafka broker.
        kafka_topic_name (str): The name of the Kafka topic.
        product_ids (List[str]): List of product IDs to subscribe to.
    Returns:
        None
    """
    logging.info(f"Starting trade producer service for products: {product_ids}")
    # Create an application. The application is the main entry point for interacting with the QuixStreams API
    app = Application(broker_address=kafka_broker_address)

    # The topic to which the trade events will be sent
    topic = app.topic(name=kafka_topic_name, value_serializer="json")

    # Create an instance of the KrakenWebsocketTradeAPI class
    kraken_api = KrakenWebsocketTradeAPI(product_ids=product_ids)

    # Create a Producer instance
    logging.info("Creating producer instance...")
    with app.get_producer() as producer:
        while True:
            try:
                # Get the trades from the Kraken API
                trades: List[Dict] = kraken_api.get_trades()
                logging.info(f"Received {len(trades)} trades from Kraken API.")

                # Iterate over the trades and send them to the Kafka topic
                for trade in trades:
                    # Serialize an event using the defined Topic 
                    message = topic.serialize(key=trade["product_id"], value=trade)

                    # Produce a message into the Kafka topic
                    producer.produce(topic=topic.name,value=message.value, key=message.key)
                    logging.info(f"Message sent to Kafka topic: {kafka_topic_name}!")
                sleep(1)
            except Exception as e:
                # Log any errors that occur during the trade production process
                logging.error(f"Error producing trades: {e}")
                

if __name__ == "__main__":

    # Start the trade producer service
    produce_trades(kafka_broker_address=config.kafka_broker_address,
                   kafka_topic_name=config.kafka_topic_name,
                   product_ids=config.product_ids)