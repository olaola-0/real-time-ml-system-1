import json
import logging
import time
from datetime import datetime, timezone
from typing import List

import websocket

from src.kraken_api.trade import Trade

# Configure logging
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'
)


class KrakenWebsocketTradeAPI:
    """
    The KrakenWebsocketTradeAPI class provides a websocket client to connect to the Kraken API and subscribe to trade updates for a given product ID.
    Attributes:
        URL (str): The URL of the Kraken websocket API.
        product_ids (List[str]): A list of product IDs to subscribe to.
        _ws (websocket.WebSocket): The websocket connection to the Kraken API.
    """

    # The URL of the Kraken websocket API
    URL = 'wss://ws.kraken.com/v2'

    def __init__(self, product_ids: List[str]):
        self.product_ids = product_ids
        self._connect()

    def _connect(self):
        """
        Establishes a connection to the Kraken websocket API and subscribes to the trades channel.
        This method attempts to establish a connection to the Kraken websocket API and subscribes to the trades channel
        for the specified product_id. If the connection fails, it will retry after a 5-second delay.
        Raises:
            Exception: If an error occurs while connecting to the Kraken websocket API.
        """
        try:
            # Establish a connection to the Kraken websocket API
            self._ws = websocket.create_connection(self.URL)
            logging.info(f'Connected to {self.URL}')
            # Subscribe to the trades channel for the specified product_id
            self._subscribe(self.product_ids)

        except Exception as e:
            logging.error(f'Error connecting to {self.URL}: {e}')
            time.sleep(5)  # Wait for 5 seconds before retrying
            self._connect()

    def _subscribe(self, product_ids: List[str]):
        """Subscribe to trade updates for the specified product IDs.
        Args:
            product_ids (List[str]): A list of product IDs to subscribe to.
        Raises:
            Exception: If there is an error subscribing to trades.
        """
        try:
            logging.info(f'Subscribing to {", ".join(product_ids)} trades...')
            # Create a subscription message
            msg = {
                'method': 'subscribe',
                'params': {
                    'channel': 'trade',
                    'symbol': product_ids,
                    'snapshot': False,
                },
            }
            # Send the subscription message to the websocket
            self._ws.send(json.dumps(msg))
            logging.info(f'Subscribed to {", ".join(product_ids)} trades.')

            # Dumping the first two messages received from the websocket as they are not trade messages
            _ = self._ws.recv()
            _ = self._ws.recv()

        except Exception as e:
            logging.error(f'Error subscribing to trades: {e}')
            self._ws.close()
            time.sleep(5)  # Delay before retrying
            self._connect()

    def get_trades(self) -> List[Trade]:
        """Retrieve the latest trades from the Kraken API.

        Returns:
            List[Dict]: A list of dictionaries representing the trades. Each dictionary contains the following fields:
                - 'product_id': The symbol of the product.
                - 'price': The price of the trade.
                - 'volume': The volume of the trade.
                - 'timestamp': The timestamp of the trade.
        Raises:
            WebSocketConnectionClosedException: If the websocket connection is closed.
            Exception: If there is an error receiving the message.
        """
        try:
            # Receive a message from the websocket
            message = self._ws.recv()

        except websocket.WebSocketConnectionClosedException:
            logging.warning('Websocket connection lost. Reconnecting...')
            self._ws.close()  # Ensure the old socket is closed
            self._connect()
            return []
        except Exception as e:
            logging.error(f'Error receiving message: {e}')
            return []

        if 'heartbeat' in message:
            return []

        # Parse the message string to a dictionary
        message_dict = json.loads(message)

        # Only process trade update messages
        if (
            message_dict.get('channel') == 'trade'
            and message_dict.get('type') == 'update'
        ):
            trades = []
            for trade in message_dict['data']:
                # Transform the timestamp from Kraken, which is in string format, to a Unix timestamp
                timestamp_ms = self.to_ms(trade['timestamp'])

                trades.append(
                    Trade(
                        product_id=trade['symbol'],
                        price=float(trade['price']),
                        volume=float(trade['qty']),
                        timestamp_ms=timestamp_ms,
                    )
                )
            return trades  
        else:
            return []

    def is_done(self) -> bool:
        """The websocket connection is always open for live data."""
        return False

    @staticmethod
    def to_ms(timestamp: str) -> int:
        """
        Transforms a timestamp expressed as a string like '2024-06-17T09:36:39.467866Z'
        to a timestamp expressed in milliseconds since the epoch.
        Args:
            timestamp (str): The timestamp in Kraken format.
        Returns:
            int: A timestamp expressed in milliseconds since the epoch.
        """
        timestamp = datetime.fromisoformat(timestamp[:-1]).replace(tzinfo=timezone.utc)
        return int(timestamp.timestamp() * 1000)
