import json
import time
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import requests
from loguru import logger as logging


class KrakenRestAPI:
    URL = 'https://api.kraken.com/0/public/Trades?pair={product_id}&since={since_sec}'

    def __init__(self, product_id: str, last_n_days: int) -> None:
        """
        Basic initialization of the KrakenRestAPI class.

        Args:
            product_ids (str): A product ID for which to retrieve trade data.
            last_n_days (int): The number of days of historical trade data to retrieve.

        Returns:
            None
        """
        self.product_id = product_id
        self.from_ms, self.to_ms = self._init_from_to_ms(last_n_days)

        logging.debug(
            f'Initializing KrakenRestAPI: from_ms={self.from_ms}, to_ms={self.to_ms}'
        )

        # The timestamp from which to retrieve trade data.
        # This will be updated after each batch of trades is retrieved from the Kraken API.
        self.last_trade_ms = self.from_ms

        # Are we done retrieving trade data?
        # This will be set to True if the last batch
        self.is_done = False

    @staticmethod
    def _init_from_to_ms(last_n_days: int) -> Tuple[int, int]:
        """
        Returns the from_ms and to_ms timestamps for the given number of days.
        These values are computed using today's date at midnight and the last_n_days value.

        Args:
            last_n_days (int): The number of days of historical trade data to retrieve.

        Returns:
            Tuple[int, int]: A tuple containing the from_ms and to_ms timestamps.
        """
        today_date = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )

        # today_date to milliseconds
        to_ms = int(today_date.timestamp()) * 1000

        # from_ms is last_n_days ago
        from_ms = to_ms - last_n_days * 24 * 60 * 60 * 1000

        return from_ms, to_ms

    def get_trades(self) -> List[Dict]:
        """
        Retrieves a batch of trade data from the Rest API and returns it as a list of dictionaries.

        Args:
            None

        Returns:
            List[Dict]: A list of dictionaries containing trade data.
        """
        payload = {}
        headers = {'Accept': 'application/json'}

        # Replace the placeholders in the URL with the actual values: product_id and since_sec
        # - product_id: The product ID for which to retrieve trade data.
        # - since_sec: The timestamp from which to retrieve trade data.
        since_sec = self.last_trade_ms // 1000
        url = self.URL.format(product_id=self.product_id, since_sec=since_sec)

        response = requests.request('GET', url, headers=headers, data=payload)

        # Parse the response string into a dictionary
        data = json.loads(response.text)

        # Check if the response contains an error
        if ('error' in data) and ('EGeneral:Too many requests' in data['error']):
            # Slow down the requests rate to avoid hitting the rate limit
            logging.warning('Too many requests. Sleeping for 20 seconds...')
            time.sleep(20)


        trades = [
            {
                'price': float(trade[0]),
                'volume': float(trade[1]),
                'time': int(trade[2]),
                'product_id': self.product_id,
            }
            for trade in data['result'][self.product_id]
        ]

        # Filter out trades that are newer than the to_ms timestamp.
        trades = [trade for trade in trades if trade['time'] <= self.to_ms // 1000]

        # Check if we are done retrieving historical trade data.
        last_ts_in_ns = int(data['result']['last'])
        self.last_trade_ms = last_ts_in_ns // 1_000_000

        # If the last trade timestamp is greater than or equal to the to_ms timestamp, we are done.
        self.is_done = self.last_trade_ms >= self.to_ms

        # Log the last trade timestamp
        logging.debug(f'Last trade timestamp: {self.last_trade_ms}')

        return trades

    def is_done(self) -> bool:
        """
        Returns True if we are done retrieving historical trade data.

        Args:
            None

        Returns:
            bool: True if we are done retrieving historical trade data.
        """
        return self.is_done


class KrakenRestAPIMultipleProducts:
    def __init__(self, product_ids: List[str], last_n_days: int) -> None:
        self.product_ids = product_ids

        self.kraken_apis = [
            KrakenRestAPI(product_id=product_id, last_n_days=last_n_days)
            for product_id in product_ids
        ]

    def get_trades(self) -> List[Dict]:
        """
        Get the trade data for each product ID in self.product_ids from the Kraken API in self.kraken_apis.
        and returns a list of trades.

        Args:
            None

        Returns:
            List[Dict]: A list of  dictionaries containing trade data for each product ID in self.product_ids.
        """
        trades: List[Dict] = []

        for kraken_api in self.kraken_apis:
            # Check if we are done retrieving historical trade data for this product ID
            if kraken_api.is_done:
                continue
            else:
                trades.extend(kraken_api.get_trades())

        return trades

    def is_done(self) -> bool:
        """
        Returns True if all the KrakenRestAPI instances in self.kraken_apis are done retrieving historical trade data.

        Args:
            None

        Returns:
            bool: True if all the KrakenRestAPI instances in self.kraken_apis are done retrieving historical trade data.
        """
        return all(kraken_api.is_done for kraken_api in self.kraken_apis)
