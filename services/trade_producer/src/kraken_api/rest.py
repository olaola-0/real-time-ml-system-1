import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from loguru import logger as logging

from src.kraken_api.trade import Trade


class KrakenRestAPIMultipleProducts:
    def __init__(
        self,
        product_ids: List[str],
        last_n_days: int,
        n_threads: Optional[int] = 1,
        cache_dir: Optional[str] = None,
    ) -> None:
        self.product_ids = product_ids

        self.kraken_apis = [
            KrakenRestAPI(
                product_id=product_id, last_n_days=last_n_days, cache_dir=cache_dir
            )
            for product_id in product_ids
        ]

        self.n_threads = n_threads

    def get_trades(self) -> List[Dict]:
        """
        Get the trade data for each product ID in self.product_ids from the Kraken API in self.kraken_apis.
        and returns a list of trades.

        Args:
            None

        Returns:
            List[Dict]: A list of  dictionaries containing trade data for each product ID in self.product_ids.
        """
        if self.n_threads == 1:
            # Single-threaded / sequential mode
            trades: List[Dict] = []

            for kraken_api in self.kraken_apis:
                # Check if we are done retrieving historical trade data for this product ID
                if kraken_api.is_done():
                    # Skip this product ID
                    continue
                else:
                    trades.extend(kraken_api.get_trades())
        else:
            # Multi-threaded / parallel mode
            with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
                trades = list(
                    executor.map(self.get_trades_for_one_product, self.kraken_apis)
                )
                # trades is a list of lists, so we need to flatten it
                trades = [trade for sublist in trades for trade in sublist]

        return trades

    def get_trades_for_one_product(self, kraken_api: 'KrakenRestAPI') -> List[Trade]:
        """
        Get the trade data for a single product ID from the Kraken API.

        Args:
            kraken_api (KrakenRestAPI): An instance of the KrakenRestAPI class from which to retrieve trade data.

        Returns:
            List[Trade]: A list of trades for the given product ID.
        """
        if not kraken_api.is_done():
            return kraken_api.get_trades()
        return []

    def is_done(self) -> bool:
        """
        Returns True if all the KrakenRestAPI instances in self.kraken_apis are done retrieving historical trade data.

        Args:
            None

        Returns:
            bool: True if all the KrakenRestAPI instances in self.kraken_apis are done retrieving historical trade data.
        """
        for kraken_api in self.kraken_apis:
            if not kraken_api.is_done():
                return False
        return True


class KrakenRestAPI:
    URL = 'https://api.kraken.com/0/public/Trades?pair={product_id}&since={since_sec}'

    def __init__(
        self,
        product_id: str,
        last_n_days: int,
        cache_dir: Optional[str] = None,
    ) -> None:
        """
        Basic initialization of the KrakenRestAPI class.

        Args:
            product_ids (str): A product ID for which to retrieve trade data.
            last_n_days (int): The number of days of historical trade data to retrieve.
            cache_dir (Optional[str]): The directory where the historical trade data will be cached.

        Returns:
            None
        """
        self.product_id = product_id
        self.from_ms, self.to_ms = self._init_from_to_ms(last_n_days)

        logging.debug(
            f'Initializing KrakenRestAPI: from_ms={ts_to_date(self.from_ms)}, to_ms={ts_to_date(self.to_ms)}'
        )

        # The timestamp from which to retrieve trade data.
        # This will be updated after each batch of trades is retrieved from the Kraken API.
        self.last_trade_ms = self.from_ms

        # Are we done retrieving trade data?
        # This will be set to True if the last batch
        # self.is_done = False

        # cahe_dir is the directory where the historical trade data will be stored to speed up service restarts.
        self.use_cache = False
        if cache_dir is not None:
            self.cache_dir = CachedTradeData(cache_dir)
            self.use_cache = True

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

    def get_trades(self) -> List[Trade]:
        """ "
        Fetches a batch of trades from the Kraken REST API API and returns a list of Trade objects.

        Args:
            None

        Returns:
            List[Trade]: A list of dictionaries containing trade data.
        """
        # Replace the placeholders in the URL with the product ID and the timestamp
        # - product_id: The product ID for which to retrieve trade data.
        # - since_ns: The timestamp in nanoseconds from which to retrieve trade data.
        since_ns = self.last_trade_ms * 1_000_000
        payload = {}
        headers = {'Accept': 'application/json'}
        url = self.URL.format(product_id=self.product_id, since_sec=since_ns)
        logging.debug(f'{url= }')

        if self.use_cache and self.cache_dir.has(url):
            # Read the trade data from the cache
            trades = self.cache_dir.read(url)
            logging.debug(
                f'Loaded {len(trades)} trades for {self.product_id}, since={ns_to_date(since_ns)} from cache'
            )
        else:
            # Fetch the trade data from the Kraken REST API
            response = requests.request('GET', url, headers=headers, data=payload)

            # Parse the string response to dictionary
            data = json.loads(response.text)

            # Check if the response contains an error and has a non-empty list value.
            if ('error' in data) and ('EGeneral:Too many requtests' in data['error']):
                # Slow down the requests to the Kraken API
                logging.warning(
                    'Too many requests to Kraken API. Sleeping for 20 seconds.'
                )
                time.sleep(20)

            trades = [
                Trade(
                    price=float(trade[0]),
                    volume=float(trade[1]),
                    timestamp_ms=int(trade[2] * 1000),
                    product_id=self.product_id,
                )
                for trade in data['result'][self.product_id]
            ]

            logging.debug(
                f'Fetched {len(trades)} trades for {self.product_id}, since={ns_to_date(since_ns)}'
            )

            if self.use_cache:
                # Write the trade data to the cache
                self.cache_dir.write(url, trades)
                logging.debug(
                    f'Saved {len(trades)} trades for {self.product_id}, since={ns_to_date(since_ns)} to cache'
                )

            # Slow down the requests rate to the Kraken API
            time.sleep(1)

        if trades[-1].timestamp_ms == self.last_trade_ms:
            # If the last trade timestamp in the batch is the same as the last_trade_ms, increment it by 1 ms
            # to avoid fetching the same trades again, which would result in an infinite loop.
            self.last_trade_ms = trades[-1].timestamp_ms + 1
        else:
            # Update the last_trade_ms to the timestamp of the last trade in the batch
            self.last_trade_ms = trades[-1].timestamp_ms

        # Filter out trades that are outside the requested time range
        trades = [trade for trade in trades if trade.timestamp_ms <= self.to_ms]

        return trades

    def is_done(self) -> bool:
        """Returns True if we are done retrieving historical trade data."""
        return self.last_trade_ms >= self.to_ms


class CachedTradeData:
    """A class to manage the caching of trade data."""

    def __init__(self, cache_dir: str) -> None:
        """Initializes the CachedTradeData class."""
        self.cache_dir = Path(cache_dir)

        if not self.cache_dir.exists():
            # Create the cache directory if it does not exist
            self.cache_dir.mkdir(parents=True)

    def read(self, url: str) -> List[Trade]:
        """Reads the trade data from the cache for the given url."""
        file_path = self._get_file_path(url)

        # Read the cached data from the file
        if file_path.exists():
            data = pd.read_parquet(file_path)
            # Convert the data to a list of Trade objects
            return [Trade(**trade) for trade in data.to_dict(orient='records')]

        return []

    def write(self, url: str, trades: List[Trade]) -> None:
        """Writes the trade data to the cache for the given url."""
        if not trades:
            return

        # Transform the list of Trade objects to a DataFrame
        data = pd.DataFrame([trade.model_dump() for trade in trades])

        # Write the data to a parquet file
        file_path = self._get_file_path(url)
        data.to_parquet(file_path)

    def has(self, url: str) -> bool:
        """Returns True if the cache contains the data for the given url."""
        file_path = self._get_file_path(url)
        return file_path.exists()

    def _get_file_path(self, url: str) -> str:
        """Returns the file path where the trade data will be stored."""
        # Use the given url to generate a unique file name in a deterministic way
        url_hash = hashlib.md5(url.encode()).hexdigest()
        return self.cache_dir / f'{url_hash}.parquet'


def ts_to_date(ts: int) -> str:
    """
    Transforms a timestamp in Unix milliseconds to a human-readable date.

    Args:
        ts (int): A timestamp in Unix milliseconds.

    Returns:
        str: A human-readable date in the format '%Y-%m-%d %H:%M:%S'.
    """
    return datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime(
        '%Y-%m-%d %H:%M:%S'
    )


def ns_to_date(ns: int) -> str:
    """
    Transforms a timestamp in Unix nanoseconds to a human-readable date.

    Args:
        ns (int): A timestamp in Unix nanoseconds.

    Returns:
        str: A human-readable date in the format '%Y-%m-%d %H:%M:%S'.
    """
    return datetime.fromtimestamp(ns / 1_000_000_000, tz=timezone.utc).strftime(
        '%Y-%m-%d %H:%M:%S'
    )
