import os
from typing import List, Dict, Any, Tuple, Optional
import time

import pandas as pd
import hopsworks
from hsfs.feature_store import FeatureStore
from hsfs.feature_view import FeatureView
from loguru import logger as logging


class OHLCDataReader:
    """
    A class to read OHLC data from the feature store.
    The Hopsworks credentials are read from the environment variables.
    - HOPSWORKS_PROJECT_NAME
    - HOPSWORKS_API_KEY
    """
    def __init__(
        self,
        ohlc_window_sec: int,
        feature_view_name: str,
        feature_view_version: int,
        feature_group_name: Optional[str] = None,
        feature_group_version: Optional[int] = None,
    ):
        self.ohlc_window_sec = ohlc_window_sec
        self.feature_view_name = feature_view_name
        self.feature_view_version = feature_view_version
        self.feature_group_name = feature_group_name
        self.feature_group_version = feature_group_version

        self._fs = self._get_feature_store()

    
    def _get_primary_keys_to_read_from_online_store(
            self, 
            product_id: str,
            last_n_minutes: int,
    ) -> List[Dict[str, Any]]:
        """
        Get the primary keys to read from the online feature store.

        Args:
            product_id: The product id to read the OHLC data for.
            last_n_minutes: The number of minutes to go back in time.
        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing the primary keys to read the OHLC data.
        """
        timestamp_keys: List[int] =  self._get_timestamp_keys(last_n_minutes=last_n_minutes)
        primary_keys: List[Dict[str, Any]] = [
            {
                "product_id": product_id,
                "timestamp": timestamp_key,
            }
            for timestamp_key in timestamp_keys
        ]
        return primary_keys
    
    
    def read_from_online_store(
        self,
        product_id: str,
        last_n_minutes: Optional[int],
    ) -> pd.DataFrame:
        """
        Reads the OHLC data from the online feature store for the given product_id and last_n_minutes.

        Args:
            product_id (str): The product id to read the OHLC data for.
            last_n_minutes (int): The number of minutes to go back in time.
        Returns:
            pd.DataFrame: The OHLC data.
        """
        # List of primary keys to read from the online store
        primary_keys: List[Dict[str, Any]] = self._get_primary_keys_to_read_from_online_store(
            product_id=product_id,
            last_n_minutes=last_n_minutes,
        )
        logging.debug(f"Primary keys: {primary_keys}")

        feature_view = self._get_feature_view()
        features = feature_view.get_feature_vectors(entry=primary_keys, return_type="pandas")
        # Sort the features by timestamp (ascending)
        features = features.sort_values(by="timestamp").reset_index(drop=True)

        return features
    

    def _get_timestamp_keys(self, last_n_minutes: int) -> List[int]:
        """
        Get the timestamp keys for the last `last_n_minutes` minutes.

        This method calculates the timestamp keys representing the timestamps for the last `last_n_minutes` minutes.
        The timestamp keys are calculated based on the current time and the specified OHLC (Open-High-Low-Close) window
        duration.

        Args:
            last_n_minutes (int): The number of minutes to go back in time.

        Returns:
            List[int]: A list of timestamp keys representing the timestamps for the last `last_n_minutes` minutes.
        """
        to_timestamp_ms = int(time.time() * 1000)
        to_timestamp_ms -= to_timestamp_ms % 60000

        n_candles_per_minute = 60 // self.ohlc_window_sec

        timestamps = [to_timestamp_ms - i * self.ohlc_window_sec * 1000 \
                        for i in range(last_n_minutes * n_candles_per_minute)]
        return timestamps
    
    def _get_feature_view(self) -> FeatureView:
            """
            Get the feature view from the feature store. 

            Returns:
                The feature view object.

            Raises:
                ValueError: If the feature group name and version are required but not provided.
                ValueError: If the feature group name and version in the feature view do not match the arguments.
            """
            if self.feature_group_name is None:
                # Try to get feature view without creating it.
                # If it does not exist, it will raise an exception. Need the feature group information to create it.
                try:
                    return self._fs.get_feature_view(
                        name=self.feature_view_name, version=self.feature_view_version
                    )
                except Exception as e:
                    raise ValueError("The feature group name and version are required if the feature view does not exist.")
                
            # Get the feature group
            feature_group = self._fs.get_feature_group(
                name=self.feature_group_name, version=self.feature_group_version
            )

            # If it does not exist, create the feature view
            feature_view = self._fs.get_or_create_feature_view(
                name=self.feature_view_name,
                version=self.feature_view_version,
                query=feature_group.select_all(),
            )

            # If it already exists, check that the feature group name and version match
            # the ones in self.feature_group_name and self.feature_group_version. Otherwise, raise an error.
            possibly_diff_feature_group = feature_view.get_feature_group().accessible[0]

            if possibly_diff_feature_group.name != self.feature_group_name or \
                possibly_diff_feature_group.version != self.feature_group_version:
                raise ValueError(
                    f"The feature group name and version in the feature view do not match the ones in the arguments."
                )
            return feature_view
    
    def read_from_offline_store(
            self,
            product_id: str,
            last_n_days: int,
    ) -> pd.DataFrame:
        """"
        Reads the OHLC data from the offline feature store for the given product_id and last_n_days.
        """
        to_timestamp_ms = int(time.time() * 1000)
        from_timestamp_ms = to_timestamp_ms - last_n_days * 24 * 60 * 60 * 1000

        feature_view = self._get_feature_view()
        features = feature_view.get_batch_data()

        # Filter the features for the given product_id and time range
        features = features[features["product_id"] == product_id]
        features = features[features["timestamp"] >= from_timestamp_ms]
        features = features[features["timestamp"] <= to_timestamp_ms]
        # Sort the features by timestamp (ascending)
        features = features.sort_values(by="timestamp").reset_index(drop=True)

        return features
    
    @staticmethod
    def _get_feature_store() -> FeatureStore:
        """
        Get the feature store.
        """
        project = hopsworks.login(
            project=os.environ["HOPSWORKS_PROJECT_NAME"],
            api_key_value=os.environ["HOPSWORKS_API_KEY"],
        )

        return project.get_feature_store()
    

if __name__ == '__main__':

    ohlc_data_reader = OHLCDataReader(
        ohlc_window_sec=60,
        feature_view_name="ohlc_feature_view",
        feature_view_version=1,
        feature_group_name="ohlc_feature_group",
        feature_group_version=1,
    )

    # Check if reading from the online store works
    output = ohlc_data_reader.read_from_online_store(
        product_id="BTC/USD",
        last_n_minutes=20,
    )
    logging.debug(f'Live OHLC data: {output}')

    # Check if reading from the offline store works
    output = ohlc_data_reader.read_from_offline_store(
        product_id="BTC/USD",
        last_n_days=90,
    )
    logging.debug(f'Historical OHLC data: {output}')