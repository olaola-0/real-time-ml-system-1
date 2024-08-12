import time
from argparse import ArgumentParser
from typing import Dict, List

import hopsworks
import pandas as pd
from hsfs.client.exceptions import FeatureStoreException
from loguru import logger as logging

from src.config import config

logging.info("Starting Features Dashboard backend")

# Authenticate with the feature store
project = hopsworks.login(
    project=config.hopsworks_project_name,
    api_key_value=config.hopsworks_api_key,
)

feature_store = project.get_feature_store()


def get_feature_view():
    """
    Retrieve the feature view from the feature store
    """ 
    # Get the feature group to read from
    feature_group = feature_store.get_feature_group(
        name=config.feature_group_name,
        version=config.feature_group_version
    )

    # Get (or possibly create) the feature view to read from the feature group
    feature_view = feature_store.get_or_create_feature_view(
        name=config.feature_view_name,
        version=config.feature_view_version,
        query=feature_group.select_all(),
    )

    return feature_view


def get_features_from_store(online_or_offline: str) -> pd.DataFrame:
    """
    Retrieve features from the feature store and returns them as a pandas DataFrame.
    The configuration parameters are read from the config file.
    """
    logging.info(f"Retrieving features from the feature store: {online_or_offline} mode")
    feature_view = get_feature_view()

    # Get all features from the feature group
    if online_or_offline == "offline":
        try:
            features: pd.DataFrame = feature_view.get_batch_data()
        except FeatureStoreException as e:
            logging.error(f"Failed to retrieve features from the feature store: {e}")
            logging.debug('Retrying the operation with "use_hive" option as recommended by Hopsworks')
            features: pd.DataFrame = feature_view.get_batch_data(read_options={"use_hive": True})
    else:
        # Retrieve features from the online feature store and build the list of dictionaries with the primary keys
        features = feature_view.get_feature_vectors(
            entry=get_primary_keys(last_n_minutes=20),
            return_type="pandas"
        )

    # Sort the features by timestamp (ascending)
    features = features.sort_values(by="timestamp")

    return features


def get_primary_keys(last_n_minutes: int) -> List[Dict]:
    """
    Returns a list of dictionaries with the primary keys of the rows to retrieve from the online feature store.
    """
    # Get the current in UTC milliseconds and floor it to the previous minute
    current_utc = int(time.time() * 1000)
    current_utc = current_utc - (current_utc % (last_n_minutes * 60 * 1000))

    # Generate a list of timestamps in milliseconds for the last "last_n_minutes" minutes
    timestamps = [current_utc - i * 60 * 1000 for i in range(last_n_minutes)]

    # Primary keys are pairs of (product_id, timestamp)
    primary_keys = [
        {
            "product_id": config.product_id,
            "timestamp": timestamp,
        } for timestamp in timestamps
    ]
    
    return primary_keys


if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument('--online', action='store_true',)
    parser.add_argument('--offline', action='store_true',)
    args = parser.parse_args()

    if args.online and args.offline:
        logging.error("Please specify either --online or --offline, not both")
    online_or_offline = 'offline' if args.offline else 'online'

    data = get_features_from_store(online_or_offline)
    logging.info(f"Retrieved {len(data)} rows of data from the feature store")
    logging.info(f"Data: {data.head()}")
