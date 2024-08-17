import pandas as pd
from loguru import logger as logging
import hopsworks
from hsfs.feature_group import FeatureGroup


class OHLCDataWriter:
    """
    A class to help read OHLC data from the feature store and write it to a file.
    """
    def __init__(
        self,
        hopsworks_project_name: str,
        hopsworks_api_key: str,
        feature_group_name: str,
        feature_group_version: int,
    ):
        self.hopsworks_project_name = hopsworks_project_name
        self.hopsworks_api_key = hopsworks_api_key
        self.feature_group_name = feature_group_name
        self.feature_group_version = feature_group_version

    def write_from_csv(self, csv_file_path: str):
        """
        Write OHLC data from a CSV file to a file.

        Args:
            csv_file_path (str): The path to the CSV file.
        """
        fg = self._get_feature_group()

        # Read the data from the CSV file
        data = pd.read_csv(csv_file_path)

        fg.insert(data, write_options={"start_offline_materialization": True})

    
    def _get_feature_group(self) -> FeatureGroup:
        """
        Get the feature group from the feature store.

        Returns:
            FeatureGroup: The feature group.
        """
        project = hopsworks.login(project=self.hopsworks_project_name, api_key=self.hopsworks_api_key)

        # Get the feature store
        fs = project.get_feature_store()

        fg = fs.get_or_create_feature_group(
            name=self.feature_group_name,
            version=self.feature_group_version,
            description="OHLC data",
            primary_key=["product_id", "timestamp"],
            event_time="timestamp",
            online_enabled=True
        )

        return fg
    

def main(
    hopsworks_project_name: str,
    hopsworks_api_key: str,
    feature_group_name: str,
    feature_group_version: int,
    csv_file_path: str
):
    writer = OHLCDataWriter(
        hopsworks_project_name=hopsworks_project_name,
        hopsworks_api_key=hopsworks_api_key,
        feature_group_name=feature_group_name,
        feature_group_version=feature_group_version
    )
    writer.write_from_csv(csv_file_path)
    logging.info(f"OHLC data from file {csv_file_path} written to feature group {feature_group_name} version {feature_group_version}")

if __name__ == "__main__":
    from fire import Fire
    Fire(main)