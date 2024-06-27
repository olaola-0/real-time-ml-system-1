from pydantic_settings import BaseSettings


class Config(BaseSettings):

    product_id: str
    
    # Feature group the feature view reads from
    feature_group_name: str
    feature_group_version: int

    # Name and version of the feature view
    feature_view_name: str
    feature_view_version: int

    # Name and api_key of the project in Hopsworks
    hopsworks_project_name: str
    hopsworks_api_key: str


config = Config()