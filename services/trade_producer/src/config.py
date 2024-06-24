from typing import List, Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    """A class to represent the configuration settings for the trade producer service."""

    kafka_broker_address: Optional[str] = None
    kafka_topic: str
    product_ids: List[str]

    live_or_historical: str

    last_n_days: Optional[int] = 1
    cache_dir_historical_data: Optional[str] = None

    @field_validator('live_or_historical')
    @classmethod
    def validate_live_or_historical(cls, value):
        assert (
            value in {'live', 'historical'}
        ), f'Invalid value for live_or_historical: {value}. Must be either "live" or "historical"'
        return value


config = Config()
