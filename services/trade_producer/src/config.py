from typing import List

from dotenv import find_dotenv, load_dotenv
from pydantic_settings import BaseSettings

# Load environment variables from .env file
load_dotenv(find_dotenv())


class Config(BaseSettings):
    kafka_broker_address: str
    kafka_topic_name: str
    product_ids: List[str] = [
        'BTC/USD',
        #'ETH/USD',
        #'ADA/USD',
        #'SOL/USD',
        #'DOT/USD',
        #'LUNA/USD',
        #'AVAX/USD',
        #'DOGE/USD',
        #'SHIB/USD',
        #'UNI/USD',
    ]
    # The mode of operation for the trade producer service (live or historical)
    live_or_historical: str = 'live'
    # The number of days of historical trade data to retrieve
    last_n_days: int = 1


config = Config()
