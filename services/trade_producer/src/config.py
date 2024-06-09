import os

from dotenv import find_dotenv, load_dotenv
from pydantic_settings import BaseSettings

# Load environment variables from .env file
load_dotenv(find_dotenv())


class Config(BaseSettings):
    kafka_broker_address: str = os.environ['KAFKA_BROKER_ADDRESS']
    kafka_topic_name: str = os.environ['KAFKA_TOPIC_NAME']


config = Config()
# List of product IDs to subscribe to
PRODUCT_IDS = [
    'BTC/USD',
    'ETH/USD',
    'ADA/USD',
    'SOL/USD',
    'DOT/USD',
    'LUNA/USD',
    'AVAX/USD',
    'DOGE/USD',
    'SHIB/USD',
    'UNI/USD',
]
