# These are the environment variables needed to run the trade_producer in historical mode in Quix Cloud
export KAFKA_TOPIC=trade_historical
export PRODUCT_IDS='["BTC/USD"]'
export LIVE_OR_HISTORICAL=historical
export LAST_N_DAYS=90
export CACHE_DIR_HISTORICAL_DATA=/tmp/historical_trade_data