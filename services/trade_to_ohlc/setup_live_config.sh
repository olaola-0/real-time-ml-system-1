# These are the environment variables that are used in the trade_to_ohlc service when running with live data
export KAFKA_INPUT_TOPIC=trade
export KAFKA_OUTPUT_TOPIC=ohlc
export KAFKA_CONSUMER_GROUP=trade_to_ohlc_consumer_group
export OHLC_WINDOW_SECONDS=60